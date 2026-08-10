package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// settingsGenerator holds settings the way a provider does and records what it
// was told changed.
type settingsGenerator struct {
	mu      sync.Mutex
	store   settings.LLM
	changed []settings.Changed
}

func (g *settingsGenerator) Generate(context.Context, *frames.LLMContext, llm.Emit) error {
	return nil
}

func (g *settingsGenerator) Settings() any { return &g.store }

func (g *settingsGenerator) UpdateSettings(_ context.Context, changed settings.Changed) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.changed = append(g.changed, changed)
	return nil
}

func (g *settingsGenerator) updates() []settings.Changed {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]settings.Changed(nil), g.changed...)
}

func (g *settingsGenerator) temperature() (float64, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.store.Temperature.Value()
}

func runLLM(t *testing.T, svc *llm.Base) (*pipeline.Task, func()) {
	t.Helper()
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

func waitForUpdates(t *testing.T, gen *settingsGenerator, n int) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for len(gen.updates()) < n {
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %d settings updates, saw %d", n, len(gen.updates()))
		}
	}
}

func TestLLMSettingsUpdateReachesTheProvider(t *testing.T) {
	gen := &settingsGenerator{store: settings.LLM{Temperature: settings.Set(0.7)}}
	svc := llm.New("FakeLLM", gen)
	task, stop := runLLM(t, svc)
	defer stop()

	task.QueueFrame(frames.NewLLMUpdateSettingsFrame(&settings.LLM{Temperature: settings.Set(0.2)}))
	waitForUpdates(t, gen, 1)

	changed := gen.updates()[0]
	if !changed.Has("temperature") {
		t.Errorf("changed = %v, want temperature reported", changed)
	}
	if changed["temperature"] != 0.7 {
		t.Errorf("previous temperature = %v, want 0.7", changed["temperature"])
	}
	if v, _ := gen.temperature(); v != 0.2 {
		t.Errorf("stored temperature = %v, want 0.2", v)
	}
}

// The model labels the tokens this service reports and is what they are priced
// against, so a model changed mid-call has to relabel what follows.
func TestLLMSettingsUpdateRelabelsTheModel(t *testing.T) {
	gen := &settingsGenerator{store: settings.LLM{
		Base: settings.Base{Model: settings.Set("old-model")},
	}}
	svc := llm.New("FakeLLM", gen)
	svc.SetModel("old-model")

	var reported []string
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableUsageMetrics:      true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mf, ok := f.(*frames.MetricsFrame)
			if !ok {
				return
			}
			for _, d := range mf.Data {
				if u, ok := d.(frames.LLMUsageMetricsData); ok {
					reported = append(reported, u.MetricsModel())
				}
			}
		},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-done
	}()

	task.QueueFrame(frames.NewLLMUpdateSettingsFrame(&settings.LLM{
		Base: settings.Base{Model: settings.Set("new-model")},
	}))
	waitForUpdates(t, gen, 1)

	if err := svc.PushTokenUsage(context.Background(), frames.LLMTokenUsage{PromptTokens: 1}); err != nil {
		t.Fatalf("push token usage: %v", err)
	}
	deadline := time.After(3 * time.Second)
	for len(reported) == 0 {
		select {
		case <-time.After(5 * time.Millisecond):
		case <-deadline:
			t.Fatal("no usage metrics reported")
		}
	}
	if reported[0] != "new-model" {
		t.Errorf("usage reported against %q, want new-model", reported[0])
	}
}

func TestLLMSettingsUpdateForAnotherServiceIsNotApplied(t *testing.T) {
	gen := &settingsGenerator{store: settings.LLM{Temperature: settings.Set(0.7)}}
	svc := llm.New("FakeLLM", gen)
	task, stop := runLLM(t, svc)
	defer stop()

	f := frames.NewLLMUpdateSettingsFrame(&settings.LLM{Temperature: settings.Set(0.2)})
	f.Service = namedService("SomeOtherLLM")
	task.QueueFrame(f)
	time.Sleep(100 * time.Millisecond)

	if got := gen.updates(); len(got) != 0 {
		t.Errorf("updates = %v, want none: the frame named another service", got)
	}
	if v, _ := gen.temperature(); v != 0.7 {
		t.Errorf("stored temperature = %v, want 0.7", v)
	}
}

// namedService stands in for a different service the frame could name.
type namedService string

func (n namedService) Name() string { return string(n) }

func TestLLMSettingsUpdateSentAsPlainDataIsApplied(t *testing.T) {
	gen := &settingsGenerator{store: settings.LLM{Temperature: settings.Set(0.7)}}
	svc := llm.New("FakeLLM", gen)
	task, stop := runLLM(t, svc)
	defer stop()

	f := frames.NewLLMUpdateSettingsFrame(nil)
	// Numbers that arrived as JSON are floats, whole ones included.
	f.Settings = map[string]any{"temperature": 0.1, "max_tokens": float64(512)}
	task.QueueFrame(f)
	waitForUpdates(t, gen, 1)

	if v, _ := gen.temperature(); v != 0.1 {
		t.Errorf("stored temperature = %v, want 0.1", v)
	}
	gen.mu.Lock()
	tokens, _ := gen.store.MaxTokens.Value()
	gen.mu.Unlock()
	if tokens != 512 {
		t.Errorf("stored max_tokens = %v, want 512", tokens)
	}
}
