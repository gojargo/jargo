package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// metricsGen emits a short response and reports token usage when usage metrics
// are enabled. It embeds *llm.Base so it can call PushTokenUsage and read
// UsageMetricsEnabled, mirroring how a real service is wired.
type metricsGen struct {
	*llm.Base
	usage frames.LLMTokenUsage
}

func (g *metricsGen) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	if err := emit("hello"); err != nil {
		return err
	}
	if g.UsageMetricsEnabled() {
		return g.PushTokenUsage(ctx, g.usage)
	}
	return nil
}

func TestBasePushesTokenUsageWhenEnabled(t *testing.T) {
	gen := &metricsGen{usage: frames.LLMTokenUsage{PromptTokens: 12, CompletionTokens: 3, TotalTokens: 15}}
	gen.Base = llm.New("MetricsLLM", gen)

	var mu sync.Mutex
	var got *frames.LLMTokenUsage
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(gen), pipeline.TaskParams{
		EnableUsageMetrics: true,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.MetricsFrame:
				got = fr.Tokens
			case *frames.LLMFullResponseEndFrame:
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("response did not complete")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if got == nil {
		t.Fatal("no MetricsFrame with token usage reached downstream")
	}
	if got.PromptTokens != 12 || got.CompletionTokens != 3 || got.TotalTokens != 15 {
		t.Fatalf("usage = %+v, want 12 in / 3 out / 15 total", *got)
	}
}

func TestBaseSkipsTokenUsageWhenDisabled(t *testing.T) {
	gen := &metricsGen{usage: frames.LLMTokenUsage{PromptTokens: 1}}
	gen.Base = llm.New("MetricsLLM", gen)

	var mu sync.Mutex
	sawMetrics := false
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(gen), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch f.(type) {
			case *frames.MetricsFrame:
				sawMetrics = true
			case *frames.LLMFullResponseEndFrame:
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("response did not complete")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if sawMetrics {
		t.Fatal("MetricsFrame emitted though usage metrics were disabled")
	}
}
