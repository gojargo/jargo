package pipeline_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// waitFor blocks until cond holds, failing the test with what it was waiting
// for if it does not hold in time.
func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// fakeLLM records what the switcher asks of it. It stands in for a language
// model service without needing a provider.
type fakeLLM struct {
	*processor.Base

	mu         sync.Mutex
	synced     []string
	registered []string
}

func newFakeLLM(name string) *fakeLLM {
	l := &fakeLLM{}
	l.Base = processor.New(name, l)
	return l
}

func (l *fakeLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := l.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return l.PushFrame(ctx, f, dir)
}

func (l *fakeLLM) SyncToolHandlers(_ context.Context, tools []frames.Tool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, t := range tools {
		l.synced = append(l.synced, t.Name)
	}
}

func (l *fakeLLM) RegisterFunction(name string, _ llm.FunctionCallHandler, _ ...llm.RegisterOption) {
	l.mu.Lock()
	l.registered = append(l.registered, name)
	l.mu.Unlock()
}

func (l *fakeLLM) syncedTools() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.synced...)
}

func (l *fakeLLM) registeredTools() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.registered...)
}

// TestLLMSwitcherSyncsToolsOnEveryMember checks the tools a conversation
// advertises reach every member, not just the one in use. Only the active
// service is sent the conversation, so an inactive one would otherwise be out
// of step with what the model is told it can call the moment it takes over.
func TestLLMSwitcherSyncsToolsOnEveryMember(t *testing.T) {
	a, b := newFakeLLM("A"), newFakeLLM("B")
	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewLLMSwitcher: %v", err)
	}

	task, _, stop := runCollector(t, sw)
	defer stop()

	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{{Name: "book_table"}})
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	waitFor(t, func() bool {
		return len(a.syncedTools()) > 0 && len(b.syncedTools()) > 0
	}, "both members to be synced with the advertised tools")

	for _, l := range []*fakeLLM{a, b} {
		got := l.syncedTools()
		if len(got) == 0 || got[0] != "book_table" {
			t.Errorf("%s synced tools = %v, want [book_table]", l.Name(), got)
		}
	}
}

// TestLLMSwitcherRegistersOnEveryMember checks a handler registered on the
// switcher reaches every member, so a tool keeps working across a switch.
func TestLLMSwitcherRegistersOnEveryMember(t *testing.T) {
	a, b := newFakeLLM("A"), newFakeLLM("B")
	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewLLMSwitcher: %v", err)
	}

	sw.RegisterFunction("book_table", func(context.Context, llm.FunctionCallParams) error { return nil })

	for _, l := range []*fakeLLM{a, b} {
		got := l.registeredTools()
		if len(got) != 1 || got[0] != "book_table" {
			t.Errorf("%s registered = %v, want [book_table]", l.Name(), got)
		}
	}
}

// TestLLMSwitcherReportsTheActiveLLM checks the switcher tracks which model is
// answering, and that switching moves it.
func TestLLMSwitcherReportsTheActiveLLM(t *testing.T) {
	a, b := newFakeLLM("A"), newFakeLLM("B")
	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{a, b}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewLLMSwitcher: %v", err)
	}

	if got := sw.ActiveLLM(); got != a {
		t.Errorf("ActiveLLM() = %v, want the first member", got)
	}
	if !sw.SwitchTo(b) {
		t.Fatal("SwitchTo(b) = false, want the switch accepted")
	}
	if got := sw.ActiveLLM(); got != b {
		t.Errorf("ActiveLLM() after the switch = %v, want b", got)
	}
	if got := len(sw.LLMs()); got != 2 {
		t.Errorf("LLMs() = %d members, want 2", got)
	}
}

// errProviderRefused stands in for whatever a provider reports when it will
// not answer.
//
//nolint:gochecknoglobals // sentinel error for the test below
var errProviderRefused = errors.New("provider refused")

// inferringLLM is a member that can also answer off to the side of the
// pipeline, recording what it was asked and returning a canned answer.
type inferringLLM struct {
	*fakeLLM

	answer string
	err    error

	mu   sync.Mutex
	got  *frames.LLMContext
	opts llm.InferenceOptions
}

func newInferringLLM(name, answer string, err error) *inferringLLM {
	l := &inferringLLM{answer: answer, err: err}
	l.fakeLLM = &fakeLLM{}
	l.fakeLLM.Base = processor.New(name, l)
	return l
}

func (l *inferringLLM) RunInference(
	_ context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	l.mu.Lock()
	l.got, l.opts = convo, opts
	l.mu.Unlock()
	return l.answer, l.err
}

func (l *inferringLLM) asked() (*frames.LLMContext, llm.InferenceOptions) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.got, l.opts
}

// TestLLMSwitcherRunInference checks the off-pipeline answer goes to whichever
// model is active, and that a member that cannot answer that way says so rather
// than failing.
func TestLLMSwitcherRunInference(t *testing.T) {
	answering := newInferringLLM("Answering", "the answer", nil)
	plain := newFakeLLM("Plain")

	sw, err := pipeline.NewLLMSwitcher(
		[]pipeline.LLMMember{answering, plain}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewLLMSwitcher: %v", err)
	}

	convo := frames.NewLLMContext("you are helpful")
	opts := llm.InferenceOptions{MaxTokens: 64, SystemInstruction: "answer briefly"}

	out, ok, err := sw.RunInference(context.Background(), convo, opts)
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if !ok {
		t.Fatal("RunInference reported the active model cannot answer off-pipeline")
	}
	if out != "the answer" {
		t.Errorf("RunInference = %q, want the answer", out)
	}
	gotConvo, gotOpts := answering.asked()
	if gotConvo != convo {
		t.Error("the model was given a different conversation")
	}
	if gotOpts != opts {
		t.Errorf("the model was given options %+v, want %+v", gotOpts, opts)
	}

	// Switching moves the inference to the model now answering, which here
	// cannot answer off-pipeline at all.
	if !sw.SwitchTo(plain) {
		t.Fatal("SwitchTo(plain) = false, want the switch accepted")
	}
	out, ok, err = sw.RunInference(context.Background(), convo, opts)
	if err != nil {
		t.Fatalf("RunInference after the switch: %v", err)
	}
	if ok {
		t.Errorf("RunInference = (%q, true), want false for a model that cannot answer", out)
	}
	if out != "" {
		t.Errorf("RunInference = %q, want no text when the model cannot answer", out)
	}
}

// TestLLMSwitcherRunInferenceReportsAnError checks a failure from the model is
// passed back rather than swallowed, and still reports that the model tried.
func TestLLMSwitcherRunInferenceReportsAnError(t *testing.T) {
	failing := newInferringLLM("Failing", "", errProviderRefused)

	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{failing}, pipeline.NewManualStrategy)
	if err != nil {
		t.Fatalf("NewLLMSwitcher: %v", err)
	}

	_, ok, err := sw.RunInference(
		context.Background(), frames.NewLLMContext(""), llm.InferenceOptions{})
	if !ok {
		t.Error("RunInference reported the model cannot answer, want it tried and failed")
	}
	if !errors.Is(err, errProviderRefused) {
		t.Errorf("RunInference error = %v, want the provider's", err)
	}
}
