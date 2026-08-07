package flows

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// fakeLLM records the handlers a FlowManager registers so a test can invoke them
// directly, standing in for the LLM service's tool loop.
type fakeLLM struct {
	funcs map[string]llm.FunctionCallHandler
}

func (f *fakeLLM) RegisterFunction(name string, h llm.FunctionCallHandler, _ ...llm.RegisterOption) {
	if f.funcs == nil {
		f.funcs = make(map[string]llm.FunctionCallHandler)
	}
	f.funcs[name] = h
}

// call runs a registered handler the way the service would and reports what it
// delivered through the result callback: the result, the properties it set, and
// whether it reported at all.
func (f *fakeLLM) call(
	t *testing.T, name string,
) (result string, props *frames.FunctionCallResultProperties, err error) {
	t.Helper()
	reported := false
	h, ok := f.funcs[name]
	if !ok {
		t.Fatalf("no handler registered for %q", name)
	}
	err = h(context.Background(), llm.FunctionCallParams{
		FunctionName: name,
		ToolCallID:   "call-1",
		Result: func(_ context.Context, r string, p *frames.FunctionCallResultProperties) error {
			reported, result, props = true, r, p
			return nil
		},
	})
	if err == nil && !reported {
		t.Fatalf("handler %q returned without reporting a result", name)
	}
	return result, props, err
}

// fakeEnq records the frames a FlowManager queues.
type fakeEnq struct {
	queued []frames.Frame
}

func (e *fakeEnq) QueueFrame(f frames.Frame) { e.queued = append(e.queued, f) }

func newManager(t *testing.T) (*FlowManager, *fakeLLM, *fakeEnq, *frames.LLMContext) {
	t.Helper()
	fake := &fakeLLM{}
	enq := &fakeEnq{}
	convo := frames.NewLLMContext("")
	fm, err := New(Config{LLM: fake, Context: convo, Enqueuer: enq})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return fm, fake, enq, convo
}

func stay(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
	return "ok", nil, nil
}

func TestNewRejectsMissingRefs(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("expected an error for an empty config")
	}
}

func TestInitializeAppliesNodeAndTriggersRun(t *testing.T) {
	fm, fake, enq, convo := newManager(t)
	node := &NodeConfig{
		Name:         "start",
		RoleMessage:  "be nice",
		TaskMessages: []frames.Message{{Role: frames.RoleUser, Text: "greet"}},
		Functions:    []NodeFunction{{Name: "go_next", Handler: stay}},
	}
	if err := fm.Initialize(context.Background(), node); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if got := convo.System(); got != "be nice" {
		t.Errorf("System = %q, want %q", got, "be nice")
	}
	if msgs := convo.Messages(); len(msgs) != 1 || msgs[0].Text != "greet" || msgs[0].Role != frames.RoleUser {
		t.Errorf("messages = %+v, want one 'greet' user message", msgs)
	}
	if tools := convo.Tools(); len(tools) != 1 || tools[0].Name != "go_next" {
		t.Errorf("tools = %+v, want [go_next]", tools)
	}
	if _, ok := fake.funcs["go_next"]; !ok {
		t.Error("handler go_next was not registered")
	}
	if len(enq.queued) != 1 {
		t.Fatalf("queued %d frames, want 1 (LLMRunFrame)", len(enq.queued))
	}
	if _, ok := enq.queued[0].(*frames.LLMRunFrame); !ok {
		t.Errorf("queued frame = %T, want *frames.LLMRunFrame", enq.queued[0])
	}
	if fm.CurrentNode() != "start" {
		t.Errorf("CurrentNode = %q, want start", fm.CurrentNode())
	}
}

func TestInitializeWaitNodeDoesNotTriggerRun(t *testing.T) {
	fm, _, enq, _ := newManager(t)
	node := &NodeConfig{Name: "inbound", RespondImmediately: new(false)}
	if err := fm.Initialize(context.Background(), node); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if len(enq.queued) != 0 {
		t.Errorf("queued %d frames, want 0", len(enq.queued))
	}
}

func TestTransitionSwapsNode(t *testing.T) {
	fm, fake, _, convo := newManager(t)
	next := &NodeConfig{
		Name:        "second",
		RoleMessage: "second persona",
		Functions:   []NodeFunction{{Name: "stay", Handler: stay}},
	}
	advance := func(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
		return "", next, nil
	}
	start := &NodeConfig{Name: "first", Functions: []NodeFunction{{Name: "advance", Handler: advance}}}
	if err := fm.Initialize(context.Background(), start); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	res, _, err := fake.call(t, "advance")
	if err != nil {
		t.Fatalf("advance handler: %v", err)
	}
	if res != acknowledged {
		t.Errorf("result = %q, want %q", res, acknowledged)
	}
	if fm.CurrentNode() != "second" {
		t.Errorf("CurrentNode = %q, want second", fm.CurrentNode())
	}
	if convo.System() != "second persona" {
		t.Errorf("System = %q, want 'second persona'", convo.System())
	}
	if tools := convo.Tools(); len(tools) != 1 || tools[0].Name != "stay" {
		t.Errorf("tools = %+v, want [stay]", tools)
	}
}

func TestTransitionToWaitNodeStopsTurn(t *testing.T) {
	fm, fake, _, _ := newManager(t)
	next := &NodeConfig{Name: "wait", RespondImmediately: new(false)}
	advance := func(_ context.Context, _ json.RawMessage, _ *FlowManager) (string, *NodeConfig, error) {
		return "done", next, nil
	}
	start := &NodeConfig{Name: "first", Functions: []NodeFunction{{Name: "advance", Handler: advance}}}
	if err := fm.Initialize(context.Background(), start); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	res, props, err := fake.call(t, "advance")
	if err != nil {
		t.Fatalf("advance handler: %v", err)
	}
	if res != "done" {
		t.Errorf("result = %q, want 'done'", res)
	}
	// The node entered waits for the user, so the result is recorded without
	// the assistant being asked to say anything about it.
	if props == nil || props.RunLLM == nil || *props.RunLLM {
		t.Errorf("properties = %+v, want RunLLM false", props)
	}
}

func TestDataFunctionKeepsNode(t *testing.T) {
	fm, fake, _, _ := newManager(t)
	start := &NodeConfig{Name: "first", Functions: []NodeFunction{{Name: "note", Handler: stay}}}
	if err := fm.Initialize(context.Background(), start); err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	res, _, err := fake.call(t, "note")
	if err != nil {
		t.Fatalf("note handler: %v", err)
	}
	if res != "ok" {
		t.Errorf("result = %q, want ok", res)
	}
	if fm.CurrentNode() != "first" {
		t.Errorf("CurrentNode = %q, want first (no transition)", fm.CurrentNode())
	}
}
