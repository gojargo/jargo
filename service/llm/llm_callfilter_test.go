package llm_test

import (
	"context"
	"encoding/json"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
)

// filterGen requests a fixed list of tool calls on its first inference and
// answers with text on any that follow.
type filterGen struct {
	mu    sync.Mutex
	turn  int
	calls []frames.ToolCall
}

func (g *filterGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (g *filterGen) GenerateWithTools(
	_ context.Context, _ *frames.LLMContext, sink llm.Sink,
) error {
	g.mu.Lock()
	turn := g.turn
	g.turn++
	g.mu.Unlock()
	if turn > 0 {
		return sink.Text("done")
	}
	for _, c := range g.calls {
		if err := sink.Tool(c); err != nil {
			return err
		}
	}
	return nil
}

// call builds a tool call with empty arguments.
func call(id, name string) frames.ToolCall {
	return frames.ToolCall{ID: id, Name: name, Args: json.RawMessage(`{}`)}
}

// watch records what a turn announced: the calls of each
// FunctionCallsStartedFrame, the calls that actually started, and the names the
// service reported to the application.
type watch struct {
	mu        sync.Mutex
	announced [][]string
	started   []string
	notified  [][]string
	ends      int
	// mute follows the frames as the mute strategy does, so a turn can be
	// checked for leaving the user muted with nothing left running.
	mute  *turns.FunctionCallUserMute
	muted bool
}

func (w *watch) see(f frames.Frame) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.muted = w.mute.ShouldMute(f)
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		w.announced = append(w.announced, names(fr.Calls))
	case *frames.FunctionCallInProgressFrame:
		w.started = append(w.started, fr.ToolName)
	case *frames.LLMFullResponseEndFrame:
		w.ends++
	}
}

func names(calls []frames.ToolCall) []string {
	out := make([]string, 0, len(calls))
	for _, c := range calls {
		out = append(out, c.Name)
	}
	return out
}

// runTurn drives one inference through svc and returns what the turn announced.
// It waits for the response to be bracketed, then settles: the calls run off the
// frame path, so their frames arrive after the response has ended.
func runTurn(t *testing.T, svc *llm.Base, tools ...string) *watch {
	t.Helper()

	w := &watch{mute: turns.NewFunctionCallUserMute()}
	svc.OnFunctionCallsStarted(func(_ context.Context, calls []frames.ToolCall) {
		w.mu.Lock()
		w.notified = append(w.notified, names(calls))
		w.mu.Unlock()
	})

	convo := frames.NewLLMContext("be brief")
	declared := make([]frames.Tool, 0, len(tools))
	for _, name := range tools {
		declared = append(declared, frames.Tool{Name: name, Description: name})
	}
	convo.SetTools(declared)
	convo.AddUserMessage("go on then")

	task := pipeline.NewTask(pipeline.New(svc, newProbe(w.see)), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		w.mu.Lock()
		done := w.ends > 0
		w.mu.Unlock()
		if done {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	// Long enough for anything the turn set going to have been seen, and cheap
	// when it has: a call that is going to start has already been queued.
	time.Sleep(100 * time.Millisecond)

	task.StopWhenDone()
	<-runDone
	return w
}

// TestFunctionCallFilterKeepsOnlyWhatItReturns checks the filter decides the
// whole of what a response does: the dropped call runs no handler, and neither
// the frame nor the application's notification mentions it.
func TestFunctionCallFilterKeepsOnlyWhatItReturns(t *testing.T) {
	gen := &filterGen{calls: []frames.ToolCall{call("c1", "keep"), call("c2", "drop")}}

	var sawConvo bool
	filter := func(convo *frames.LLMContext, calls []frames.ToolCall) []frames.ToolCall {
		sawConvo = convo != nil
		return slices.DeleteFunc(slices.Clone(calls), func(c frames.ToolCall) bool {
			return c.Name == "drop"
		})
	}
	svc := llm.New("FakeToolLLM", gen, llm.WithFunctionCallFilter(filter))

	var mu sync.Mutex
	var ran []string
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		mu.Lock()
		ran = append(ran, p.FunctionName)
		mu.Unlock()
		return p.Result(ctx, "ok", nil)
	}
	svc.RegisterFunction("keep", handler)
	svc.RegisterFunction("drop", handler)

	w := runTurn(t, svc, "keep", "drop")

	if !sawConvo {
		t.Error("the filter was not given the conversation the calls were made in")
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(ran, []string{"keep"}) {
		t.Errorf("handlers run = %v, want only the kept call", ran)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.announced) != 1 || !slices.Equal(w.announced[0], []string{"keep"}) {
		t.Errorf("FunctionCallsStartedFrame calls = %v, want one frame naming only the kept call", w.announced)
	}
	if len(w.notified) != 1 || !slices.Equal(w.notified[0], []string{"keep"}) {
		t.Errorf("OnFunctionCallsStarted saw %v, want only the kept call", w.notified)
	}
	if !slices.Equal(w.started, []string{"keep"}) {
		t.Errorf("calls started = %v, want only the kept call", w.started)
	}
}

// TestFunctionCallFilterDroppingEverythingAnnouncesNothing checks a response
// whose calls are all dropped is, to everything downstream, a response that
// requested none: no announcement, no handler, and the response still ends.
func TestFunctionCallFilterDroppingEverythingAnnouncesNothing(t *testing.T) {
	gen := &filterGen{calls: []frames.ToolCall{call("c1", "drop"), call("c2", "drop")}}
	svc := llm.New("FakeToolLLM", gen, llm.WithFunctionCallFilter(
		func(*frames.LLMContext, []frames.ToolCall) []frames.ToolCall { return nil },
	))

	var mu sync.Mutex
	ran := 0
	svc.RegisterFunction("drop", func(ctx context.Context, p llm.FunctionCallParams) error {
		mu.Lock()
		ran++
		mu.Unlock()
		return p.Result(ctx, "ok", nil)
	})

	w := runTurn(t, svc, "drop")

	mu.Lock()
	defer mu.Unlock()
	if ran != 0 {
		t.Errorf("handler ran %d times, want none", ran)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.announced) != 0 {
		t.Errorf("FunctionCallsStartedFrame calls = %v, want no announcement", w.announced)
	}
	if len(w.notified) != 0 {
		t.Errorf("OnFunctionCallsStarted saw %v, want nothing", w.notified)
	}
	if w.ends != 1 {
		t.Errorf("response-end frames = %d, want the response bracketed once", w.ends)
	}
}

// TestCancelAsyncToolIsNotAnnounced checks the built-in cancellation tool is
// kept out of what the application is told: it is how the model abandons an
// asynchronous call, not a tool the application put up. It still runs.
func TestCancelAsyncToolIsNotAnnounced(t *testing.T) {
	t.Run("alone", func(t *testing.T) {
		gen := &filterGen{calls: []frames.ToolCall{call("c1", llm.CancelAsyncToolName)}}
		svc := llm.New("FakeToolLLM", gen, llm.WithAsyncToolCancellation())
		svc.RegisterFunction("watch", func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "watching", nil)
		}, llm.WithCancelOnInterruption(false))

		w := runTurn(t, svc, "watch")

		w.mu.Lock()
		defer w.mu.Unlock()
		if len(w.announced) != 0 {
			t.Errorf("FunctionCallsStartedFrame calls = %v, want no announcement", w.announced)
		}
		if len(w.notified) != 0 {
			t.Errorf("OnFunctionCallsStarted saw %v, want nothing", w.notified)
		}
		if !slices.Contains(w.started, llm.CancelAsyncToolName) {
			t.Errorf("calls started = %v, want the cancellation to have run", w.started)
		}
		// Leaving the announcement out means the mute strategy sees a result for
		// a call it never tracked, which must leave it unmuted rather than stuck.
		if w.muted {
			t.Error("the user was left muted by a call that was never announced")
		}
	})

	t.Run("alongside a tool of the application's", func(t *testing.T) {
		gen := &filterGen{calls: []frames.ToolCall{
			call("c1", llm.CancelAsyncToolName),
			call("c2", "watch"),
		}}
		svc := llm.New("FakeToolLLM", gen, llm.WithAsyncToolCancellation())
		svc.RegisterFunction("watch", func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, "watching", nil)
		}, llm.WithCancelOnInterruption(false))

		w := runTurn(t, svc, "watch")

		w.mu.Lock()
		defer w.mu.Unlock()
		if len(w.announced) != 1 || !slices.Equal(w.announced[0], []string{"watch"}) {
			t.Errorf("FunctionCallsStartedFrame calls = %v, want only the application's tool", w.announced)
		}
		if len(w.notified) != 1 || !slices.Equal(w.notified[0], []string{"watch"}) {
			t.Errorf("OnFunctionCallsStarted saw %v, want only the application's tool", w.notified)
		}
		if !slices.Contains(w.started, llm.CancelAsyncToolName) {
			t.Errorf("calls started = %v, want the cancellation to have run as well", w.started)
		}
	})
}
