package observers_test

import (
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// callRecorder collects everything a FunctionCalls observer reports, against a
// clock a test advances rather than waits on.
type callRecorder struct {
	mu     sync.Mutex
	clock  time.Time
	events []observers.FunctionCallEvent
}

func newCallRecorder() *callRecorder {
	return &callRecorder{clock: time.Date(2026, time.September, 13, 12, 0, 0, 0, time.UTC)}
}

func (r *callRecorder) now() time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.clock
}

func (r *callRecorder) wait(d time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.clock = r.clock.Add(d)
}

func (r *callRecorder) record(e observers.FunctionCallEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

func (r *callRecorder) all() []observers.FunctionCallEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]observers.FunctionCallEvent(nil), r.events...)
}

func (r *callRecorder) callKinds() []observers.FunctionCallEventKind {
	all := r.all()
	out := make([]observers.FunctionCallEventKind, 0, len(all))
	for _, e := range all {
		out = append(out, e.Kind)
	}
	return out
}

// at returns the nth event recorded, failing when there are fewer.
func (r *callRecorder) at(t *testing.T, n int) observers.FunctionCallEvent {
	t.Helper()
	all := r.all()
	if len(all) <= n {
		t.Fatalf("events = %d, want more than %d", len(all), n)
	}
	return all[n]
}

// newFunctionCalls builds an observer reporting into the recorder, on its clock.
func newFunctionCalls(r *callRecorder, cfg observers.FunctionCallConfig) *observers.FunctionCalls {
	cfg.Now = r.now
	cfg.OnFunctionCallEvent = r.record
	return observers.NewFunctionCalls(cfg)
}

// testArgs is the arguments every call in these tests was made with.
func testArgs() json.RawMessage { return json.RawMessage(`{"city":"SF"}`) }

// startedCalls is the frame announcing the calls one model response asked for.
func startedCalls(calls ...[2]string) *frames.FunctionCallsStartedFrame {
	tools := make([]frames.ToolCall, 0, len(calls))
	for _, c := range calls {
		tools = append(tools, frames.ToolCall{ID: c[1], Name: c[0], Args: testArgs()})
	}
	return frames.NewFunctionCallsStartedFrame(tools)
}

// inProgress is the frame reporting that a call is running.
func inProgress(toolCallID string, cancelOnInterruption bool) *frames.FunctionCallInProgressFrame {
	return frames.NewFunctionCallInProgressFrame(
		toolCallID, "get_weather", testArgs(), cancelOnInterruption, "group_1")
}

// callResult is the frame carrying what a call returned.
func callResult(toolCallID string) *frames.FunctionCallResultFrame {
	return frames.NewFunctionCallResultFrame(toolCallID, "get_weather", testArgs(), "12 degrees")
}

// TestCallReportedWhenItsExecutionStarts covers the first of the moments: the
// model has asked for the call, which is not yet the call running.
func TestCallReportedWhenItsExecutionStarts(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, startedCalls([2]string{"get_weather", "call_1"}), processor.Downstream)

	e := r.at(t, 0)
	if e.Kind != observers.FunctionCallStarted || e.FunctionName != "get_weather" || e.ToolCallID != "call_1" {
		t.Errorf("event = %+v, want a started get_weather/call_1", e)
	}
}

// TestEveryCallInAResponseIsItsOwnMoment covers the one frame that reports more
// than one moment.
func TestEveryCallInAResponseIsItsOwnMoment(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, startedCalls([2]string{"get_weather", "call_1"}, [2]string{"get_time", "call_2"}), processor.Downstream)

	all := r.all()
	if len(all) != 2 || all[0].ToolCallID != "call_1" || all[1].ToolCallID != "call_2" {
		t.Fatalf("events = %+v, want one per call", all)
	}
}

// TestCallThatGoesInProgressNamesWhenItStarted covers the wait a call can spend
// queued: calls run one at a time unless the service was built to run them in
// parallel, so starting and running are two moments rather than one.
func TestCallThatGoesInProgressNamesWhenItStarted(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	startedAt := r.now()
	push(o, startedCalls([2]string{"get_weather", "call_1"}), processor.Downstream)
	r.wait(900 * time.Millisecond)
	push(o, inProgress("call_1", true), processor.Downstream)

	e := r.at(t, 1)
	if e.Kind != observers.FunctionCallInProgress {
		t.Fatalf("kind = %s, want in progress", e.Kind)
	}
	if !e.StartedAt.Equal(startedAt) {
		t.Errorf("started at %v, want %v", e.StartedAt, startedAt)
	}
	if got, want := e.Timestamp.Sub(e.StartedAt), 900*time.Millisecond; got != want {
		t.Errorf("the call waited %v, want %v", got, want)
	}
}

// TestCallThatNeverRunsIsLeftWhereItStopped covers a call still waiting when the
// conversation moves on: it runs no further, and reports no further.
func TestCallThatNeverRunsIsLeftWhereItStopped(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, startedCalls([2]string{"get_weather", "call_1"}), processor.Downstream)

	if got := r.callKinds(); len(got) != 1 || got[0] != observers.FunctionCallStarted {
		t.Errorf("kinds = %v, want only a start", got)
	}
}

// TestCallDescribesItselfWhenItGoesInProgress covers the moment carrying what
// the call is: the group it runs with, and the arguments it was made with.
func TestCallDescribesItselfWhenItGoesInProgress(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, inProgress("call_1", true), processor.Downstream)

	e := r.at(t, 0)
	if e.GroupID != "group_1" {
		t.Errorf("group = %q, want group_1", e.GroupID)
	}
	if string(e.Arguments) != string(testArgs()) {
		t.Errorf("arguments = %s, want %s", e.Arguments, testArgs())
	}
}

// TestCallThatSettlesNamesWhenItBeganRunning covers the time a call ran reading
// from the record that ends it.
func TestCallThatSettlesNamesWhenItBeganRunning(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, inProgress("call_1", true), processor.Downstream)
	r.wait(1400 * time.Millisecond)
	push(o, callResult("call_1"), processor.Downstream)

	e := r.at(t, 1)
	if e.Kind != observers.FunctionCallCompleted {
		t.Fatalf("kind = %s, want completed", e.Kind)
	}
	if got, want := e.Timestamp.Sub(e.InProgressAt), 1400*time.Millisecond; got != want {
		t.Errorf("the call ran for %v, want %v", got, want)
	}
}

// TestCallTheConversationWaitsOnIsMarkedBlocking covers the distinction a
// timeline needs: a call that survives an interruption is one the conversation
// was never waiting on.
func TestCallTheConversationWaitsOnIsMarkedBlocking(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, inProgress("call_1", true), processor.Downstream)
	push(o, inProgress("call_2", false), processor.Downstream)

	blocking, nonBlocking := r.at(t, 0), r.at(t, 1)
	if blocking.Blocking == nil || !*blocking.Blocking {
		t.Errorf("blocking = %v, want true", blocking.Blocking)
	}
	if nonBlocking.Blocking == nil || *nonBlocking.Blocking {
		t.Errorf("blocking = %v, want false", nonBlocking.Blocking)
	}
}

// TestHandlerThatFailedIsReportedAsAFailure covers the failure reaching the
// record rather than being flattened into an ordinary result.
func TestHandlerThatFailedIsReportedAsAFailure(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, inProgress("call_1", true), processor.Downstream)
	f := callResult("call_1")
	f.Error = "the api is down"
	push(o, f, processor.Downstream)

	e := r.at(t, 1)
	if e.Kind != observers.FunctionCallFailed {
		t.Errorf("kind = %s, want failed", e.Kind)
	}
	if e.Error != "the api is down" {
		t.Errorf("error = %q, want the handler's", e.Error)
	}
}

// TestDeadlineAndInterruptionSettleACallDifferently covers the one thing that
// tells them apart: only a call canceled by its own deadline asks for inference.
func TestDeadlineAndInterruptionSettleACallDifferently(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, inProgress("call_1", true), processor.Downstream)
	timedOut := frames.NewFunctionCallCancelFrame("call_1", "get_weather")
	timedOut.RunLLM = true
	push(o, timedOut, processor.Downstream)

	push(o, inProgress("call_2", true), processor.Downstream)
	push(o, frames.NewFunctionCallCancelFrame("call_2", "get_weather"), processor.Downstream)

	if got, want := r.at(t, 1).Kind, observers.FunctionCallTimedOut; got != want {
		t.Errorf("kind = %s, want %s", got, want)
	}
	if got, want := r.at(t, 3).Kind, observers.FunctionCallCanceled; got != want {
		t.Errorf("kind = %s, want %s", got, want)
	}
}

// TestProgressAlongTheWayDoesNotSettleACall covers the call that does not block:
// it can report before it is done, and only its final result settles it.
func TestProgressAlongTheWayDoesNotSettleACall(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, inProgress("call_1", false), processor.Downstream)

	notFinal := false
	interim := frames.NewFunctionCallResultFrame("call_1", "get_weather", testArgs(), "still looking")
	interim.Properties = &frames.FunctionCallResultProperties{IsFinal: &notFinal}
	push(o, interim, processor.Downstream)

	push(o, callResult("call_1"), processor.Downstream)

	want := []observers.FunctionCallEventKind{observers.FunctionCallInProgress, observers.FunctionCallCompleted}
	if got := r.callKinds(); len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("kinds = %v, want %v", got, want)
	}
}

// TestResultsTravelOnlyWhenAskedFor covers the default: a result is whatever a
// provider decided to return, so it stays out unless it is asked for.
func TestResultsTravelOnlyWhenAskedFor(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, callResult("call_1"), processor.Downstream)
	if got := r.at(t, 0).Result; got != "" {
		t.Errorf("result = %q, want empty by default", got)
	}

	reporting := newCallRecorder()
	yes := true
	o2 := newFunctionCalls(reporting, observers.FunctionCallConfig{IncludeResults: &yes})
	push(o2, callResult("call_1"), processor.Downstream)
	if got := reporting.at(t, 0).Result; got != "12 degrees" {
		t.Errorf("result = %q, want what the call returned", got)
	}
}

// TestArgumentsCanBeLeftOut covers the other half of the choice: arguments
// travel by default, being the reason a call is worth reading at all.
func TestArgumentsCanBeLeftOut(t *testing.T) {
	r := newCallRecorder()
	no := false
	o := newFunctionCalls(r, observers.FunctionCallConfig{IncludeArguments: &no})

	push(o, inProgress("call_1", true), processor.Downstream)

	e := r.at(t, 0)
	if e.FunctionName != "get_weather" {
		t.Errorf("function = %q, want get_weather: the call is still named", e.FunctionName)
	}
	if e.Arguments != nil {
		t.Errorf("arguments = %s, want none", e.Arguments)
	}
}

// TestCallThatBeganBeforeTheObserverSettlesWithoutThatMoment covers the observer
// attached mid-conversation.
func TestCallThatBeganBeforeTheObserverSettlesWithoutThatMoment(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, callResult("call_1"), processor.Downstream)

	e := r.at(t, 0)
	if e.Kind != observers.FunctionCallCompleted {
		t.Errorf("kind = %s, want completed", e.Kind)
	}
	if !e.InProgressAt.IsZero() {
		t.Errorf("in progress at %v, want the zero time", e.InProgressAt)
	}
}

// TestBroadcastCallMomentReportedOnce covers the pair a broadcast builds.
func TestBroadcastCallMomentReportedOnce(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	down := callResult("call_1")
	up := callResult("call_1")
	down.SetBroadcastSiblingID(up.ID())
	up.SetBroadcastSiblingID(down.ID())

	push(o, down, processor.Downstream)
	push(o, up, processor.Upstream)

	if got := r.all(); len(got) != 1 {
		t.Errorf("events = %d, want 1", len(got))
	}
}

// TestFramesFromElsewhereAreIgnored covers the observer minding its own
// business.
func TestFramesFromElsewhereAreIgnored(t *testing.T) {
	r := newCallRecorder()
	o := newFunctionCalls(r, observers.FunctionCallConfig{})

	push(o, frames.NewTextFrame("hello"), processor.Downstream)

	if got := r.all(); len(got) != 0 {
		t.Errorf("events = %v, want none", got)
	}
}
