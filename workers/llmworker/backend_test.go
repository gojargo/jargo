package llmworker_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
)

// Ported from upstream's backend worker suite. A scripted LLM service stands in
// for the backend model: each context frame plays the next scripted response
// (text and/or function calls), so the tests exercise the real aggregators, tool
// loop and job plumbing under a runner, driven through a BackendSession as a
// frontend drives them.

// step is one part of a scripted response: text, a thought, an error, or a
// function call.
type step struct {
	kind string
	text string
	name string
	id   string
	args string
}

func text(s string) step    { return step{kind: "text", text: s} }
func thought(s string) step { return step{kind: "thought", text: s} }
func fail(s string) step    { return step{kind: "error", text: s} }

func call(name, id, args string) step {
	return step{kind: "call", name: name, id: id, args: args}
}

// scriptedLLM plays one scripted response per context frame. settle holds the
// response open after issuing its calls, so a fast tool can return before the
// response ends.
type scriptedLLM struct {
	*llm.Base

	mu       sync.Mutex
	runs     [][]step
	settle   time.Duration
	contexts [][]frames.Message
}

func newScriptedLLM(settle time.Duration, runs ...[]step) *scriptedLLM {
	s := &scriptedLLM{runs: runs, settle: settle}
	s.Base = llm.New("ScriptedLLM", s)
	return s
}

// Generate implements llm.Generator. The script is played from ProcessFrame
// instead, which is where a response can be held open after its calls.
func (s *scriptedLLM) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func (s *scriptedLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	fr, ok := f.(*frames.LLMContextFrame)
	if !ok {
		return s.Base.ProcessFrame(ctx, f, dir)
	}
	if err := s.Base.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	convo := fr.Context
	s.SyncToolHandlers(ctx, convo.ToolsSchema())

	s.mu.Lock()
	s.contexts = append(s.contexts, convo.Messages())
	var steps []step
	if len(s.runs) > 0 {
		steps, s.runs = s.runs[0], s.runs[1:]
	}
	s.mu.Unlock()

	if err := s.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	var calls []frames.ToolCall
	for _, st := range steps {
		switch st.kind {
		case "text":
			if err := s.PushFrame(ctx, frames.NewLLMTextFrame(st.text), processor.Downstream); err != nil {
				return err
			}
		case "thought":
			for _, tf := range []frames.Frame{
				frames.NewLLMThoughtStartFrame(), frames.NewLLMThoughtTextFrame(st.text), frames.NewLLMThoughtEndFrame(),
			} {
				if err := s.PushFrame(ctx, tf, processor.Downstream); err != nil {
					return err
				}
			}
		case "error":
			s.PushError(ctx, st.text, nil, false)
		default:
			calls = append(calls, frames.ToolCall{ID: st.id, Name: st.name, Args: json.RawMessage(st.args)})
		}
	}
	if len(calls) > 0 {
		if err := s.RunFunctionCalls(ctx, convo, calls); err != nil {
			return err
		}
		if s.settle > 0 {
			time.Sleep(s.settle)
		}
	}
	return s.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// seen is the conversations the model was run on, in order.
func (s *scriptedLLM) seen() [][]frames.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.contexts)
}

// tool builds a tool the model may call.
func tool(name string, h llm.FunctionCallHandler) frames.Tool {
	return frames.Tool{
		Name: name, Description: name, Parameters: json.RawMessage(`{"type":"object"}`), Handler: h,
	}
}

// asyncTool builds a tool that keeps running after the model has replied.
func asyncTool(name string, cancellable bool, h llm.FunctionCallHandler) frames.Tool {
	t := tool(name, h)
	no := false
	t.CancelOnInterruption = &no
	t.CancellableByLLM = &cancellable
	return t
}

func answering(result string) llm.FunctionCallHandler {
	return func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, result, nil)
	}
}

func getWeather() frames.Tool {
	return tool("get_weather", answering(`{"temp":62,"conditions":"rain"}`))
}

func checkFlightStatus() frames.Tool {
	return tool("check_flight_status", answering(`{"status":"delayed","departure_time":"14:30"}`))
}

func bookTaxi() frames.Tool {
	return tool("book_taxi", answering(`{"status":"done"}`))
}

// slowLookup builds a tool that says it started, then waits to be let go or
// canceled before answering.
func slowLookup(started chan<- struct{}, mayFinish <-chan struct{}) llm.FunctionCallHandler {
	return func(ctx context.Context, p llm.FunctionCallParams) error {
		started <- struct{}{}
		select {
		case <-mayFinish:
		case <-ctx.Done():
			return ctx.Err()
		}
		return p.Result(ctx, `{"found":true}`, nil)
	}
}

// backendSetup is a backend and a requester under one runner, not yet running.
type backendSetup struct {
	backend   *llmworker.BackendWorker
	requester *workers.Base
	runner    *workers.Runner
}

func attachedBackend(
	t *testing.T, model *scriptedLLM, tools []frames.Tool, transform llmworker.BackendOutputTransform,
) *backendSetup {
	t.Helper()
	if tools == nil {
		tools = []frames.Tool{getWeather()}
	}
	convo := frames.NewLLMContext("You are the backend.")
	convo.SetTools(tools)
	backend, err := llmworker.NewBackend(llmworker.BackendConfig{
		Name: "backend", LLM: model, Context: convo, TransformOutput: transform,
	})
	if err != nil {
		t.Fatal(err)
	}
	off := false
	return &backendSetup{
		backend:   backend,
		requester: workers.New(workers.Config{Name: "requester"}, nil),
		runner:    workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &off}),
	}
}

// drive runs the runner and body together, canceling the runner when the body
// is done.
func (s *backendSetup) drive(t *testing.T, body func(ctx context.Context)) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	s.runner.AddWorkers(ctx, s.requester, s.backend)
	runDone := make(chan error, 1)
	go func() { runDone <- s.runner.Run(ctx, workers.RunOptions{}) }()
	func() {
		defer s.runner.Cancel(context.Background(), "")
		body(ctx)
	}()
	select {
	case <-runDone:
	case <-time.After(15 * time.Second):
		t.Fatal("the runner did not finish")
	}
}

// open attaches the requester to the backend.
func (s *backendSetup) open(ctx context.Context, t *testing.T) *llmworker.BackendSession {
	t.Helper()
	session, err := llmworker.OpenBackendSession(ctx, s.requester, "backend", 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

func send(ctx context.Context, t *testing.T, session *llmworker.BackendSession, request string) string {
	t.Helper()
	status, err := session.Send(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	return status
}

// untilIdle collects the session's events up to and including the next idle.
func untilIdle(ctx context.Context, t *testing.T, session *llmworker.BackendSession) []llmworker.BackendEvent {
	t.Helper()
	var events []llmworker.BackendEvent
	for {
		event, err := session.Next(ctx)
		if err != nil {
			t.Fatalf("the stream ended before going idle: %v (after %v)", err, events)
		}
		events = append(events, event)
		if _, idle := event.(llmworker.BackendIdle); idle {
			return events
		}
	}
}

func outputsOf(events []llmworker.BackendEvent) []llmworker.BackendOutput {
	var out []llmworker.BackendOutput
	for _, e := range events {
		if o, ok := e.(llmworker.BackendOutput); ok {
			out = append(out, o)
		}
	}
	return out
}

func textsOf(events []llmworker.BackendEvent) []string {
	var out []string
	for _, o := range outputsOf(events) {
		out = append(out, o.Text)
	}
	return out
}

func callsOf(events []llmworker.BackendEvent) []llmworker.BackendToolCall {
	var out []llmworker.BackendToolCall
	for _, e := range events {
		if c, ok := e.(llmworker.BackendToolCall); ok {
			out = append(out, c)
		}
	}
	return out
}

func isIdle(e llmworker.BackendEvent) bool {
	_, ok := e.(llmworker.BackendIdle)
	return ok
}

// waitStarted waits for a tool to say it started.
func waitStarted(t *testing.T, started <-chan struct{}) {
	t.Helper()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the tool never started")
	}
}

// inProgressHandled returns a channel closed once the backend's assistant
// aggregator has handled the named call's in-progress frame, which is what
// records the call, writes it into the conversation and tells an asynchronous
// call from a synchronous one.
//
// The tests that act on a running call wait for it after the call's handler has
// started. Upstream's test loop has handled the frame by the time the handler
// tells the test it started; goroutines give no such order, and a message or a
// cancellation sent before the frame is handled finds the call announced but not
// yet started. Call it before the call is made.
func inProgressHandled(backend *llmworker.BackendWorker, name string) <-chan struct{} {
	done := make(chan struct{})
	var once sync.Once
	events.On(backend.AssistantAggregator().Events(), processor.EventAfterProcessFrame,
		func(_ context.Context, f frames.Frame) {
			if fr, ok := f.(*frames.FunctionCallInProgressFrame); ok && fr.ToolName == name {
				once.Do(func() { close(done) })
			}
		})
	return done
}

// waitHandled waits for a call's in-progress frame to have been handled.
func waitHandled(t *testing.T, handled <-chan struct{}) {
	t.Helper()
	select {
	case <-handled:
	case <-time.After(5 * time.Second):
		t.Fatal("the call's in-progress frame was never handled")
	}
}

func usersOf(messages []frames.Message) []string {
	var out []string
	for _, m := range messages {
		// A tool's result is written as the user's turn too, with no text.
		if m.Role == frames.RoleUser && len(m.ToolResults) == 0 {
			out = append(out, m.Text)
		}
	}
	return out
}

func TestRenderTranscriptRequestIsTheInstructionAloneWhenNothingWasSaid(t *testing.T) {
	if got := llmworker.RenderTranscriptRequest(nil, "Do it", true); got != "Do it" {
		t.Errorf("rendered %q", got)
	}
}

func TestRenderExplicitRequestPutsTheInstructionAfterTheRequest(t *testing.T) {
	if got := llmworker.RenderExplicitRequest("Book a taxi", "Do it"); got != "Book a taxi\n\nDo it" {
		t.Errorf("rendered %q", got)
	}
}

func TestRenderTranscriptRequestPointsTheBackendAtTheConversation(t *testing.T) {
	got := llmworker.RenderTranscriptRequest(
		[]frames.Message{{Role: frames.RoleUser, Text: "what's the weather"}}, "", true)
	want := "Voice conversation so far:\n" +
		"USER: what's the weather\n" +
		"\n" +
		"Act on the user's most recent request in the conversation above. If it asks for something, " +
		"tell the user the result as soon as you have it, before going on with other work, in a message " +
		"that begins with >>. If it only stops or changes work already under way, tell them nothing: " +
		"the assistant already has."
	if got != want {
		t.Errorf("rendered\n%s\nwant\n%s", got, want)
	}
}

func TestAPayloadNamesItsType(t *testing.T) {
	if got := llmworker.NewBackendOutput("hello").ToPayload()["type"]; got != "output" {
		t.Errorf("an output's type is %v", got)
	}
	started := llmworker.BackendToolCall{Phase: llmworker.BackendToolCallStarted, FunctionName: "f", ToolCallID: "c1"}
	if got := started.ToPayload()["type"]; got != "tool_call" {
		t.Errorf("a tool call's type is %v", got)
	}
}

func TestAToolCallPhaseSurvivesThePayloadRoundTrip(t *testing.T) {
	c := llmworker.BackendToolCall{
		Phase: llmworker.BackendToolCallResult, FunctionName: "get_weather", ToolCallID: "c1",
		Arguments: json.RawMessage(`{"a":1}`), Result: `{"t":62}`, IsFinal: true,
	}
	// Through JSON, as a payload crossing a bus to another process would go.
	raw, err := json.Marshal(c.ToPayload())
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	got := llmworker.BackendToolCallFromPayload(payload)
	if got.Phase != c.Phase || got.FunctionName != c.FunctionName || got.ToolCallID != c.ToolCallID ||
		string(got.Arguments) != string(c.Arguments) || got.Result != c.Result || got.IsFinal != c.IsFinal {
		t.Errorf("round trip gave %+v, want %+v", got, c)
	}
}

// TestAPayloadLeavesTheFlagsItOmitsAtTheirDefaults checks a sender that predates
// a flag does not decide its value.
func TestAPayloadLeavesTheFlagsItOmitsAtTheirDefaults(t *testing.T) {
	got := llmworker.BackendOutputFromPayload(map[string]any{"text": "hello"})
	if got != llmworker.NewBackendOutput("hello") {
		t.Errorf("rebuilt %+v", got)
	}
}

// TestAPayloadCoercesTheFlagsItCarries checks flags that crossed a bus, which
// may not arrive as bools, are read as flags.
func TestAPayloadCoercesTheFlagsItCarries(t *testing.T) {
	got := llmworker.BackendOutputFromPayload(map[string]any{"text": "hello", "is_thought": 1.0, "prefers_spoken": 0.0})
	if !got.IsThought || got.PrefersSpoken {
		t.Errorf("rebuilt %+v, want a thought not spoken", got)
	}
}

// TestRenderTranscriptRequestFlattensWhatATranscriptCanHold checks a frontend
// can pass its context slice as it is: only spoken text survives.
func TestRenderTranscriptRequestFlattensWhatATranscriptCanHold(t *testing.T) {
	got := llmworker.RenderTranscriptRequest([]frames.Message{
		{Role: frames.RoleSystem, Text: "You are helpful."},
		{Role: frames.RoleUser, Text: "what is this"},
		{Role: frames.RoleAssistant, Text: "Let me look.", ToolCalls: []frames.ToolCall{
			{ID: "call_1", Name: "look", Args: json.RawMessage(`{}`)},
		}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "call_1", Name: "look", Content: `{"seen":true}`}}},
		frames.NewLLMSpecificMessage("anthropic", map[string]any{"type": "thought", "text": "hmm"}),
	}, "", true)
	want := "Voice conversation so far:\n" +
		"USER: what is this\n" +
		"ASSISTANT: Let me look.\n" +
		"\n" + llmworker.TranscriptRequestInstruction
	if got != want {
		t.Errorf("rendered\n%s\nwant\n%s", got, want)
	}
}

func TestAnAttachedFrontendHearsTheBackendWorkARequestThrough(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{text("Let me check."), call("get_weather", "call_1", `{"location":"Seattle"}`)},
		[]step{text(">> It's 62 and raining in Seattle.")},
	)
	s := attachedBackend(t, model, nil, nil)
	var events []llmworker.BackendEvent
	var statuses []string

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		statuses = append(statuses, send(ctx, t, session, "what's the weather in seattle"))
		events = untilIdle(ctx, t, session)
		if got := session.Capabilities(); !maps.Equal(got, map[string]bool{"steering": true, "progress": true}) {
			t.Errorf("capabilities = %v", got)
		}
	})

	if !slices.Equal(statuses, []string{"idle"}) {
		t.Errorf("statuses = %v", statuses)
	}
	// A message the model begins with the mark is for the user, and arrives
	// without it; anything else it writes is notes on the work.
	want := []llmworker.BackendOutput{
		{Text: "Let me check."},
		llmworker.NewBackendOutput("It's 62 and raining in Seattle."),
	}
	if got := outputsOf(events); !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
	var phases []string
	for _, c := range callsOf(events) {
		phases = append(phases, string(c.Phase)+" "+c.FunctionName)
	}
	wantPhases := []string{"started get_weather", "in_progress get_weather", "result get_weather"}
	if !slices.Equal(phases, wantPhases) {
		t.Errorf("call phases = %v, want %v", phases, wantPhases)
	}
	if !isIdle(events[len(events)-1]) {
		t.Errorf("the last event is %+v, want idle", events[len(events)-1])
	}
	first := model.seen()[0]
	if last := first[len(first)-1]; last.Role != frames.RoleUser || last.Text != "what's the weather in seattle" {
		t.Errorf("the model was first run on %+v, want the request last", last)
	}
	if s.backend.Working() {
		t.Error("the backend is still working")
	}
}

// TestAMessageSentWhileTheModelRunsIsTakenUpByTheNextRun checks a message has no
// run of its own: the run that follows the current step sees the request beside
// the result.
func TestAMessageSentWhileTheModelRunsIsTakenUpByTheNextRun(t *testing.T) {
	started, mayFinish := make(chan struct{}, 1), make(chan struct{})
	// The response stays open for a while after issuing its call, so the second
	// message arrives while the model is still busy with the first.
	model := newScriptedLLM(time.Second,
		[]step{call("slow_lookup", "call_1", `{}`)},
		[]step{text(">> Done with both.")},
	)
	s := attachedBackend(t, model, []frames.Tool{tool("slow_lookup", slowLookup(started, mayFinish))}, nil)
	var statuses []string
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		statuses = append(statuses, send(ctx, t, session, "First"))
		waitStarted(t, started)
		statuses = append(statuses, send(ctx, t, session, "Second"))
		close(mayFinish)
		events = untilIdle(ctx, t, session)
	})

	if !slices.Equal(statuses, []string{"idle", "working"}) {
		t.Errorf("statuses = %v", statuses)
	}
	seen := model.seen()
	if len(seen) != 2 {
		t.Fatalf("the model ran %d times, want 2", len(seen))
	}
	if got := usersOf(seen[1]); !slices.Equal(got, []string{"First", "Second"}) {
		t.Errorf("the second run saw the requests %v", got)
	}
	if got := textsOf(events); !slices.Equal(got, []string{"Done with both."}) {
		t.Errorf("outputs = %v", got)
	}
}

// TestAMessageSentWhileASynchronousToolIsInFlightWaitsForItsResult checks the
// request rides the run the call's result brings, since the context lacks that
// result until it returns.
func TestAMessageSentWhileASynchronousToolIsInFlightWaitsForItsResult(t *testing.T) {
	started, mayFinish := make(chan struct{}, 1), make(chan struct{})
	model := newScriptedLLM(0,
		[]step{call("slow_lookup", "call_1", `{}`)},
		[]step{text(">> The lookup is done, and on to the second thing.")},
	)
	s := attachedBackend(t, model, []frames.Tool{tool("slow_lookup", slowLookup(started, mayFinish))}, nil)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "First")
		waitStarted(t, started)
		if got := send(ctx, t, session, "Second"); got != "working" {
			t.Errorf("status = %q, want working", got)
		}
		time.Sleep(200 * time.Millisecond)
		if n := len(model.seen()); n != 1 {
			t.Errorf("the model ran %d times before the result, want 1", n)
		}
		close(mayFinish)
		events = untilIdle(ctx, t, session)
	})

	if got := textsOf(events); !slices.Equal(got, []string{"The lookup is done, and on to the second thing."}) {
		t.Errorf("outputs = %v", got)
	}
	// One run for both: the result and the request were in view together.
	seen := model.seen()
	if len(seen) != 2 {
		t.Fatalf("the model ran %d times, want 2", len(seen))
	}
	var sawRequest, sawResult bool
	for _, m := range seen[1] {
		sawRequest = sawRequest || m.Text == "Second"
		for _, r := range m.ToolResults {
			sawResult = sawResult || strings.Contains(r.Content, "found")
		}
	}
	if !sawRequest || !sawResult {
		t.Errorf("the second run saw request %t, result %t: %+v", sawRequest, sawResult, seen[1])
	}
}

// TestAMessageSentWhileOnlyAnAsynchronousToolIsInFlightRunsAtOnce checks the
// request does not wait: the model is idle and the call has its placeholder.
func TestAMessageSentWhileOnlyAnAsynchronousToolIsInFlightRunsAtOnce(t *testing.T) {
	started, mayFinish := make(chan struct{}, 1), make(chan struct{})
	model := newScriptedLLM(0,
		[]step{call("slow_lookup", "call_1", `{}`)},
		[]step{text(">> On the second thing now.")},
		[]step{text(">> And the lookup is done.")},
	)
	s := attachedBackend(t, model, []frames.Tool{asyncTool("slow_lookup", false, slowLookup(started, mayFinish))}, nil)
	handled := inProgressHandled(s.backend, "slow_lookup")
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "First")
		waitStarted(t, started)
		waitHandled(t, handled)
		if got := send(ctx, t, session, "Second"); got != "working" {
			t.Errorf("status = %q, want working", got)
		}
		for {
			event, err := session.Next(ctx)
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, event)
			if _, ok := event.(llmworker.BackendOutput); ok {
				break
			}
		}
		close(mayFinish)
		events = append(events, untilIdle(ctx, t, session)...)
	})

	// The second request ran before the lookup returned, on a context that
	// still had the call in progress.
	if got := textsOf(events); !slices.Equal(got, []string{"On the second thing now.", "And the lookup is done."}) {
		t.Errorf("outputs = %v", got)
	}
	seen := model.seen()
	if len(seen) != 3 {
		t.Fatalf("the model ran %d times, want 3", len(seen))
	}
	if !slices.Contains(usersOf(seen[1]), "Second") {
		t.Errorf("the second run saw %v, want the second request", usersOf(seen[1]))
	}
}

func TestClosingTheSessionStopsTheBackendAndTheNextSessionStartsClean(t *testing.T) {
	started := make(chan struct{}, 1)
	model := newScriptedLLM(0,
		[]step{call("slow_lookup", "call_1", `{}`)},
		[]step{text(">> Second time round.")},
	)
	s := attachedBackend(t, model, []frames.Tool{tool("slow_lookup", slowLookup(started, nil))}, nil)
	handled := inProgressHandled(s.backend, "slow_lookup")
	var second []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		send(ctx, t, session, "First")
		waitStarted(t, started)
		waitHandled(t, handled)
		session.Close(ctx)
		if !waitFor(5*time.Second, func() bool { return !s.backend.Working() }) {
			t.Fatal("the backend kept working after the frontend let go")
		}
		session = s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Second")
		second = untilIdle(ctx, t, session)
	})

	if got := textsOf(second); !slices.Equal(got, []string{"Second time round."}) {
		t.Errorf("outputs = %v", got)
	}
}

// TestDetachingLetsGoOfARequestHeldForACallInFlight checks the cancellations a
// detach brings do not run the model on the request they settle.
func TestDetachingLetsGoOfARequestHeldForACallInFlight(t *testing.T) {
	started := make(chan struct{}, 1)
	model := newScriptedLLM(0,
		[]step{call("slow_lookup", "call_1", `{}`)},
		[]step{text(">> Never ran.")},
	)
	s := attachedBackend(t, model, []frames.Tool{tool("slow_lookup", slowLookup(started, nil))}, nil)
	handled := inProgressHandled(s.backend, "slow_lookup")
	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		send(ctx, t, session, "First")
		waitStarted(t, started)
		waitHandled(t, handled)
		// Held for the synchronous call in flight.
		if got := send(ctx, t, session, "Second"); got != "working" {
			t.Errorf("status = %q, want working", got)
		}
		// Detached: the call is canceled, and the held request goes with it.
		session.Close(ctx)
		time.Sleep(500 * time.Millisecond)
		if s.backend.Working() {
			t.Error("the backend is still working")
		}
		if n := len(model.seen()); n != 1 {
			t.Errorf("the model ran %d times, want 1", n)
		}
	})
}

func TestASecondFrontendCannotAttachAndAMessageNeedsAnAttachedFrontend(t *testing.T) {
	model := newScriptedLLM(0, []step{text(">> Hello.")})
	s := attachedBackend(t, model, nil, nil)
	other := workers.New(workers.Config{Name: "other"}, nil)
	s.runner.AddWorkers(t.Context(), other)

	s.drive(t, func(ctx context.Context) {
		lone := llmworker.NewBackendSession(s.requester, "backend", 30*time.Second)
		if _, err := lone.Send(ctx, "Anyone there?"); !errors.Is(err, jobcontext.ErrJob) ||
			!strings.Contains(err.Error(), "no frontend is attached") {
			t.Errorf("sending with no frontend attached: %v", err)
		}

		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Hello")
		untilIdle(ctx, t, session)
		second, err := llmworker.OpenBackendSession(ctx, other, "backend", 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		defer second.Close(ctx)
		if _, err := second.Next(ctx); !errors.Is(err, jobcontext.ErrJob) ||
			!strings.Contains(err.Error(), "already attached") {
			t.Errorf("a second attachment: %v", err)
		}
	})
}

func TestABackendErrorReachesTheFrontendAndTheStreamGoesOn(t *testing.T) {
	model := newScriptedLLM(0, []step{fail("the provider is down")}, []step{text(">> Back again.")})
	s := attachedBackend(t, model, nil, nil)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "First")
		// The empty turn goes idle and the error frame reaches the worker, in
		// either order: the two are reported from different places.
		var sawError, sawIdle bool
		for !sawError || !sawIdle {
			event, err := session.Next(ctx)
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, event)
			switch event.(type) {
			case llmworker.BackendError:
				sawError = true
			case llmworker.BackendIdle:
				sawIdle = true
			}
		}
		send(ctx, t, session, "Second")
		events = append(events, untilIdle(ctx, t, session)...)
	})

	if !slices.Contains(events, llmworker.BackendEvent(llmworker.BackendError{Error: "the provider is down"})) {
		t.Errorf("events = %+v, want the error", events)
	}
	if got := textsOf(events); !slices.Equal(got, []string{"Back again."}) {
		t.Errorf("outputs = %v", got)
	}
}

func TestAnAttachedFrontendHearsThoughtsAndTheAppsOwnOutputs(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{thought("Check the weather first."), call("report", "call_1", `{"text":"Halfway there."}`)},
		[]step{text(">> Done.")},
	)
	var backend *llmworker.BackendWorker
	report := tool("report", func(ctx context.Context, p llm.FunctionCallParams) error {
		var args struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(p.Arguments, &args); err != nil {
			return err
		}
		backend.SendOutput(ctx, llmworker.NewBackendOutput(args.Text), false)
		backend.SendOutput(ctx, llmworker.NewBackendOutput("shaped"), true)
		return p.Result(ctx, "told", nil)
	})
	shout := func(_ context.Context, o llmworker.BackendOutput) (*llmworker.BackendOutput, error) {
		if o.IsThought {
			return nil, nil //nolint:nilnil // no output and no error drops the output
		}
		o.Text = strings.ToUpper(o.Text)
		return &o, nil
	}
	s := attachedBackend(t, model, []frames.Tool{report}, shout)
	backend = s.backend
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Do it")
		events = untilIdle(ctx, t, session)
	})

	// The thought was dropped by the transform; the app's outputs pass it unless
	// asked.
	want := []llmworker.BackendOutput{
		llmworker.NewBackendOutput("Halfway there."),
		llmworker.NewBackendOutput("SHAPED"),
		llmworker.NewBackendOutput("DONE."),
	}
	if got := outputsOf(events); !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
}

// contractBackend speaks the attach contract without an LLM, sending an update
// type the session does not know among the ones it does.
type contractBackend struct {
	*workers.Base
}

func newContractBackend() *contractBackend {
	c := &contractBackend{}
	c.Base = workers.New(workers.Config{Name: "backend"}, c)
	c.HandleJob(llmworker.AttachJobName, workers.JobOptions{}, func(ctx context.Context, m *bus.JobRequestMessage) {
		for _, update := range []map[string]any{
			{"type": "progress", "percent": 50},
			llmworker.NewBackendOutput("Done.").ToPayload(),
			{"type": "idle"},
		} {
			_ = c.SendJobUpdate(ctx, m.JobID, update, false)
		}
		<-ctx.Done()
	})
	return c
}

// TestTheSessionIgnoresStreamUpdatesItDoesNotKnow checks a backend speaking the
// contract may send update types the session does not know.
func TestTheSessionIgnoresStreamUpdatesItDoesNotKnow(t *testing.T) {
	off := false
	requester := workers.New(workers.Config{Name: "requester"}, nil)
	runner := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &off})
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	runner.AddWorkers(ctx, requester, newContractBackend())
	runDone := make(chan error, 1)
	go func() { runDone <- runner.Run(ctx, workers.RunOptions{}) }()

	var events []llmworker.BackendEvent
	func() {
		defer runner.Cancel(context.Background(), "")
		session, err := llmworker.OpenBackendSession(ctx, requester, "backend", 30*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		defer session.Close(ctx)
		events = untilIdle(ctx, t, session)
	}()
	<-runDone

	want := []llmworker.BackendEvent{llmworker.NewBackendOutput("Done."), llmworker.BackendIdle{}}
	if !slices.Equal(events, want) {
		t.Errorf("events = %+v, want %+v", events, want)
	}
}

// TestClosingTheSessionEndsItsStream checks Next reports the end of a stream
// this side let go of, and that it was this side.
func TestClosingTheSessionEndsItsStream(t *testing.T) {
	model := newScriptedLLM(0)
	s := attachedBackend(t, model, nil, nil)

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		session.Close(ctx)
		if _, err := session.Next(ctx); !errors.Is(err, io.EOF) {
			t.Errorf("Next after closing = %v, want io.EOF", err)
		}
		if !session.Detached() {
			t.Error("the session does not know it let go")
		}
	})
}

func TestTransformOutputShapesWhatTheModelWrites(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{text(">> Checking."), call("get_weather", "call_1", `{"location":"Seattle"}`)},
		[]step{thought("Keep it short."), text("Internal note.")},
		[]step{text("")},
	)
	transform := func(_ context.Context, o llmworker.BackendOutput) (*llmworker.BackendOutput, error) {
		if o.IsThought {
			return nil, nil //nolint:nilnil // no output and no error drops the output
		}
		if o.PrefersSpoken {
			o.Text = strings.ToUpper(o.Text)
		} else {
			o.Text = ""
		}
		return &o, nil
	}
	s := attachedBackend(t, model, nil, transform)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Do it")
		events = untilIdle(ctx, t, session)
	})

	// The transform sees each output after the mark rule (flag set, mark
	// stripped): the thought was dropped, the spoken output reshaped, and an
	// emptied note still sent.
	want := []llmworker.BackendOutput{llmworker.NewBackendOutput("CHECKING."), {Text: ""}}
	if got := outputsOf(events); !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
}

func TestAMarkedMessageBesideToolCallsIsSpokenAndTheCallsStillRun(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{
			text(">> Your flight is delayed, so I'm booking you a taxi."),
			call("book_taxi", "call_1", `{"time":"12:30"}`),
		},
		[]step{text(">> Taxi booked for 12:30.")},
	)
	s := attachedBackend(t, model, []frames.Tool{bookTaxi()}, nil)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Check my flight and book a taxi")
		events = untilIdle(ctx, t, session)
	})

	// The mark decides, not whether the turn made calls: a result can ride the
	// turn that starts the next step.
	want := []llmworker.BackendOutput{
		llmworker.NewBackendOutput("Your flight is delayed, so I'm booking you a taxi."),
		llmworker.NewBackendOutput("Taxi booked for 12:30."),
	}
	if got := outputsOf(events); !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
	for _, c := range callsOf(events) {
		if c.FunctionName != "book_taxi" {
			t.Errorf("a call to %s was reported", c.FunctionName)
		}
	}
	seen := model.seen()
	if len(seen) != 2 {
		t.Fatalf("the model ran %d times, want 2", len(seen))
	}
	if !isIdle(events[len(events)-1]) {
		t.Errorf("the last event is %+v, want idle", events[len(events)-1])
	}
	// The model's own context keeps the message as written, mark and all.
	var kept bool
	for _, m := range seen[1] {
		kept = kept || (m.Role == frames.RoleAssistant && strings.Contains(m.Text, ">> Your flight is delayed"))
	}
	if !kept {
		t.Errorf("the model's context lost the marked message: %+v", seen[1])
	}
}

func TestAMarkPartwayThroughAMessageSplitsItIntoANoteAndAResult(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{
			text("Found the issue: a fixed backoff. Fixing it next.\n\n>> The runbook says to check the status page first."),
			call("book_taxi", "call_1", `{"time":"12:30"}`),
		},
		[]step{text(">> Fixed, and the tests pass.")},
	)
	s := attachedBackend(t, model, []frames.Tool{bookTaxi()}, nil)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Fix the test and look up the runbook")
		events = untilIdle(ctx, t, session)
	})

	// What precedes the marked line is a note; from the mark on is spoken.
	want := []llmworker.BackendOutput{
		{Text: "Found the issue: a fixed backoff. Fixing it next."},
		llmworker.NewBackendOutput("The runbook says to check the status page first."),
		llmworker.NewBackendOutput("Fixed, and the tests pass."),
	}
	if got := outputsOf(events); !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
}

// TestAnUnmarkedMessageIsANoteAndAThoughtIsNeverSpoken checks a message that
// stops the work gets no spoken reply: the model writes notes, or nothing.
func TestAnUnmarkedMessageIsANoteAndAThoughtIsNeverSpoken(t *testing.T) {
	started := make(chan struct{}, 1)
	model := newScriptedLLM(0,
		[]step{thought(">> Look it up first."), call("slow_lookup", "call_1", `{}`)},
		[]step{text("Stopping the lookup."), call("cancel_slow_lookup", "call_c", `{}`)},
		// The cancellation's result brings one more run.
		[]step{text("Nothing left to do.")},
	)
	s := attachedBackend(t, model, []frames.Tool{asyncTool("slow_lookup", true, slowLookup(started, nil))}, nil)
	handled := inProgressHandled(s.backend, "slow_lookup")
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Look it up")
		waitStarted(t, started)
		waitHandled(t, handled)
		if got := send(ctx, t, session, "Never mind"); got != "working" {
			t.Errorf("status = %q, want working", got)
		}
		events = untilIdle(ctx, t, session)
	})

	// A thought stays a note whatever it begins with, and keeps its text as is.
	want := []llmworker.BackendOutput{
		{Text: ">> Look it up first.", IsThought: true},
		{Text: "Stopping the lookup."},
		{Text: "Nothing left to do."},
	}
	if got := outputsOf(events); !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
	var canceled bool
	for _, c := range callsOf(events) {
		canceled = canceled || (c.FunctionName == "slow_lookup" && c.Phase == llmworker.BackendToolCallCanceled)
	}
	if !canceled {
		t.Errorf("calls = %+v, want the lookup canceled", callsOf(events))
	}
	if !isIdle(events[len(events)-1]) {
		t.Errorf("the last event is %+v, want idle", events[len(events)-1])
	}
	if n := len(model.seen()); n != 3 {
		t.Errorf("the model ran %d times, want 3", n)
	}
}

func TestTheModelIsToldHowWhatItWritesReachesTheUser(t *testing.T) {
	model := newScriptedLLM(0)
	if _, err := llmworker.NewBackend(llmworker.BackendConfig{LLM: model}); err != nil {
		t.Fatal(err)
	}
	if told := model.SystemInstruction(); !strings.Contains(told, llmworker.BackendPairingInstruction) {
		t.Errorf("the model was told %q", told)
	}
	if !strings.Contains(llmworker.BackendPairingInstruction, llmworker.SpokenMark) {
		t.Error("the pairing instruction does not name the mark")
	}
}

func TestTheLLMRunsAtItsProvidersDefaultThinking(t *testing.T) {
	model := newScriptedLLM(0)
	if model.UseProviderThinkingDefault() {
		t.Fatal("use_provider_thinking_default is on before the backend set it")
	}
	if _, err := llmworker.NewBackend(llmworker.BackendConfig{LLM: model}); err != nil {
		t.Fatal(err)
	}
	if !model.UseProviderThinkingDefault() {
		t.Error("the backend left the model at the lowest thinking setting")
	}
}

// objectingLLM is a model that declines the backend role.
type objectingLLM struct{ *scriptedLLM }

func (objectingLLM) LLMWithBackendRoleObjection(role llm.LLMWithBackendRole) string {
	return "cannot be the " + string(role)
}

func TestAnLLMThatDeclinesTheBackendRoleIsRefused(t *testing.T) {
	model := objectingLLM{newScriptedLLM(0)}
	_, err := llmworker.NewBackend(llmworker.BackendConfig{LLM: model})
	if err == nil || !strings.Contains(err.Error(), "cannot be the backend") {
		t.Errorf("err = %v, want the objection", err)
	}
	if model.UseProviderThinkingDefault() {
		t.Error("a refused backend still changed the model's thinking")
	}
}

func TestAnAppCanReplaceTheBackendsPairingInstruction(t *testing.T) {
	model := newScriptedLLM(0)
	if _, err := llmworker.NewBackend(llmworker.BackendConfig{
		LLM: model, PairingInstruction: "Begin what the user should hear with >>.",
	}); err != nil {
		t.Fatal(err)
	}
	told := model.SystemInstruction()
	if !strings.Contains(told, "Begin what the user should hear with >>.") ||
		strings.Contains(told, llmworker.BackendPairingInstruction) {
		t.Errorf("the model was told %q", told)
	}
}

//nolint:gochecknoglobals // sentinel error for the test below
var errToolBroke = errors.New("the tool broke")

func TestAToolHandlerThatFailsLeavesTheBackendWorking(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{call("raise_an_error", "call_1", `{}`)},
		[]step{text(">> That did not work, sorry.")},
	)
	broken := tool("raise_an_error", func(context.Context, llm.FunctionCallParams) error {
		return errToolBroke
	})
	s := attachedBackend(t, model, []frames.Tool{broken}, nil)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Do it")
		events = untilIdle(ctx, t, session)
	})

	for _, e := range events {
		if _, ok := e.(llmworker.BackendError); ok {
			t.Errorf("a failing tool was reported as the backend failing: %+v", e)
		}
	}
	if got := textsOf(events); !slices.Equal(got, []string{"That did not work, sorry."}) {
		t.Errorf("outputs = %v", got)
	}
}

func TestAChainedRequestReportsAtEachStepAndIdlesAtTheEnd(t *testing.T) {
	model := newScriptedLLM(0,
		[]step{thought("Check the flight first."), call("check_flight_status", "call_1", `{"flight_number":"AA100"}`)},
		[]step{text("It's delayed, so I'm booking a taxi for 12:30."), call("book_taxi", "call_2", `{"time":"12:30"}`)},
		[]step{text(">> Taxi booked for 12:30.")},
	)
	s := attachedBackend(t, model, []frames.Tool{checkFlightStatus(), bookTaxi()}, nil)
	var events []llmworker.BackendEvent

	s.drive(t, func(ctx context.Context) {
		session := s.open(ctx, t)
		defer session.Close(ctx)
		send(ctx, t, session, "Get me to the airport")
		events = untilIdle(ctx, t, session)
	})

	if n := len(model.seen()); n != 3 {
		t.Errorf("the model ran %d times, want 3", n)
	}
	type said struct {
		text    string
		thought bool
	}
	var got []said
	for _, o := range outputsOf(events) {
		got = append(got, said{o.Text, o.IsThought})
	}
	want := []said{
		{"Check the flight first.", true},
		{"It's delayed, so I'm booking a taxi for 12:30.", false},
		{"Taxi booked for 12:30.", false},
	}
	if !slices.Equal(got, want) {
		t.Errorf("outputs = %+v, want %+v", got, want)
	}
	idles := 0
	for _, e := range events {
		if isIdle(e) {
			idles++
		}
	}
	if idles != 1 || !isIdle(events[len(events)-1]) {
		t.Errorf("events = %+v, want one idle, at the end", events)
	}
}
