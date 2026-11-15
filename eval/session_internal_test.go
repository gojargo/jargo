package eval

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// fakeRTVIServer is a minimal RTVI WebSocket bot with scripted replies. It
// answers client-ready with bot-ready, sends each send-text the messages
// registered for its content, and records every message it receives.
type fakeRTVIServer struct {
	*httptest.Server

	// onConnect, when set, takes the connection over once it is accepted, for
	// a bot that refuses the run or never answers.
	onConnect func(ctx context.Context, conn *websocket.Conn)
	// capabilities, when set, is what bot-ready reports.
	capabilities map[string]any
	// greeting is sent right after bot-ready, like a bot that greets on
	// connect.
	greeting []rtvi.Message

	mu       sync.Mutex
	script   map[string][]rtvi.Message
	received []rtvi.Incoming
}

func newFakeRTVIServer(t *testing.T) *fakeRTVIServer {
	t.Helper()
	f := &fakeRTVIServer{script: map[string][]rtvi.Message{}}
	f.Server = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeRTVIServer) url() string { return wsURL(f.URL) }

// onText registers the messages the bot sends back for a send-text of content.
func (f *fakeRTVIServer) onText(content string, replies ...rtvi.Message) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.script[content] = append(f.script[content], replies...)
}

// botMessage is an RTVI server message the fake bot sends.
func botMessage(msgType string, data any) rtvi.Message {
	return rtvi.Message{Label: rtvi.MessageLabel, Type: msgType, Data: data}
}

// reply is the messages of one LLM response carrying text.
func reply(text string) []rtvi.Message {
	return []rtvi.Message{
		botMessage(rtvi.TypeBotLLMStarted, nil),
		botMessage(rtvi.TypeBotLLMText, rtvi.TextData{Text: text}),
		botMessage(rtvi.TypeBotLLMStopped, nil),
	}
}

func (f *fakeRTVIServer) serve(w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	ctx := r.Context()
	if f.onConnect != nil {
		f.onConnect(ctx, conn)
		return
	}
	defer func() { _ = conn.CloseNow() }()
	speaking := false
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		in, err := rtvi.ParseIncoming(data)
		if err != nil {
			continue
		}
		f.mu.Lock()
		f.received = append(f.received, in)
		var replies []rtvi.Message
		switch in.Type {
		case rtvi.TypeClientReady:
			ready := map[string]any{"version": rtvi.ProtocolVersion}
			if f.capabilities != nil {
				ready["capabilities"] = f.capabilities
			}
			replies = append([]rtvi.Message{botMessage(rtvi.TypeBotReady, ready)}, f.greeting...)
		case rtvi.TypeSendText:
			var d rtvi.SendTextData
			_ = json.Unmarshal(in.Data, &d)
			replies = f.script[d.Content]
		case rtvi.TypeRawAudio:
			// The harness streams the user's side continuously (silence around
			// each utterance). A real bot transcribes the audio and answers once
			// the user stops speaking; this one answers with its one scripted
			// reply on the first silent frame after speech.
			var d rtvi.RawAudioData
			_ = json.Unmarshal(in.Data, &d)
			pcm, _ := base64.StdEncoding.DecodeString(d.Base64Audio)
			switch {
			case slices.ContainsFunc(pcm, func(b byte) bool { return b != 0 }):
				speaking = true
			case speaking:
				speaking = false
				for _, script := range f.script {
					replies = script
					break
				}
			}
		}
		f.mu.Unlock()
		for _, reply := range replies {
			out, _ := json.Marshal(reply)
			if conn.Write(ctx, websocket.MessageText, out) != nil {
				return
			}
		}
	}
}

// messages is every message of msgType the bot received, in order.
func (f *fakeRTVIServer) messages(msgType string) []rtvi.Incoming {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []rtvi.Incoming
	for _, in := range f.received {
		if in.Type == msgType {
			out = append(out, in)
		}
	}
	return out
}

// evalMessages is the payload of every client-message of evalType the bot
// received.
func (f *fakeRTVIServer) evalMessages(evalType string) []map[string]any {
	var out []map[string]any
	for _, in := range f.messages(rtvi.TypeClientMessage) {
		var d struct {
			T string         `json:"t"`
			D map[string]any `json:"d"`
		}
		_ = json.Unmarshal(in.Data, &d)
		if d.T == evalType {
			out = append(out, d.D)
		}
	}
	return out
}

// sentTexts is the content of every send-text the bot received, in order.
func (f *fakeRTVIServer) sentTexts() []string {
	var out []string
	for _, in := range f.messages(rtvi.TypeSendText) {
		var d rtvi.SendTextData
		_ = json.Unmarshal(in.Data, &d)
		out = append(out, d.Content)
	}
	return out
}

// oneTurn is a scenario of one turn saying text and expecting expect.
func oneTurn(text string, expect ...Expectation) *ScriptScenario {
	return &ScriptScenario{Name: "t", Turns: []ScriptTurn{{User: text, Expect: expect}}}
}

// statuses is the status of each turn of a result, in order.
func statuses(res ScriptResult) []string {
	out := make([]string, 0, len(res.Turns))
	for _, t := range res.Turns {
		out = append(out, t.Status)
	}
	return out
}

// run plays scenario against the bot at url in a session built from it, with
// judge and params.
func run(t *testing.T, scenario *ScriptScenario, url string, params SessionParams, judge Judge) ScriptResult {
	t.Helper()
	s, err := ScriptSessionFromScenario(scenario, url, params, judge, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func wantPassed(t *testing.T, res ScriptResult) {
	t.Helper()
	if !res.Passed() {
		t.Fatalf("want a pass, got:\n%s", res)
	}
}

// wantRunFailure asserts the result failed on the run itself, once, with event
// and kind, and a reason holding reason, and that no turn was run.
func wantRunFailure(t *testing.T, res ScriptResult, event, kind, reason string) {
	t.Helper()
	if len(res.Failures) != 1 {
		t.Fatalf("failures = %+v, want one", res.Failures)
	}
	f := res.Failures[0]
	if f.EventName != event || f.Kind != kind || !strings.Contains(f.Reason, reason) {
		t.Fatalf("failure = %+v, want %s %s holding %q", f, event, kind, reason)
	}
	if got := statuses(res); !slices.Equal(got, []string{TurnNotRun}) {
		t.Fatalf("turns = %v, want the turn not run", got)
	}
}

func TestOneEventPass(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("what is the capital of France?", reply("Paris")...)
	res := run(t, oneTurn("what is the capital of France?",
		Expectation{Event: EventLLMStarted, WithinMS: 2000},
		Expectation{Event: EventLLMResponse, WithinMS: 2000, TextContains: "Paris"},
	), bot.url(), SessionParams{}, nil)
	wantPassed(t, res)
	// The connection opened with the flags it asks for, then the handshake.
	connect := bot.messages(rtvi.TypeClientMessage)
	if len(connect) == 0 || connect[0].ID != evalws.ConnectMessageType {
		t.Fatalf("the first client message = %+v, want eval-connect", connect)
	}
	if flags := bot.evalMessages(evalws.ConnectMessageType); len(flags) != 1 || flags[0]["skip_tts"] != true {
		t.Fatalf("eval-connect = %v, want skip_tts in text mode", flags)
	}
}

// The first response is filler; the answer arrives in a second segment, and
// text_contains aggregates across both.
func TestOneEventAggregatesPastFiller(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("weather?", append(reply("Let me check on that. "), reply("It is sunny in Paris.")...)...)
	wantPassed(t, run(t, oneTurn("weather?", Expectation{Event: EventLLMResponse, WithinMS: 2000, TextContains: "Paris"}),
		bot.url(), SessionParams{}, nil))
}

// An STT that finalizes an utterance in pieces emits one user-input per piece;
// text_contains accumulates the finals (skipping interims), so a phrase
// spanning pieces matches however each piece is spaced.
func TestUserTranscriptionAggregatesPieces(t *testing.T) {
	bot := newFakeRTVIServer(t)
	piece := func(text string, final bool) rtvi.Message {
		return botMessage(rtvi.TypeUserInput, map[string]any{"text": text, "input_type": "transcription", "final": final})
	}
	bot.onText("what is the capital of Germany?", append([]rtvi.Message{
		piece(" What is", false), piece(" What", true), piece(" is the capital", true), piece(" of", true),
		piece(" Germany?", true),
	}, reply("Berlin.")...)...)
	wantPassed(t, run(t, oneTurn("what is the capital of Germany?",
		Expectation{Event: EventUserTranscription, WithinMS: 2000, TextContains: "capital of Germany"},
		Expectation{Event: EventLLMResponse, WithinMS: 2000, TextContains: "Berlin"},
	), bot.url(), SessionParams{}, nil))
}

// inProgress is the bot reporting a call it is making.
func inProgress(name string, args map[string]any, id string) rtvi.Message {
	return botMessage(rtvi.TypeLLMFunctionCall,
		map[string]any{"function_name": name, "arguments": args, "tool_call_id": id})
}

// stopped is the bot reporting a call it stopped.
func stopped(name, id string, canceled bool) rtvi.Message {
	return botMessage(rtvi.TypeLLMFunctionCallStop, map[string]any{
		"function_name": name, "tool_call_id": id, keyCanceled: canceled,
	})
}

func TestFunctionCallPass(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("weather in Paris?", inProgress("get_weather", map[string]any{"city": "Paris"}, "1"))
	res := run(t, oneTurn("weather in Paris?", Expectation{
		Event: EventFunctionCall, WithinMS: 2000,
		Calls: []FunctionCall{{Name: "get_weather", Args: map[string]any{"city": "Paris"}}},
	}), bot.url(), SessionParams{}, nil)
	wantPassed(t, res)
	// Asserting on the arguments raised the bot's report level for the run.
	configure := bot.evalMessages(evalws.ConfigureMessageType)
	if len(configure) != 1 || !strings.Contains(mustJSON(t, configure[0]), `"function_call_report_level":{"*":"full"}`) {
		t.Fatalf("eval-configure = %v, want the full report level", configure)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// cancelScenario is one turn asserting a cancel call named a particular id.
func cancelScenario(expectedID string) *ScriptScenario {
	return oneTurn("never mind that one", Expectation{
		Event: EventFunctionCall, WithinMS: 2000,
		Calls: []FunctionCall{{Name: "cancel_write_report", Args: map[string]any{"tool_call_id": expectedID}}},
	})
}

// A model that mistypes an id, is refused, and repeats the call with the right
// one has made the call the turn asks for.
func TestCorrectedCallSatisfiesTheTurn(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("never mind that one",
		inProgress("cancel_write_report", map[string]any{"tool_call_id": "cull_turtles"}, "call_cancel_1"),
		inProgress("cancel_write_report", map[string]any{"tool_call_id": "call_turtles"}, "call_cancel_2"))
	wantPassed(t, run(t, cancelScenario("call_turtles"), bot.url(), SessionParams{}, nil))
}

// No call carried the expected id, so the failure names the ones that did
// arrive rather than claiming the call was never made.
func TestWrongArgsReportsWhatTheBotSent(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("never mind that one",
		inProgress("cancel_write_report", map[string]any{"tool_call_id": "call_volcanoes"}, "call_cancel"))
	res := run(t, cancelScenario("call_turtles"), bot.url(), SessionParams{}, nil)
	if res.Passed() || res.Failures[0].Kind != FailureFunctionArgsMismatch ||
		!strings.Contains(res.Failures[0].String(), "call_volcanoes") {
		t.Fatalf("failures = %+v", res.Failures)
	}
}

// runStopped plays a turn asserting a call stopped and how it ended, against a
// bot reporting it stopped as reported.
func runStopped(t *testing.T, reported, expected bool) ScriptResult {
	t.Helper()
	bot := newFakeRTVIServer(t)
	bot.onText("never mind that one", stopped("write_report", "call_turtles", reported))
	return run(t, oneTurn("never mind that one", Expectation{
		Event: EventFunctionCallStopped, WithinMS: 2000,
		Calls: []FunctionCall{{
			Name: "write_report", Args: map[string]any{"tool_call_id": "call_turtles", keyCanceled: expected},
		}},
	}), bot.url(), SessionParams{}, nil)
}

func TestCancelledCallMatches(t *testing.T) {
	wantPassed(t, runStopped(t, true, true))
}

// Work that finished on its own stops too, and a scenario asserting
// cancellation must not accept it.
func TestCallThatRanToCompletionIsNotACancellation(t *testing.T) {
	res := runStopped(t, false, true)
	if res.Passed() || res.Failures[0].Kind != FailureFunctionArgsMismatch {
		t.Fatalf("failures = %+v", res.Failures)
	}
}

// Both carry a name and a tool_call_id, so an expectation for one must not be
// satisfied by the other.
func TestStartedAndStoppedEventsDoNotClaimEachOther(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("report on sea turtles",
		inProgress("write_report", map[string]any{"topic": "sea turtles"}, "call_turtles"))
	res := run(t, oneTurn("report on sea turtles", Expectation{
		Event: EventFunctionCallStopped, WithinMS: 1500, Calls: []FunctionCall{{Name: "write_report"}},
	}), bot.url(), SessionParams{}, nil)
	if res.Passed() || res.Failures[0].Kind != FailureMissingFunctionCall {
		t.Fatalf("failures = %+v", res.Failures)
	}
}

// The bot reports the cancel tool running, then the call it canceled stopping,
// back to back; the events keep that order.
func TestCancelledCallStopsAfterTheCancelToolStarts(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("never mind that one",
		inProgress("cancel_write_report", map[string]any{"tool_call_id": "call_turtles"}, "call_cancel"),
		stopped("write_report", "call_turtles", true))
	wantPassed(t, run(t, oneTurn("never mind that one",
		Expectation{Event: EventFunctionCall, WithinMS: 2000, Calls: []FunctionCall{{Name: "cancel_write_report"}}},
		Expectation{
			Event: EventFunctionCallStopped, WithinMS: 2000,
			Calls: []FunctionCall{{Name: "write_report", Args: map[string]any{keyCanceled: true}}},
		},
	), bot.url(), SessionParams{}, nil))
}

func TestTextMismatchFailsClearly(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", reply("Paris")...)
	res := run(t, oneTurn("hi", Expectation{Event: EventLLMResponse, WithinMS: 2000, TextContains: "London"}),
		bot.url(), SessionParams{}, nil)
	if res.Passed() || len(res.Failures) != 1 || !strings.Contains(res.Failures[0].Reason, "does not contain") {
		t.Fatalf("failures = %+v", res.Failures)
	}
}

// twoTurnsFirstFails is a scenario whose first turn fails on content and whose
// second passes, against a bot that answers each.
func twoTurnsFirstFails(t *testing.T, stopOnFailure bool) (*fakeRTVIServer, *ScriptScenario) {
	t.Helper()
	bot := newFakeRTVIServer(t)
	bot.onText("first", reply("Paris")...)
	bot.onText("second", reply("Berlin")...)
	return bot, &ScriptScenario{
		Name: "stop", StopOnFailure: &stopOnFailure,
		Turns: []ScriptTurn{
			{User: "first", Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 300, TextContains: "London"}}},
			{User: "second", Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 2000, TextContains: "Berlin"}}},
		},
	}
}

// failureTurns is the turn index of each failure, in order.
func failureTurns(fs []AssertionFailure) []int {
	out := make([]int, 0, len(fs))
	for _, f := range fs {
		out = append(out, f.TurnIndex)
	}
	return out
}

// A failed turn ends the scenario by default, and the turn it stopped short of
// is not run rather than passed.
func TestFailedTurnStopsScenarioByDefault(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, true)
	scenario.StopOnFailure = nil
	res := run(t, scenario, bot.url(), SessionParams{}, nil)
	if got := failureTurns(res.Failures); !slices.Equal(got, []int{0}) {
		t.Fatalf("failures on turns %v, want only the first", got)
	}
	if got := bot.sentTexts(); !slices.Equal(got, []string{"first"}) {
		t.Fatalf("sent %v, want the second turn never sent", got)
	}
	if got := statuses(res); !slices.Equal(got, []string{TurnFailed, TurnNotRun}) {
		t.Fatalf("turns = %v, want failed then not run", got)
	}
}

// With stop_on_failure false every turn is driven, and the passing turn after
// the failure adds no failure of its own.
func TestStopOnFailureFalseDrivesRemainingTurns(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res := run(t, scenario, bot.url(), SessionParams{}, nil)
	if got := failureTurns(res.Failures); !slices.Equal(got, []int{0}) {
		t.Fatalf("failures on turns %v, want only the first", got)
	}
	if got := bot.sentTexts(); !slices.Equal(got, []string{"first", "second"}) {
		t.Fatalf("sent %v, want both turns", got)
	}
	if got := statuses(res); !slices.Equal(got, []string{TurnFailed, TurnPassed}) {
		t.Fatalf("turns = %v, want failed then passed", got)
	}
}

// Each turn carries its own failures, and the result's are every turn's, in
// order.
func TestTurnResultsCarryTheirOwnFailures(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res := run(t, scenario, bot.url(), SessionParams{}, nil)
	failed, passed := res.Turns[0], res.Turns[1]
	if len(failed.Failures) != 1 || failed.Failures[0].Kind != FailureTextMismatch || len(passed.Failures) != 0 {
		t.Fatalf("turns = %+v", res.Turns)
	}
	var flat []AssertionFailure
	for _, turn := range res.Turns {
		flat = append(flat, turn.Failures...)
	}
	if !slices.Equal(flat, res.Failures) {
		t.Fatalf("turn failures %+v, want the result's %+v", flat, res.Failures)
	}
}

// A turn records what each expectation matched; the marker the LLM produced is
// kept, so incomplete can be told apart.
func TestTurnResultsRecordWhatEachExpectationMatched(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hello",
		botMessage(rtvi.TypeBotLLMStarted, nil),
		botMessage(rtvi.TypeBotLLMText, rtvi.TextData{Text: "Hi there"}),
		botMessage(rtvi.TypeBotLLMMarker, map[string]any{"text": "◐", "kind": "short", "raw": "◐ Hi there"}),
		botMessage(rtvi.TypeBotLLMStopped, nil))
	res := run(t, oneTurn("hello",
		Expectation{Event: EventLLMMarker, Marker: MarkerIncomplete, WithinMS: 300},
		Expectation{Event: EventLLMResponse, WithinMS: 300},
	), bot.url(), SessionParams{}, nil)
	wantPassed(t, res)
	want := []ExpectationResult{
		{ExpectationIndex: 0, EventName: EventLLMMarker, Passed: true, Matched: "◐"},
		{ExpectationIndex: 1, EventName: EventLLMResponse, Passed: true, Matched: "Hi there"},
	}
	if got := res.Turns[0].Expectations; !slices.Equal(got, want) {
		t.Fatalf("expectations = %+v, want %+v", got, want)
	}
	if markers := bot.evalMessages(evalws.ConfigureMessageType); len(markers) != 1 || markers[0]["llm_markers"] != true {
		t.Fatalf("eval-configure = %v, want the markers asked for", markers)
	}
}

func TestTurnResultsRecordFailedExpectationsToo(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res := run(t, scenario, bot.url(), SessionParams{}, nil)
	want := [][]ExpectationResult{
		{{ExpectationIndex: 0, EventName: EventLLMResponse, Passed: false}},
		{{ExpectationIndex: 0, EventName: EventLLMResponse, Passed: true, Matched: "Berlin"}},
	}
	for i, turn := range res.Turns {
		if !slices.Equal(turn.Expectations, want[i]) {
			t.Errorf("turn %d expectations = %+v, want %+v", i, turn.Expectations, want[i])
		}
	}
}

func TestTurnResultsStopRecordingAtATimeout(t *testing.T) {
	bot := newFakeRTVIServer(t)
	res := run(t, oneTurn("hi",
		Expectation{Event: EventLLMResponse, WithinMS: 200},
		Expectation{Event: EventTTSResponse, WithinMS: 200},
	), bot.url(), SessionParams{}, nil)
	want := []ExpectationResult{{ExpectationIndex: 0, EventName: EventLLMResponse}}
	if got := res.Turns[0].Expectations; !slices.Equal(got, want) {
		t.Fatalf("expectations = %+v, want %+v", got, want)
	}
}

// A driven turn is timed; one that never ran has no time to report.
func TestTurnResultsAreTimed(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res := run(t, scenario, bot.url(), SessionParams{}, nil)
	if res.Turns[0].DurationMS <= 0 || res.Turns[1].Status != TurnPassed {
		t.Fatalf("turns %+v, want both driven and the first timed", res.Turns)
	}
	bot, scenario = twoTurnsFirstFails(t, true)
	stopping := run(t, scenario, bot.url(), SessionParams{}, nil)
	if stopping.Turns[1].DurationMS != 0 {
		t.Fatalf("duration of the turn never run = %dms, want none", stopping.Turns[1].DurationMS)
	}
}

func TestMissingEventTimesOut(t *testing.T) {
	bot := newFakeRTVIServer(t)
	res := run(t, oneTurn("hi", Expectation{Event: EventLLMResponse, WithinMS: 200}), bot.url(), SessionParams{}, nil)
	if len(res.Failures) != 1 || !strings.Contains(res.Failures[0].Reason, "arrived within") ||
		!strings.Contains(res.Failures[0].Reason, "200ms") {
		t.Fatalf("failures = %+v", res.Failures)
	}
}

// Only the first failed expectation reports once an event never arrived.
func TestSubsequentAssertionsSkippedAfterTimeout(t *testing.T) {
	bot := newFakeRTVIServer(t)
	res := run(t, oneTurn("hi",
		Expectation{Event: EventLLMStarted, WithinMS: 100},
		Expectation{Event: EventLLMResponse, WithinMS: 100},
		Expectation{Event: EventFunctionCall, WithinMS: 100},
	), bot.url(), SessionParams{}, nil)
	if len(res.Failures) != 1 {
		t.Fatalf("failures = %+v, want only the first", res.Failures)
	}
}

// A turn expecting a call and a response, getting neither, fails within one
// budget: both expectations wait against one deadline.
func TestTurnSharesOneDeadlineAcrossExpectations(t *testing.T) {
	bot := newFakeRTVIServer(t)
	start := time.Now()
	res := run(t, oneTurn("weather?",
		Expectation{Event: EventFunctionCall, WithinMS: 400, Calls: []FunctionCall{{Name: "get_weather"}}},
		Expectation{Event: EventLLMResponse, WithinMS: 400},
	), bot.url(), SessionParams{}, nil)
	if res.Passed() {
		t.Fatal("passed with nothing arriving")
	}
	if elapsed := time.Since(start); elapsed > 800*time.Millisecond+500*time.Millisecond {
		t.Fatalf("took %s, want one 400ms budget for both", elapsed)
	}
}

func TestSendAfterDelaysRun(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("first", botMessage(rtvi.TypeBotLLMStarted, nil), botMessage(rtvi.TypeBotLLMStopped, nil))
	bot.onText("second", reply("ok")...)
	res := run(t, &ScriptScenario{Name: "send_after", Turns: []ScriptTurn{
		{User: "first", Expect: []Expectation{{Event: EventLLMStarted, WithinMS: 2000}}},
		{
			User: "second", Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 2000}},
			SendAfter: &SendAfter{Event: EventLLMStarted, DelayMS: 200},
		},
	}}, bot.url(), SessionParams{}, nil)
	wantPassed(t, res)
	if res.DurationMS < 200 {
		t.Fatalf("duration = %dms, want the 200ms delay", res.DurationMS)
	}
}

// A run against a bot that is not there fails cleanly, as a failure to connect
// that scored nothing, and closes its judge.
func TestConnectFailureReportedCleanly(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := wsURL(srv.URL)
	srv.Close() // nothing listens there any more
	judge := newFakeJudge()
	scenario := oneTurn("x", Expectation{Event: EventLLMResponse, Eval: "ok"})
	res := run(t, scenario, url, SessionParams{ConnectTimeout: 500 * time.Millisecond}, judge)
	wantRunFailure(t, res, RunEventConnect, FailureConnectFailed, "failed to connect to "+url)
	if !judge.isClosed() {
		t.Fatal("the judge of a run that never reached the bot was not closed")
	}
}

// boomJudge panics as soon as it is given the conversation, as a broken judge
// might.
type boomJudge struct{ fakeJudge }

func (*boomJudge) AddUserMessage(string) {
	panic("judge boom") //nolint:forbidigo // the failure under test
}

// An unexpected failure mid-run, here a judge that panics, is reported as the
// run's error with its stack in the trace, scored against the turn it
// interrupted, rather than taking the caller down with it.
func TestUnexpectedErrorSurfacedNotSwallowed(t *testing.T) {
	bot := newFakeRTVIServer(t)
	res := run(t, oneTurn("hi", Expectation{Event: EventLLMStarted}), bot.url(), SessionParams{}, &boomJudge{})
	if len(res.Failures) != 1 {
		t.Fatalf("failures = %+v, want one", res.Failures)
	}
	f := res.Failures[0]
	if f.EventName != RunEventError || f.Kind != FailureError || !strings.Contains(f.Reason, "panic: judge boom") {
		t.Fatalf("failure = %+v, want the panic as the run's error", f)
	}
	trace := strings.Join(res.DebugLog, "\n")
	if !strings.Contains(trace, "judge boom") || !strings.Contains(trace, "goroutine") {
		t.Fatalf("trace lacks the panic and its stack:\n%s", trace)
	}
	got := statuses(res)
	if !slices.Equal(got, []string{TurnFailed}) || !slices.Equal(res.Turns[0].Failures, res.Failures) {
		t.Fatalf("turns = %+v, want the interrupted turn failed with the error", res.Turns)
	}
}

func TestContextSendsEvalContextMessage(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", botMessage(rtvi.TypeBotLLMStarted, nil), botMessage(rtvi.TypeBotLLMStopped, nil))
	scenario := oneTurn("hi", Expectation{Event: EventLLMStarted, WithinMS: 2000})
	scenario.Context = []map[string]any{{"role": "system", "content": "be terse"}}
	wantPassed(t, run(t, scenario, bot.url(), SessionParams{}, nil))
	sent := bot.evalMessages(evalws.ContextMessageType)
	if len(sent) != 1 || mustJSON(t, sent[0]["messages"]) != `[{"content":"be terse","role":"system"}]` {
		t.Fatalf("eval-context = %v, want the scenario's messages as they are", sent)
	}
}

// The context: message seeds the bot only when the scenario has one: an empty
// one would wipe the context the bot just built for itself.
func TestNoEvalContextMessageWhenEmpty(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", botMessage(rtvi.TypeBotLLMStarted, nil), botMessage(rtvi.TypeBotLLMStopped, nil))
	run(t, oneTurn("hi", Expectation{Event: EventLLMStarted, WithinMS: 2000}), bot.url(), SessionParams{}, nil)
	if sent := bot.evalMessages(evalws.ContextMessageType); len(sent) != 0 {
		t.Fatalf("eval-context = %v, want none", sent)
	}
}

// Stopping the run ends the bot's session with eval-cancel.
func TestARunEndsTheBotsSession(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", reply("hello")...)
	run(t, oneTurn("hi", Expectation{Event: EventLLMResponse, WithinMS: 2000}), bot.url(), SessionParams{}, nil)
	deadline := time.Now().Add(2 * time.Second)
	for len(bot.evalMessages(evalws.CancelMessageType)) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if len(bot.evalMessages(evalws.CancelMessageType)) != 1 {
		t.Fatal("the bot never got eval-cancel")
	}
}

// A bot that ends the call fails the turn waiting on it at once.
func TestTheBotHangingUpFailsTheTurnAsBotEnded(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onConnect = func(ctx context.Context, conn *websocket.Conn) {
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			in, _ := rtvi.ParseIncoming(data)
			switch in.Type {
			case rtvi.TypeClientReady:
				out, _ := json.Marshal(botMessage(rtvi.TypeBotReady, nil))
				_ = conn.Write(ctx, websocket.MessageText, out)
			case rtvi.TypeSendText:
				_ = conn.Close(websocket.StatusNormalClosure, "goodbye")
				return
			}
		}
	}
	start := time.Now()
	res := run(t, oneTurn("bye", Expectation{Event: EventLLMResponse, WithinMS: 10_000}), bot.url(), SessionParams{}, nil)
	if time.Since(start) > 3*time.Second {
		t.Fatal("the turn waited out its budget")
	}
	if len(res.Failures) != 1 || res.Failures[0].Kind != FailureBotEnded {
		t.Fatalf("failures = %+v, want the bot ending the call", res.Failures)
	}
}

// A bot that closes the connection before bot-ready, one at its capacity say,
// has refused the run: it fails at once, saying how the bot closed it.
func TestACloseBeforeBotReadyFailsAtOnceWithItsCode(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onConnect = func(_ context.Context, conn *websocket.Conn) {
		_ = conn.Close(websocket.StatusInternalError, "Concurrency limit reached")
	}
	start := time.Now()
	res := run(t, oneTurn("x", Expectation{Event: EventLLMStarted}), bot.url(), SessionParams{}, nil)
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("took %s, want it to fail at once", elapsed)
	}
	wantRunFailure(t, res, RunEventBotReady, FailureClosedBeforeReady, "1011 Concurrency limit reached")
}

// A bot that refuses the connection outright fails the run at once, with the
// status it answered.
func TestARefusedConnectionFailsAtOnce(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()
	start := time.Now()
	res := run(t, oneTurn("x", Expectation{Event: EventLLMStarted}), wsURL(srv.URL), SessionParams{}, nil)
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("took %s, want it to fail at once", elapsed)
	}
	wantRunFailure(t, res, RunEventConnect, FailureConnectFailed, "403")
}

// A bot that accepts the connection but never says it is ready still fails,
// once the handshake's budget runs out.
func TestABotThatAcceptsButNeverSaysReadyStillTimesOut(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onConnect = func(ctx context.Context, conn *websocket.Conn) {
		for {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
	}
	res := run(t, oneTurn("x", Expectation{Event: EventLLMStarted}), bot.url(),
		SessionParams{BotReadyTimeout: 300 * time.Millisecond}, nil)
	wantRunFailure(t, res, RunEventBotReady, FailureHandshakeTimeout, "bot-ready not received within 300ms")
}

// failingSynth is a user voice whose synthesis fails, reporting it as a service
// does.
type failingSynth struct{}

func (failingSynth) SampleRate() int { return 16000 }

func (failingSynth) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	return yield(frames.NewErrorFrame("failingTTS error: synthesis failed"))
}

// failingVoice is a user voice that never speaks.
func failingVoice(t *testing.T) *CachingTTSService {
	t.Helper()
	noCache := false
	voice, err := NewCachingTTSService(failingSynth{}, CachingTTSConfig{CacheKey: "failing", UseCache: &noCache})
	if err != nil {
		t.Fatal(err)
	}
	return voice
}

// A user voice that fails to synthesize the turn is the run's error, naming the
// failure, rather than a turn that waits out its budget for a reply to nothing.
func TestAUserVoiceThatFailsIsTheRunsError(t *testing.T) {
	bot := newFakeRTVIServer(t)
	scenario := oneTurn("a table for two", Expectation{Event: EventLLMResponse, WithinMS: 5000})
	scenario.UserAudio = true
	s, err := NewScriptSession(scenario, bot.url(), SessionParams{}, nil, failingVoice(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Failures) != 1 || res.Failures[0].Kind != FailureError ||
		!strings.Contains(res.Failures[0].Reason, "synthesis failed") || res.Turns[0].Status != TurnFailed {
		t.Fatalf("failures = %+v, want the synthesis failure as the run's error", res.Failures)
	}
}

// The response transcription needs the bot's audio: without audio mode the
// scenario is skipped, not failed, and its judge is closed.
func TestResponseTranscriptionSkip(t *testing.T) {
	scenario := oneTurn("x", Expectation{Event: EventResponse, Eval: "ok"})
	judge := newFakeJudge()
	res := run(t, scenario, "ws://localhost:0", SessionParams{}, judge)
	if res.Skipped == "" || res.Passed() || !strings.Contains(res.Skipped, "response") {
		t.Fatalf("result = %+v, want it skipped", res)
	}
	if got := statuses(res); !slices.Equal(got, []string{TurnNotRun}) {
		t.Fatalf("turns = %v", got)
	}
	if !judge.isClosed() {
		t.Fatal("the skipped run's judge was not closed")
	}
}

// progressScenario is one turn expecting the bot's reply.
func progressScenario() *ScriptScenario {
	return oneTurn("hi",
		Expectation{Event: EventLLMStarted, WithinMS: 2000},
		Expectation{Event: EventLLMResponse, WithinMS: 2000})
}

// EventProgress handlers see every turn and expectation, in order, before Run
// returns.
func TestProgressEvent(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", reply("hello")...)
	s, err := ScriptSessionFromScenario(progressScenario(), bot.url(), SessionParams{}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var (
		mu       sync.Mutex
		finished []string
		sources  []any
	)
	s.Add(EventProgress, func(_ context.Context, source any, args ...any) {
		time.Sleep(10 * time.Millisecond)
		p, _ := args[0].(Progress)
		mu.Lock()
		finished = append(finished, p.EventName+" "+p.Status)
		sources = append(sources, source)
		mu.Unlock()
	})
	if _, err := s.Run(t.Context()); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"hi turn", "llm_started matched", "llm_response matched"}
	if !slices.Equal(finished, want) {
		t.Fatalf("progress = %v, want %v", finished, want)
	}
	for _, source := range sources {
		if source != any(s.Session) {
			t.Fatalf("source = %T, want the session", source)
		}
	}
}

// Options.OnProgress is told the same records, in order.
func TestOptionsOnProgressIsCalled(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", reply("hello")...)
	var seen []string
	res, err := RunURL(t.Context(), progressScenario(), bot.url(), Options{
		OnProgress: func(p Progress) { seen = append(seen, p.EventName+" "+p.Status) },
	})
	if err != nil || !res.Passed() {
		t.Fatalf("run = %v, %v", res, err)
	}
	if want := []string{"hi turn", "llm_started matched", "llm_response matched"}; !slices.Equal(seen, want) {
		t.Fatalf("progress = %v, want %v", seen, want)
	}
}

// An audio run checks the audio the bot's bot-ready reports its transport
// carries.
func TestAudioSupportCheck(t *testing.T) {
	callerOff := map[string]any{"audio_in": false, "audio_out": true}
	botOff := map[string]any{"audio_in": true, "audio_out": false}
	bothOff := map[string]any{"audio_in": false, "audio_out": false}
	bothOn := map[string]any{"audio_in": true, "audio_out": true}

	runWith := func(
		t *testing.T, capabilities map[string]any, userAudio, botAudio bool, check string,
	) (ScriptResult, time.Duration) {
		t.Helper()
		bot := newFakeRTVIServer(t)
		bot.capabilities = capabilities
		bot.greeting = reply("Welcome!")
		scenario := &ScriptScenario{Name: "greeting", Turns: []ScriptTurn{
			{Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 2000}}},
		}}
		scenario.UserAudio, scenario.BotAudio = userAudio, botAudio
		var voice *CachingTTSService
		if userAudio {
			// The caller never speaks; the voice only makes the run send caller
			// audio.
			voice = failingVoice(t)
		}
		s, err := NewScriptSession(scenario, bot.url(), SessionParams{AudioSupportCheck: check}, nil, voice, nil)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		res, err := s.Run(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		return res, time.Since(start)
	}

	t.Run("a transport off for audio the run uses is refused at once", func(t *testing.T) {
		for _, tc := range []struct {
			userAudio, botAudio bool
			capabilities        map[string]any
			named               []string
		}{
			{true, false, callerOff, []string{"audio_in_enabled"}},
			{false, true, bothOff, []string{"audio_out_enabled"}},
			{true, true, bothOff, []string{"audio_in_enabled", "audio_out_enabled"}},
		} {
			res, elapsed := runWith(t, tc.capabilities, tc.userAudio, tc.botAudio, AudioSupportCheckWhenAdvertised)
			if elapsed > 2*time.Second {
				t.Fatalf("%+v: took %s, want it refused at once", tc, elapsed)
			}
			wantRunFailure(t, res, RunEventBotReady, FailureAudioUnsupported, "")
			for _, setting := range []string{"audio_in_enabled", "audio_out_enabled"} {
				if strings.Contains(res.Failures[0].Reason, setting) != slices.Contains(tc.named, setting) {
					t.Fatalf("%+v: reason %q, want it to name %v", tc, res.Failures[0].Reason, tc.named)
				}
			}
		}
	})
	t.Run("a run goes ahead when the audio it uses is on or unreported", func(t *testing.T) {
		for _, tc := range []struct {
			userAudio, botAudio bool
			capabilities        map[string]any
		}{
			{true, false, bothOn},
			{true, true, bothOn},
			// A bot that reports no capabilities runs unchecked.
			{true, true, nil},
			// A direction the run does not use is off.
			{false, true, callerOff},
			{true, false, botOff},
			{false, false, bothOff},
		} {
			if res, _ := runWith(t, tc.capabilities, tc.userAudio, tc.botAudio, AudioSupportCheckWhenAdvertised); !res.Passed() {
				t.Fatalf("%+v: want a pass, got:\n%s", tc, res)
			}
		}
	})
	t.Run("the check is off by default", func(t *testing.T) {
		if res, _ := runWith(t, callerOff, true, false, ""); !res.Passed() {
			t.Fatalf("want a pass, got:\n%s", res)
		}
	})
}

// An audio support check of no known value is refused.
func TestSessionParamsAreValidated(t *testing.T) {
	_, err := NewScriptSession(&ScriptScenario{Name: "t"}, "ws://localhost:0",
		SessionParams{AudioSupportCheck: "always"}, nil, nil, nil)
	if !errors.Is(err, errAudioSupportCheck) {
		t.Fatalf("err = %v, want the check refused", err)
	}
}
