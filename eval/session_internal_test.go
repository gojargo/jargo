package eval

import (
	"bytes"
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
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport/wsserver/rtviws"
)

// fakeRTVIServer is a minimal RTVI WebSocket bot with scripted replies. It
// answers client-ready with bot-ready, sends each send-text the messages
// registered for its content, and records every message it receives.
type fakeRTVIServer struct {
	*httptest.Server

	// onConnect, when set, takes the connection over once it is accepted, for
	// a bot that refuses the run or never answers.
	onConnect func(ctx context.Context, conn *websocket.Conn)
	script    map[string][]rtvi.Message

	mu       sync.Mutex
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
	conn.SetReadLimit(readLimit)
	ctx := r.Context()
	if f.onConnect != nil {
		f.onConnect(ctx, conn)
		return
	}
	defer func() { _ = conn.CloseNow() }()
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
		f.mu.Unlock()

		var replies []rtvi.Message
		switch in.Type {
		case rtvi.TypeClientReady:
			replies = []rtvi.Message{botMessage(rtvi.TypeBotReady, nil)}
		case rtvi.TypeSendText:
			var d rtvi.SendTextData
			_ = json.Unmarshal(in.Data, &d)
			replies = f.script[d.Content]
		}
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
func oneTurn(text string, expect ...Expectation) *Scenario {
	return &Scenario{Name: "t", Turns: []Turn{{User: text, Expect: expect}}, hasTurns: true}
}

// statuses is the status of each turn of a result, in order.
func statuses(res Result) []string {
	out := make([]string, 0, len(res.Turns))
	for _, t := range res.Turns {
		out = append(out, t.Status)
	}
	return out
}

// wantRunFailure asserts the result failed on the run itself, once, with event
// and kind, and a reason holding reason, and that no turn was run.
func wantRunFailure(t *testing.T, res Result, err error, event, kind, reason string) {
	t.Helper()
	if err != nil {
		t.Fatalf("err = %v, want the failure in the result", err)
	}
	if len(res.Failures) != 1 {
		t.Fatalf("failures = %+v, want one", res.Failures)
	}
	f := res.Failures[0]
	if f.Event != event || f.Kind != kind || !strings.Contains(f.Reason, reason) {
		t.Fatalf("failure = %+v, want %s %s holding %q", f, event, kind, reason)
	}
	if got := statuses(res); !slices.Equal(got, []string{TurnNotRun}) {
		t.Fatalf("turns = %v, want the turn not run", got)
	}
}

// The context: message seeds the bot only when the scenario has one: an empty
// one would wipe the context the bot just built for itself.
func TestNoEvalContextMessageIsSentWhenTheContextIsEmpty(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onText("hi", botMessage(rtvi.TypeBotLLMStarted, nil), botMessage(rtvi.TypeBotLLMStopped, nil))

	res, err := runAgainst(t.Context(), oneTurn("hi", Expectation{Event: EventLLMStarted, WithinMS: 2000}),
		bot.url(), Options{})
	if err != nil || !res.Passed() {
		t.Fatalf("run = %v, %v", res, err)
	}
	clientMessages := bot.messages(typeClientMessage)
	for _, in := range clientMessages {
		var d clientMessageData
		_ = json.Unmarshal(in.Data, &d)
		if d.T == evalContextMessage {
			t.Fatalf("an empty context was sent: %s", in.Data)
		}
	}
}

// A bot that closes the connection before bot-ready, one at its capacity say,
// has refused the run: it fails at once, saying how the bot closed it, rather
// than waiting out the handshake.
func TestACloseBeforeBotReadyFailsAtOnceWithItsCode(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onConnect = func(_ context.Context, conn *websocket.Conn) {
		_ = conn.Close(websocket.StatusInternalError, "Concurrency limit reached")
	}

	start := time.Now()
	res, err := runAgainst(t.Context(), oneTurn("x", Expectation{Event: EventLLMStarted}), bot.url(), Options{})
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("took %s, want it to fail at once rather than wait out the handshake", elapsed)
	}
	wantRunFailure(t, res, err, RunEventBotReady, FailureClosedBeforeReady, "1011 Concurrency limit reached")
}

// A bot that refuses the connection outright fails the run at once, with the
// status it answered.
func TestARefusedConnectionFailsAtOnce(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	start := time.Now()
	res, err := runAgainst(t.Context(), oneTurn("x", Expectation{Event: EventLLMStarted}), wsURL(srv.URL), Options{})
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("took %s, want it to fail at once", elapsed)
	}
	wantRunFailure(t, res, err, RunEventConnect, FailureConnectFailed, "403")
}

// A bot that accepts the connection but never says it is ready still fails,
// once the handshake's budget runs out.
func TestABotThatAcceptsButNeverSaysReadyStillTimesOut(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.onConnect = func(ctx context.Context, conn *websocket.Conn) {
		defer func() { _ = conn.CloseNow() }()
		for {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
	}

	res, err := runAgainst(t.Context(), oneTurn("x", Expectation{Event: EventLLMStarted}), bot.url(),
		Options{BotReadyTimeout: 300 * time.Millisecond})
	wantRunFailure(t, res, err, RunEventBotReady, FailureHandshakeTimeout, "bot-ready not received within 300ms")
}

// A run against a bot that is not there fails cleanly, as a failure to connect
// that scored nothing.
func TestAConnectFailureIsReportedCleanly(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := wsURL(srv.URL)
	srv.Close() // nothing listens there any more

	res, err := runAgainst(t.Context(), oneTurn("x", Expectation{Event: EventLLMStarted}), url,
		Options{ConnectTimeout: 500 * time.Millisecond})
	wantRunFailure(t, res, err, RunEventConnect, FailureConnectFailed, "failed to connect to "+url)
	if res.Scenario != "t" {
		t.Fatalf("scenario = %q, want it named", res.Scenario)
	}
}

// failingSynth is a user voice whose synthesis fails.
type failingSynth struct{}

func (failingSynth) SampleRate() int { return 16000 }

func (failingSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return errors.New("synthesis failed") //nolint:err113 // a test double's own failure
}

// A user voice that fails to synthesize the turn is the run's error, naming the
// failure, rather than a turn that waits out its budget for a reply to nothing.
func TestAUserVoiceThatFailsIsTheRunsError(t *testing.T) {
	bot := newFakeRTVIServer(t)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()

	res, err := runAgainst(ctx, oneTurn("a table for two", Expectation{Event: EventLLMResponse, WithinMS: 5000}),
		bot.url(), Options{UserTTS: tts.New("failingTTS", failingSynth{})})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Failures) != 1 || res.Failures[0].Kind != FailureError ||
		!strings.Contains(res.Failures[0].Reason, "synthesis failed") {
		t.Fatalf("failures = %+v, want the synthesis failure as the run's error", res.Failures)
	}
	if ctx.Err() != nil {
		t.Fatal("the run waited for its context to end rather than failing")
	}
}

// boomJudge panics as soon as it is given the conversation, as a broken judge
// might.
type boomJudge struct{ countingJudge }

func (*boomJudge) AddUserMessage(string) {
	panic("judge boom") //nolint:forbidigo // the failure under test
}

// An unexpected failure mid-run, here a judge that panics, is reported as the
// run's error with its stack in the trace, scored against the turn it
// interrupted, rather than taking the caller down with it.
func TestAnUnexpectedErrorIsSurfacedNotSwallowed(t *testing.T) {
	bot := newFakeRTVIServer(t)
	res, err := runAgainst(t.Context(), oneTurn("hi", Expectation{Event: EventLLMStarted}), bot.url(),
		Options{Judge: &boomJudge{}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Passed() || len(res.Failures) != 1 {
		t.Fatalf("failures = %+v, want one", res.Failures)
	}
	f := res.Failures[0]
	if f.Event != RunEventError || f.Kind != FailureError || !strings.Contains(f.Reason, "judge boom") {
		t.Fatalf("failure = %+v, want the panic as the run's error", f)
	}
	trace := strings.Join(res.DebugLog, "\n")
	if !strings.Contains(trace, "judge boom") || !strings.Contains(trace, "goroutine") {
		t.Fatalf("trace lacks the panic and its stack:\n%s", trace)
	}
	if got := statuses(res); !slices.Equal(got, []string{TurnFailed}) {
		t.Fatalf("turns = %v, want the interrupted turn failed", got)
	}
	if !slices.Equal(res.Turns[0].Failures, res.Failures) {
		t.Fatalf("turn failures = %+v, want the run's error", res.Turns[0].Failures)
	}
}

// twoTurnsFirstFails is a scenario whose first turn fails on content and whose
// second passes, against a bot that answers each.
func twoTurnsFirstFails(t *testing.T, stopOnFailure bool) (*fakeRTVIServer, *Scenario) {
	t.Helper()
	bot := newFakeRTVIServer(t)
	bot.onText("first", reply("Paris")...)
	bot.onText("second", reply("Berlin")...)
	return bot, &Scenario{
		Name: "stop", StopOnFailure: &stopOnFailure, hasTurns: true,
		Turns: []Turn{
			{User: "first", Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 300, TextContains: "London"}}},
			{User: "second", Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 2000, TextContains: "Berlin"}}},
		},
	}
}

// failureTurns is the turn of each failure, in order.
func failureTurns(fs []Failure) []int {
	out := make([]int, 0, len(fs))
	for _, f := range fs {
		out = append(out, f.Turn)
	}
	return out
}

// A failed turn ends the scenario by default, and the turn it stopped short of
// is not run rather than passed.
func TestAFailedTurnStopsTheScenarioByDefault(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, true)
	scenario.StopOnFailure = nil
	res, err := runAgainst(t.Context(), scenario, bot.url(), Options{})
	if err != nil || res.Passed() {
		t.Fatalf("run = %v, %v; want a failure", res, err)
	}
	if got := failureTurns(res.Failures); !slices.Equal(got, []int{1}) {
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
func TestStopOnFailureFalseDrivesTheRemainingTurns(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res, err := runAgainst(t.Context(), scenario, bot.url(), Options{})
	if err != nil || res.Passed() {
		t.Fatalf("run = %v, %v; want a failure", res, err)
	}
	if got := failureTurns(res.Failures); !slices.Equal(got, []int{1}) {
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
	res, err := runAgainst(t.Context(), scenario, bot.url(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	failed, passed := res.Turns[0], res.Turns[1]
	if len(failed.Failures) != 1 || failed.Failures[0].Kind != FailureTextMismatch {
		t.Fatalf("first turn failures = %+v, want one text mismatch", failed.Failures)
	}
	if len(passed.Failures) != 0 {
		t.Fatalf("second turn failures = %+v, want none", passed.Failures)
	}
	var flat []Failure
	for _, turn := range res.Turns {
		flat = append(flat, turn.Failures...)
	}
	if !slices.Equal(flat, res.Failures) {
		t.Fatalf("turn failures %+v, want the result's %+v", flat, res.Failures)
	}
}

// A turn records its failed expectations as well as what the passing ones
// matched.
func TestTurnResultsRecordFailedExpectationsToo(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res, err := runAgainst(t.Context(), scenario, bot.url(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]ExpectationResult{
		{{Expectation: 1, Event: EventLLMResponse, Passed: false}},
		{{Expectation: 1, Event: EventLLMResponse, Passed: true, Matched: "Berlin"}},
	}
	for i, turn := range res.Turns {
		if !slices.Equal(turn.Expectations, want[i]) {
			t.Errorf("turn %d expectations = %+v, want %+v", i+1, turn.Expectations, want[i])
		}
	}
}

// A driven turn is timed; one that never ran has no time to report.
func TestTurnResultsAreTimed(t *testing.T) {
	bot, scenario := twoTurnsFirstFails(t, false)
	res, err := runAgainst(t.Context(), scenario, bot.url(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Turns[0].Duration <= 0 || res.Turns[1].Duration <= 0 {
		t.Fatalf("durations %s and %s, want both driven turns timed", res.Turns[0].Duration, res.Turns[1].Duration)
	}

	bot, scenario = twoTurnsFirstFails(t, true)
	stopping, err := runAgainst(t.Context(), scenario, bot.url(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if stopping.Turns[1].Duration != 0 {
		t.Fatalf("duration of the turn never run = %s, want none", stopping.Turns[1].Duration)
	}
}

// toneSynth renders any text as 120ms of non-silent 16 kHz PCM.
type toneSynth struct{}

func (toneSynth) SampleRate() int { return 16000 }

func (toneSynth) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	return tts.PCMYielder(yield, 16000)(bytes.Repeat([]byte{0x01, 0x02}, 16000*12/100))
}

// A user turn in audio mode is streamed as the microphone would: the whole
// utterance, in order, in 20ms frames at real-time pace, then a second of
// silence so the bot's VAD ends the turn.
func TestUserAudioIsPacedThenFollowedBySilence(t *testing.T) {
	bot := newFakeRTVIServer(t)
	c, err := dial(t.Context(), bot.url())
	if err != nil {
		t.Fatal(err)
	}
	s := &session{client: c, userTTS: tts.New("toneTTS", toneSynth{})}

	start := time.Now()
	if err := s.sendUserAudio(t.Context(), "hello"); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	// The closing handshake follows every frame sent, so once it completes the
	// bot has read them all.
	c.close()

	const chunkBytes = 16000 * audioChunkMS / 1000 * 2
	utterance := bytes.Repeat([]byte{0x01, 0x02}, 16000*12/100)
	wantChunks := len(utterance)/chunkBytes + silenceTailMS/audioChunkMS
	in := bot.messages(rtviws.TypeRawAudio)
	if len(in) != wantChunks {
		t.Fatalf("the bot got %d frames, want %d", len(in), wantChunks)
	}
	var chunks [][]byte
	for _, m := range in {
		var d rtviws.RawAudioData
		if err := json.Unmarshal(m.Data, &d); err != nil {
			t.Fatal(err)
		}
		pcm, err := base64.StdEncoding.DecodeString(d.Audio)
		if err != nil {
			t.Fatal(err)
		}
		if len(pcm) != chunkBytes || d.SampleRate != 16000 {
			t.Fatalf("a chunk of %d bytes at %d Hz, want %d bytes at 16000 Hz", len(pcm), d.SampleRate, chunkBytes)
		}
		chunks = append(chunks, pcm)
	}
	speech := bytes.Join(chunks[:len(utterance)/chunkBytes], nil)
	if !bytes.Equal(speech, utterance) {
		t.Fatal("the utterance did not arrive whole and in order")
	}
	for _, pcm := range chunks[len(utterance)/chunkBytes:] {
		if !bytes.Equal(pcm, make([]byte, chunkBytes)) {
			t.Fatal("the tail after the utterance is not silence")
		}
	}
	// Real-time pacing: each frame waits out its 20ms, so the turn takes as long
	// to send as it lasts rather than going out at once.
	if minimum := time.Duration(wantChunks-1) * audioChunkMS * time.Millisecond; elapsed < minimum {
		t.Fatalf("sent %d frames in %s, want at least %s at real-time pace", wantChunks, elapsed, minimum)
	}
}

// A keypad turn of one key is one dtmf message carrying that key.
func TestADTMFTurnOfOneKeyIsOneMessageWithThatKey(t *testing.T) {
	m := dtmf("1")
	d, ok := m.Data.(rtvi.DTMFData)
	if m.Type != rtvi.TypeDTMF || !ok || len(d.Buttons) != 1 || d.Buttons[0] != "1" {
		t.Fatalf("message = %+v, want one dtmf message carrying the key 1", m)
	}
}

// A marker the bot gave no kind fails a check on what it meant, saying so.
func TestAMarkerOfNoKindFails(t *testing.T) {
	exp := Expectation{Event: EventLLMMarker, Marker: MarkerComplete}
	fail := func(reason, kind string) *Failure { return &Failure{Reason: reason, Kind: kind} }
	f := checkMarker(Event{Kind: EventLLMMarker, Text: "?"}, exp, fail)
	if f == nil || !strings.Contains(f.Reason, "no known kind") || f.Kind != FailureMarkerMismatch {
		t.Fatalf("failure = %+v, want it to say the marker has no known kind", f)
	}
}

// A response that does not follow the marker protocol a check asks about fails
// as a marker format failure, and text that misses a pattern as an unmatched
// one.
func TestFormatAndPatternChecksHaveTheirOwnKinds(t *testing.T) {
	fail := func(reason, kind string) *Failure { return &Failure{Reason: reason, Kind: kind} }
	one := 1
	exp := Expectation{Event: EventLLMMarker, Markers: &one}
	ev := Event{Kind: EventLLMMarker, Raw: "<x> hi <x>", Markers: []string{"<x>"}}
	if f := checkMarker(ev, exp, fail); f == nil || f.Kind != FailureMarkerFormat {
		t.Fatalf("failure = %+v, want a marker format failure", f)
	}
	patterns := Expectation{Event: EventLLMResponse, TextMatches: Patterns{"^Hello"}}
	if f := checkTextMatches("Goodbye", patterns, fail); f == nil || f.Kind != FailureTextUnmatched {
		t.Fatalf("failure = %+v, want an unmatched text failure", f)
	}
}

// countingJudge counts how often it is asked.
type countingJudge struct{ asked int }

func (*countingJudge) AddUserMessage(string)      {}
func (*countingJudge) AddAssistantMessage(string) {}

func (j *countingJudge) Evaluate(context.Context, string) JudgeVerdict {
	j.asked++
	return JudgeVerdict{Verdict: VerdictYes}
}

func (j *countingJudge) EvaluateCall(context.Context, string, map[string]any, string) JudgeVerdict {
	j.asked++
	return JudgeVerdict{Verdict: VerdictYes}
}

// A reply with nothing in it yet has nothing to judge, so the judge is not
// asked and the reply is waited on.
func TestAnEmptyAggregateSkipsTheJudge(t *testing.T) {
	judge := &countingJudge{}
	s := &session{judge: judge}
	exp := Expectation{Event: EventLLMResponse, Eval: "describes the weather"}
	status, _ := s.evaluateAggregate(t.Context(), "   ", exp)
	if status != statusContinue || judge.asked != 0 {
		t.Fatalf("status %q after %d judge calls, want continue without asking", status, judge.asked)
	}
}

// requiredLevel is the report level a one-turn scenario asks the bot for, and
// whether it asks at all.
func requiredLevel(expect ...Expectation) (rtvi.FunctionCallReportLevel, bool) {
	s := &session{scenario: oneTurn("x", expect...)}
	return s.requiredReportLevel()
}

// The harness raises the bot's function-call report level only as far as the
// scenario's assertions need.
func TestRequiredReportLevel(t *testing.T) {
	t.Run("none without a function call", func(t *testing.T) {
		if _, want := requiredLevel(Expectation{Event: EventLLMResponse}); want {
			t.Fatal("asked for a report level with no call asserted")
		}
	})
	t.Run("none for a bare function call", func(t *testing.T) {
		// A bare function_call matches any call, which is how it is normalized.
		bare := Expectation{Event: EventFunctionCall, Calls: []FunctionCall{{}}}
		if _, want := requiredLevel(bare); want {
			t.Fatal("asked for a report level to assert only that a call happened")
		}
	})
	t.Run("a judged reply after a call does not need args", func(t *testing.T) {
		// A reply is judged on the spoken conversation only, so the call needs
		// no more than a name to be matched.
		level, want := requiredLevel(
			Expectation{Event: EventFunctionCall, Calls: []FunctionCall{{Name: "submit"}}},
			Expectation{Event: EventLLMResponse, Eval: "confirms what it submitted"},
		)
		if !want || level != rtvi.ReportName {
			t.Fatalf("level = %v (%v), want the name only", level, want)
		}
	})
}

// The harness asks for the raw VAD events only when the scenario references
// them, as an assertion or as what a turn is scheduled from.
func TestNeedsVADEvents(t *testing.T) {
	needs := func(turn Turn) bool {
		return (&session{scenario: &Scenario{Name: "t", Turns: []Turn{turn}}}).needsVADEvents()
	}
	if needs(Turn{User: "x", Expect: []Expectation{{Event: EventLLMResponse}}}) {
		t.Error("asked for VAD events with none referenced")
	}
	if !needs(Turn{User: "x", Expect: []Expectation{{Event: EventVADUserStartedSpeaking}}}) {
		t.Error("did not ask for VAD events a turn expects")
	}
	if !needs(Turn{
		User: "x", Expect: []Expectation{{Event: EventLLMResponse}},
		SendAfter: &SendAfter{Event: EventVADUserStoppedSpeaking, DelayMS: 2000},
	}) {
		t.Error("did not ask for the VAD event a turn is scheduled from")
	}
}
