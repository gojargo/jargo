package eval

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/processor/rtvi"
)

// fakeJudge returns queued verdicts without calling a real LLM.
type fakeJudge struct {
	mu       sync.Mutex
	verdicts []string
	// calls are the criteria a reply was judged against, segments the bot's
	// reply segments it was given, and callAsks the calls it judged.
	calls    []string
	segments []string
	callAsks []judgedCall
	closed   bool
}

// judgedCall is one call put to the judge.
type judgedCall struct {
	name      string
	args      map[string]any
	criterion string
}

func newFakeJudge(verdicts ...string) *fakeJudge { return &fakeJudge{verdicts: verdicts} }

func (j *fakeJudge) Close(context.Context) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.closed = true
	return nil
}

func (*fakeJudge) AddUserMessage(string) {}

func (j *fakeJudge) AddAssistantMessage(text string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.segments = append(j.segments, text)
}

func (j *fakeJudge) Evaluate(_ context.Context, criterion string, _ *bool) JudgeVerdict {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.calls = append(j.calls, criterion)
	return j.nextVerdict()
}

func (j *fakeJudge) EvaluateCall(_ context.Context, name string, args map[string]any, criterion string) JudgeVerdict {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.callAsks = append(j.callAsks, judgedCall{name, args, criterion})
	return j.nextVerdict()
}

func (j *fakeJudge) nextVerdict() JudgeVerdict {
	if len(j.verdicts) == 0 {
		panic("the judge was asked more than expected") //nolint:forbidigo // a test double's guard
	}
	v := j.verdicts[0]
	j.verdicts = j.verdicts[1:]
	return JudgeVerdict{Verdict: v, Reason: "(" + v + ")"}
}

func (j *fakeJudge) isClosed() bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.closed
}

// newMatcher is a matcher over a stream of its own. A nil judge stays a nil
// Judge.
func newMatcher(judge *fakeJudge, botAudio bool) *ExpectationMatcher {
	var j Judge
	if judge != nil {
		j = judge
	}
	return NewExpectationMatcher(newStream(botAudio), j, NewTrace())
}

// match matches exp from now with budgetMS, as expectation 0 of turn 0.
func match(t *testing.T, m *ExpectationMatcher, exp Expectation, budgetMS int) (*AssertionFailure, error) {
	t.Helper()
	return m.Match(t.Context(), &exp, time.Now(), budgetMS, 0, 0)
}

// mustMatch matches exp and fails the test on an error.
func mustMatch(t *testing.T, m *ExpectationMatcher, exp Expectation, budgetMS int) *AssertionFailure {
	t.Helper()
	f, err := match(t, m, exp, budgetMS)
	if err != nil {
		t.Fatalf("match: %v", err)
	}
	return f
}

// enqueue puts events on the stream's queue as they are, without stamping
// them.
func enqueue(s *EventStream, events ...map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.queue = append(s.queue, events...)
	s.signalLocked()
}

// A judge's "no" fails the reply only once the reply is over.
func TestJudgeNoIsProvisional(t *testing.T) {
	exp := Expectation{Event: EventResponse, Eval: "gives the weather"}

	t.Run("a no while the bot speaks waits for the rest", func(t *testing.T) {
		judge := newFakeJudge(VerdictNo, VerdictYes)
		m := newMatcher(judge, true)
		m.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		m.stream.Append(map[string]any{keyType: EventResponse, "text": "Sure. Why don't skeletons fight?"})
		go func() {
			time.Sleep(100 * time.Millisecond)
			m.stream.Append(map[string]any{keyType: EventResponse, "text": "The weather is 75 degrees."})
		}()
		if f := mustMatch(t, m, exp, 5000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		if !slices.Equal(judge.calls, []string{"gives the weather", "gives the weather"}) {
			t.Fatalf("calls = %v", judge.calls)
		}
	})
	t.Run("a no after the bot stopped waits out the transcription", func(t *testing.T) {
		m := newMatcher(newFakeJudge(VerdictNo, VerdictYes), true)
		m.stream.Append(map[string]any{keyType: EventResponse, "text": "Transferring you now."})
		go func() {
			time.Sleep(100 * time.Millisecond)
			m.stream.Append(map[string]any{keyType: EventResponse, "text": "The boots cost $299."})
		}()
		if f := mustMatch(t, m, exp, 5000); f != nil {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("a no on a finished reply fails", func(t *testing.T) {
		grace := judgeNoGrace
		judgeNoGrace = 100 * time.Millisecond
		t.Cleanup(func() { judgeNoGrace = grace })
		m := newMatcher(newFakeJudge(VerdictNo), true)
		m.stream.Append(map[string]any{keyType: EventResponse, "text": "I like turtles."})
		if f := mustMatch(t, m, exp, 5000); f == nil || f.Kind != FailureJudgeNo {
			t.Fatalf("failure = %v, want judge_no", f)
		}
	})
	t.Run("a failed judge call is no verdict at once", func(t *testing.T) {
		// A provider out of credits, say: nothing was decided about the reply.
		m := newMatcher(newFakeJudge(VerdictNone), true)
		m.stream.Append(map[string]any{keyType: EventResponse, "text": "I like turtles."})
		f := mustMatch(t, m, exp, 5000)
		if f == nil || f.Kind != FailureJudgeNoVerdict || !strings.Contains(f.Reason, "judge said none: (none)") {
			t.Fatalf("failure = %v, want judge_no_verdict", f)
		}
	})
	t.Run("a failed judge call on a single event is no verdict", func(t *testing.T) {
		m := newMatcher(newFakeJudge(VerdictNone), false)
		m.stream.Append(map[string]any{keyType: EventUserTranscription, "text": "hello"})
		f := mustMatch(t, m, Expectation{Event: EventUserTranscription, Eval: "says hello"}, 1000)
		if f == nil || f.Kind != FailureJudgeNoVerdict || !strings.Contains(f.Reason, "judge said none: (none)") {
			t.Fatalf("failure = %v, want judge_no_verdict", f)
		}
	})
}

// An empty reply followed by the bot ending the call fails the wait at once.
func TestAnEmptyReplyThenTheBotEndingFailsAtOnce(t *testing.T) {
	m := newMatcher(nil, false)
	m.stream.Append(map[string]any{keyType: EventLLMResponse, "text": ""})
	m.stream.BotEnded()
	_, err := match(t, m, Expectation{Event: EventLLMResponse, TextContains: "Berlin"}, 5000)
	if !errors.Is(err, ErrBotEnded) {
		t.Fatalf("err = %v, want ErrBotEnded", err)
	}
}

// An absent: true expectation passes on a quiet window and fails on arrival.
func TestMatchAbsent(t *testing.T) {
	t.Run("passes when no event arrives", func(t *testing.T) {
		m := newMatcher(nil, false)
		if f := mustMatch(t, m, Expectation{Event: EventLLMResponse, Absent: true}, 100); f != nil {
			t.Fatalf("failure = %v", f)
		}
		if !strings.Contains(m.LastMatchText, `no "llm_response"`) {
			t.Fatalf("matched = %q", m.LastMatchText)
		}
	})
	t.Run("passes when the bot ends the call", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.stream.BotEnded()
		if f := mustMatch(t, m, Expectation{Event: EventLLMResponse, Absent: true}, 5000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		if !strings.Contains(m.LastMatchText, "before the bot ended the call") {
			t.Fatalf("matched = %q", m.LastMatchText)
		}
	})
	t.Run("fails when the event arrives", func(t *testing.T) {
		m := newMatcher(nil, false)
		enqueue(m.stream, map[string]any{keyType: EventLLMResponse, "text": "I repeat myself"})
		exp := Expectation{Event: EventLLMResponse, Absent: true}
		f, err := m.Match(t.Context(), &exp, time.Now(), 100, 3, 2)
		if err != nil || f == nil || f.TurnIndex != 3 || f.ExpectationIndex != 2 ||
			!strings.Contains(f.Reason, "I repeat myself") {
			t.Fatalf("failure = %v, %v", f, err)
		}
	})
	t.Run("ignores the rest of the matched reply", func(t *testing.T) {
		m := newMatcher(nil, false)
		// The bot began one reply, and the previous expectation matched its
		// first segment; a later segment of the same reply is not a new
		// response.
		m.stream.Append(map[string]any{keyType: EventBotStartedSpeaking})
		m.LastMatchText = "first sentence"
		m.lastMatchAt = time.Now()
		enqueue(m.stream, map[string]any{keyType: EventResponse, "text": "second sentence"})
		if f := mustMatch(t, m, Expectation{Event: EventResponse, Absent: true}, 100); f != nil {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("fails on a reply begun after the match", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.lastMatchAt = time.Now()
		time.Sleep(time.Millisecond)
		m.stream.Append(map[string]any{keyType: EventBotStartedSpeaking})
		enqueue(m.stream, map[string]any{keyType: EventResponse, "text": "I repeat myself"})
		f := mustMatch(t, m, Expectation{Event: EventResponse, Absent: true}, 100)
		if f == nil || !strings.Contains(f.Reason, "I repeat myself") {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("ignores other event types", func(t *testing.T) {
		m := newMatcher(nil, false)
		// Unrelated events in the window must not trip the absence check.
		enqueue(m.stream, map[string]any{keyType: EventUserStoppedSpeaking},
			map[string]any{keyType: EventTTSResponse, "text": "spoken"})
		if f := mustMatch(t, m, Expectation{Event: EventLLMResponse, Absent: true}, 100); f != nil {
			t.Fatalf("failure = %v", f)
		}
	})
}

// call is a function call event.
func call(name string, args map[string]any, stopped bool) map[string]any {
	eventType := EventFunctionCall
	if stopped {
		eventType = EventFunctionCallStopped
	}
	return map[string]any{keyType: eventType, "name": name, "args": args}
}

// eval: on a function call puts each matched call to the judge.
func TestFunctionCallEval(t *testing.T) {
	const criterion = "the suggestion is about tracing, for Jennifer Smith"
	args := map[string]any{"title": "OpenTelemetry tracing", "speaker": "Jennifer Smith"}
	exp := func(names ...string) Expectation {
		calls := make([]FunctionCall, 0, len(names))
		for _, n := range names {
			calls = append(calls, FunctionCall{Name: n})
		}
		return Expectation{Event: EventFunctionCall, Calls: calls, Eval: criterion}
	}

	t.Run("yes passes and the ask carries the args", func(t *testing.T) {
		judge := newFakeJudge(VerdictYes)
		m := newMatcher(judge, false)
		m.stream.Append(call("submit_session_suggestion", args, false))
		if f := mustMatch(t, m, exp("submit_session_suggestion"), 1000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		want := []judgedCall{{"submit_session_suggestion", args, criterion}}
		if !reflect.DeepEqual(judge.callAsks, want) {
			t.Fatalf("asks = %v", judge.callAsks)
		}
		if m.LastMatchText != "submit_session_suggestion" {
			t.Fatalf("matched = %q", m.LastMatchText)
		}
	})
	t.Run("no fails with the judge's reason", func(t *testing.T) {
		m := newMatcher(newFakeJudge(VerdictNo), false)
		m.stream.Append(call("submit_session_suggestion", args, false))
		f := mustMatch(t, m, exp("submit_session_suggestion"), 1000)
		if f == nil || f.Kind != FailureJudgeNo || !strings.Contains(f.Reason, "judge said no: (no)") ||
			!strings.Contains(f.Reason, "submit_session_suggestion(") {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("continue counts as no", func(t *testing.T) {
		m := newMatcher(newFakeJudge(VerdictContinue), false)
		m.stream.Append(call("submit_session_suggestion", args, false))
		f := mustMatch(t, m, exp("submit_session_suggestion"), 1000)
		if f == nil || f.Kind != FailureJudgeNo || !strings.Contains(f.Reason, "judge said continue") {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("a failed judge call is no verdict", func(t *testing.T) {
		m := newMatcher(newFakeJudge(VerdictNone), false)
		m.stream.Append(call("submit_session_suggestion", args, false))
		f := mustMatch(t, m, exp("submit_session_suggestion"), 1000)
		if f == nil || f.Kind != FailureJudgeNoVerdict || !strings.Contains(f.Reason, "judge said none: (none)") {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("every listed call is judged", func(t *testing.T) {
		judge := newFakeJudge(VerdictYes, VerdictNo)
		m := newMatcher(judge, false)
		m.stream.Append(call("lookup", map[string]any{"q": "a"}, false))
		m.stream.Append(call("submit", map[string]any{"q": "b"}, false))
		f := mustMatch(t, m, exp("lookup", "submit"), 1000)
		if f == nil || f.Kind != FailureJudgeNo {
			t.Fatalf("failure = %v", f)
		}
		if len(judge.callAsks) != 2 || judge.callAsks[0].name != "lookup" || judge.callAsks[1].name != "submit" {
			t.Fatalf("asks = %v", judge.callAsks)
		}
	})
	t.Run("verbatim args are checked before the judge", func(t *testing.T) {
		judge := newFakeJudge() // would panic if the judge were asked
		m := newMatcher(judge, false)
		m.stream.Append(call("submit", map[string]any{"speaker": "someone else"}, false))
		f := mustMatch(t, m, Expectation{
			Event: EventFunctionCall, Eval: criterion,
			Calls: []FunctionCall{{Name: "submit", Args: map[string]any{"speaker": "Jennifer Smith"}}},
		}, 200)
		if f == nil || f.Kind != FailureFunctionArgsMismatch || len(judge.callAsks) != 0 {
			t.Fatalf("failure = %v, asks %v", f, judge.callAsks)
		}
	})
	t.Run("no judge fails before matching", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.stream.Append(call("submit", args, false))
		if f := mustMatch(t, m, exp("submit"), 1000); f == nil || f.Kind != FailureNoJudge {
			t.Fatalf("failure = %v", f)
		}
	})
}

// The pass, fail or continue decision over accumulated response text.
func TestEvaluateAggregate(t *testing.T) {
	ctx := t.Context()
	t.Run("text_contains present passes", func(t *testing.T) {
		m := newMatcher(nil, false)
		exp := Expectation{Event: EventLLMResponse, TextContains: "Paris"}
		status, reason := m.evaluateAggregate(ctx, "The capital is Paris.", &exp)
		if status != aggregatePass || reason != "" {
			t.Fatalf("status = %q %q", status, reason)
		}
	})
	t.Run("text_contains absent continues", func(t *testing.T) {
		m := newMatcher(nil, false)
		exp := Expectation{Event: EventLLMResponse, TextContains: "Paris"}
		if status, _ := m.evaluateAggregate(ctx, "Let me check on that.", &exp); status != aggregateContinue {
			t.Fatalf("status = %q", status)
		}
	})
	t.Run("text_excludes fails the reply as soon as it appears", func(t *testing.T) {
		m := newMatcher(nil, false)
		enqueue(m.stream, map[string]any{keyType: EventLLMResponse, "text": "Berlin.●"},
			map[string]any{keyType: EventLLMResponse, "text": "The capital of Germany."})
		f := mustMatch(t, m, Expectation{Event: EventLLMResponse, TextContains: "Germany", TextExcludes: "●"}, 100)
		if f == nil || f.Kind != FailureTextPresent || !strings.Contains(f.Reason, `"Berlin.●"`) {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("text_matches checks the accumulated reply", func(t *testing.T) {
		exp := Expectation{Event: EventLLMResponse, TextContains: "Germany", TextMatches: Patterns{`^●`}}
		for first, kind := range map[string]string{"● Berlin.": "", "Berlin. ●": FailureTextUnmatched} {
			m := newMatcher(nil, false)
			enqueue(m.stream, map[string]any{keyType: EventLLMResponse, "text": first},
				map[string]any{keyType: EventLLMResponse, "text": "The capital of Germany."})
			f := mustMatch(t, m, exp, 100)
			got := ""
			if f != nil {
				got = f.Kind
			}
			if got != kind {
				t.Fatalf("%q: kind = %q, want %q", first, got, kind)
			}
		}
	})
	for _, tc := range []struct{ verdict, status, reason string }{
		{VerdictYes, aggregatePass, "judge said yes"},
		{VerdictNo, aggregateFail, "judge said no"},
		{VerdictContinue, aggregateContinue, "judge said continue"},
	} {
		t.Run("eval "+tc.verdict, func(t *testing.T) {
			m := newMatcher(newFakeJudge(tc.verdict), false)
			exp := Expectation{Event: EventLLMResponse, Eval: "describes the weather"}
			status, reason := m.evaluateAggregate(ctx, "It's 75 and sunny.", &exp)
			if status != tc.status || !strings.Contains(reason, tc.reason) {
				t.Fatalf("status = %q %q", status, reason)
			}
		})
	}
	t.Run("eval on an empty aggregate skips the judge", func(t *testing.T) {
		judge := newFakeJudge() // would panic if the judge were asked
		m := newMatcher(judge, false)
		exp := Expectation{Event: EventLLMResponse, Eval: "describes the weather"}
		if status, _ := m.evaluateAggregate(ctx, "   ", &exp); status != aggregateContinue || len(judge.calls) != 0 {
			t.Fatalf("status = %q after %d asks", status, len(judge.calls))
		}
	})
}

// checkPayload checks event against exp as expectation 0 of turn 0.
func checkPayload(event map[string]any, exp Expectation) *AssertionFailure {
	return newMatcher(nil, false).checkPayload(event, &exp, 0, 0)
}

// text_contains resolves against whichever event carries the text.
func TestTextContainsResolution(t *testing.T) {
	t.Run("on one event's text", func(t *testing.T) {
		exp := Expectation{Event: EventLLMResponse, TextContains: "Paris"}
		if f := checkPayload(map[string]any{keyType: EventLLMResponse, "text": "It's Paris."}, exp); f != nil {
			t.Fatalf("failure = %v", f)
		}
		if f := checkPayload(map[string]any{keyType: EventLLMResponse, "text": "London."}, exp); f == nil {
			t.Fatal("London passed for Paris")
		}
	})
	t.Run("on a user transcription, ignoring spacing", func(t *testing.T) {
		exp := Expectation{Event: EventUserTranscription, TextContains: "the capital of"}
		event := map[string]any{keyType: EventUserTranscription, "transcript": " What  is the  capital  of Germany?"}
		if f := checkPayload(event, exp); f != nil {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("on a user transcription's transcript", func(t *testing.T) {
		exp := Expectation{Event: EventUserTranscription, TextContains: "hello"}
		if f := checkPayload(map[string]any{keyType: EventUserTranscription, "transcript": "hello world"}, exp); f != nil {
			t.Fatalf("failure = %v", f)
		}
		f := checkPayload(map[string]any{keyType: EventUserTranscription, "transcript": "bye"}, exp)
		if f == nil || !strings.Contains(f.Reason, "does not contain") {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("text_excludes", func(t *testing.T) {
		exp := Expectation{Event: EventLLMResponse, TextExcludes: "●"}
		if f := checkPayload(map[string]any{keyType: EventLLMResponse, "text": "Berlin."}, exp); f != nil {
			t.Fatalf("failure = %v", f)
		}
		f := checkPayload(map[string]any{keyType: EventLLMResponse, "text": "Berlin.● The capital."}, exp)
		if f == nil || f.Kind != FailureTextPresent || !strings.Contains(f.Reason, `contains "●"`) {
			t.Fatalf("failure = %v", f)
		}
		// Spacing is ignored, as for text_contains.
		exp = Expectation{Event: EventLLMResponse, TextExcludes: "not sure"}
		if f := checkPayload(map[string]any{keyType: EventLLMResponse, "text": "I'm not  sure."}, exp); f == nil {
			t.Fatal("excluded text passed")
		}
	})
	t.Run("text_matches", func(t *testing.T) {
		exp := Expectation{Event: EventLLMResponse, TextMatches: Patterns{`^\[\[voice:[^\]]*\]\]`, `\]\]\s*\S`}}
		ok := map[string]any{keyType: EventLLMResponse, "text": "[[voice: emotion=warm]] Sure thing."}
		if f := checkPayload(ok, exp); f != nil {
			t.Fatalf("failure = %v", f)
		}
		f := checkPayload(map[string]any{keyType: EventLLMResponse, "text": "Sure. [[voice: x]]"}, exp)
		if f == nil || f.Kind != FailureTextUnmatched || !strings.Contains(f.Reason, `does not match "^`) {
			t.Fatalf("failure = %v", f)
		}
		// Each pattern must match; the failure names the first that did not.
		f = checkPayload(map[string]any{keyType: EventLLMResponse, "text": "[[voice: emotion=warm]]"}, exp)
		if f == nil || f.Kind != FailureTextUnmatched || !strings.Contains(f.Reason, `"\\]\\]\\s*\\S"`) {
			t.Fatalf("failure = %v", f)
		}
	})
}

// marker: matches the kind the bot stamped on the marker, not its text.
func TestMarkerCheck(t *testing.T) {
	check := func(kind string, marker string) *AssertionFailure {
		return checkPayload(map[string]any{keyType: EventLLMMarker, "text": "?", "kind": kind},
			Expectation{Event: EventLLMMarker, Marker: marker})
	}
	if f := check(MarkerComplete, MarkerComplete); f != nil {
		t.Fatalf("failure = %v", f)
	}
	if f := check(MarkerShort, MarkerComplete); f == nil || f.Kind != FailureMarkerMismatch {
		t.Fatalf("failure = %v", f)
	}
	// incomplete accepts short or long.
	if check(MarkerShort, MarkerIncomplete) != nil || check(MarkerLong, MarkerIncomplete) != nil ||
		check(MarkerComplete, MarkerIncomplete) == nil {
		t.Fatal("incomplete does not mean short or long")
	}
	// A marker of no kind fails.
	if f := check("", MarkerComplete); f == nil || !strings.Contains(f.Reason, "no known kind") {
		t.Fatalf("failure = %v", f)
	}
}

// Checks on how the raw LLM text was laid out around its markers.
func TestMarkerFormatChecks(t *testing.T) {
	check := func(raw string, exp Expectation) *AssertionFailure {
		exp.Event = EventLLMMarker
		return checkPayload(map[string]any{
			keyType: EventLLMMarker, "text": "●", "kind": MarkerComplete, "raw": raw,
			"markers": []string{"●", "◐", "○"},
		}, exp)
	}
	yes, no := true, false
	n := func(i int) *int { return &i }

	if f := check("Hi ● there ○", Expectation{}); f != nil {
		t.Fatalf("no format fields: failure = %v", f)
	}
	// marker_first.
	if check("● Hi", Expectation{MarkerFirst: &yes}) != nil || check("  ● Hi", Expectation{MarkerFirst: &yes}) != nil {
		t.Fatal("a leading marker is not first")
	}
	if f := check("Hi ●", Expectation{MarkerFirst: &yes}); f == nil || f.Kind != FailureMarkerFormat ||
		!strings.Contains(f.Reason, "does not start") {
		t.Fatalf("failure = %v", f)
	}
	if check("Hi ●", Expectation{MarkerFirst: &no}) != nil || check("● Hi", Expectation{MarkerFirst: &no}) == nil {
		t.Fatal("marker_first: false reads wrong")
	}
	// No marker at all cannot be marker-first.
	if check("Hi", Expectation{MarkerFirst: &yes}) == nil {
		t.Fatal("no marker passed as first")
	}
	// markers.
	if check("● Hi", Expectation{Markers: n(1)}) != nil || check("Hi", Expectation{Markers: n(0)}) != nil ||
		check("● Hi ◐ ○", Expectation{Markers: n(3)}) != nil {
		t.Fatal("a right marker count failed")
	}
	if f := check("● Hi ●", Expectation{Markers: n(1)}); f == nil || f.Kind != FailureMarkerFormat ||
		!strings.Contains(f.Reason, "holds 2 marker(s)") {
		t.Fatalf("failure = %v", f)
	}
	// text_after.
	if check("● Hi", Expectation{TextAfter: &yes}) != nil || check("●", Expectation{TextAfter: &no}) != nil ||
		check("● ", Expectation{TextAfter: &no}) != nil {
		t.Fatal("a right text_after failed")
	}
	if check("● Hi", Expectation{TextAfter: &no}) == nil || check("●", Expectation{TextAfter: &yes}) == nil {
		t.Fatal("a wrong text_after passed")
	}
	// No marker means nothing follows one.
	if check("Hi", Expectation{TextAfter: &yes}) == nil {
		t.Fatal("text after no marker passed")
	}
}

// Expectations match in order unless one says check_from: turn_start.
func TestCheckFrom(t *testing.T) {
	matchAt := func(t *testing.T, m *ExpectationMatcher, expIdx int, exp Expectation) string {
		t.Helper()
		if exp.WithinMS == 0 {
			exp.WithinMS = 500
		}
		f, err := m.Match(t.Context(), &exp, time.Now(), exp.WithinMS, 0, expIdx)
		if err != nil {
			t.Fatalf("match: %v", err)
		}
		if f == nil {
			return ""
		}
		return f.Reason
	}
	delegate := Expectation{Event: EventFunctionCall, Calls: []FunctionCall{{Name: "delegate"}}}
	duck := func() *ExpectationMatcher {
		m := newMatcher(nil, false)
		enqueue(m.stream,
			map[string]any{keyType: EventLLMResponse, "text": "Why did the duck cross?"},
			map[string]any{keyType: EventFunctionCall, "name": "delegate", "args": map[string]any{}},
			map[string]any{keyType: EventLLMResponse, "text": "I'll let you know."})
		return m
	}

	t.Run("an ordered expectation passes over an earlier reply", func(t *testing.T) {
		m := duck()
		if r := matchAt(t, m, 0, delegate); r != "" {
			t.Fatal(r)
		}
		if r := matchAt(t, m, 1, Expectation{Event: EventLLMResponse, TextContains: "duck"}); !strings.Contains(r, "duck") {
			t.Fatalf("reason = %q, want the earlier reply passed over", r)
		}
	})
	t.Run("a flagged expectation claims the earlier reply", func(t *testing.T) {
		m := duck()
		if r := matchAt(t, m, 0, delegate); r != "" {
			t.Fatal(r)
		}
		exp := Expectation{Event: EventLLMResponse, TextContains: "duck", CheckFrom: CheckFromTurnStart}
		if r := matchAt(t, m, 1, exp); r != "" {
			t.Fatal(r)
		}
		if !strings.Contains(m.LastMatchText, "duck") {
			t.Fatalf("matched = %q", m.LastMatchText)
		}
	})
	t.Run("the expectations after a flagged one are still ordered after it", func(t *testing.T) {
		m := newMatcher(nil, false)
		enqueue(m.stream,
			map[string]any{keyType: EventLLMResponse, "text": "Transferring you now."},
			map[string]any{keyType: EventFunctionCall, "name": "transfer_to_agent", "args": map[string]any{}},
			map[string]any{keyType: EventLLMResponse, "text": "The Rocket Boots do 500 mph."})
		steps := []Expectation{
			{Event: EventFunctionCall, Calls: []FunctionCall{{Name: "transfer_to_agent"}}},
			{Event: EventLLMResponse, TextContains: "Transferring", CheckFrom: CheckFromTurnStart},
			{Event: EventLLMResponse, TextContains: "Boots"},
			// Nothing new after the second reply.
			{Event: EventLLMResponse, Absent: true, WithinMS: 100},
		}
		for i, exp := range steps {
			if r := matchAt(t, m, i, exp); r != "" {
				t.Fatalf("expectation %d: %s", i, r)
			}
		}
	})
	t.Run("a flagged call is found from before the reply it follows", func(t *testing.T) {
		m := newMatcher(nil, false)
		enqueue(m.stream,
			map[string]any{keyType: EventFunctionCall, "name": "delegate", "args": map[string]any{}},
			map[string]any{
				keyType: EventFunctionCallStopped, "name": "run_integration_tests",
				"args": map[string]any{"tool_call_id": "t1", keyCanceled: true},
			},
			map[string]any{keyType: EventLLMResponse, "text": "Okay, I've stopped the tests."})
		steps := []Expectation{
			delegate,
			{Event: EventLLMResponse, TextContains: "stopped"},
			{
				Event: EventFunctionCallStopped, CheckFrom: CheckFromTurnStart,
				Calls: []FunctionCall{{Name: "run_integration_tests", Args: map[string]any{keyCanceled: true}}},
			},
		}
		for i, exp := range steps {
			if r := matchAt(t, m, i, exp); r != "" {
				t.Fatalf("expectation %d: %s", i, r)
			}
		}
	})
	t.Run("a flagged expectation looks back no further than its turn", func(t *testing.T) {
		m := newMatcher(nil, false)
		enqueue(m.stream, map[string]any{keyType: EventFunctionCall, "name": "check_ci_status", "args": map[string]any{}})
		// The previous turn's.
		_, err := m.stream.NextEvent(t.Context(), EventFunctionCall, time.Now().Add(time.Second), FromNow, nil)
		if err != nil {
			t.Fatal(err)
		}
		m.ResetTurn()
		exp := Expectation{
			Event: EventFunctionCall, CheckFrom: CheckFromTurnStart,
			Calls: []FunctionCall{{Name: "check_ci_status"}},
		}
		if r := matchAt(t, m, 0, exp); !strings.Contains(r, `"check_ci_status" not seen`) {
			t.Fatalf("reason = %q", r)
		}
	})
	t.Run("a call claimed by one expectation is not matched again", func(t *testing.T) {
		m := newMatcher(nil, false)
		enqueue(m.stream, map[string]any{keyType: EventFunctionCall, "name": "delegate", "args": map[string]any{}})
		if r := matchAt(t, m, 0, delegate); r != "" {
			t.Fatal(r)
		}
		again := delegate
		again.CheckFrom = CheckFromTurnStart
		if r := matchAt(t, m, 1, again); !strings.Contains(r, `"delegate" not seen`) {
			t.Fatalf("reason = %q", r)
		}
	})
}

// A criterion on the expression the bot performed is a judge's to grade, and
// accumulates what the turn performed.
func TestAnExpressionCheckIsJudgedAndAggregates(t *testing.T) {
	if !slices.Contains(JudgeableEvents, EventExpression) {
		t.Fatal("expression is not judgeable")
	}
	judged := Expectation{Event: EventExpression, Eval: "the bot laughs"}
	if !judged.Aggregates() {
		t.Fatal("an expression check with a criterion does not aggregate")
	}
	bare := Expectation{Event: EventExpression}
	if bare.Aggregates() {
		t.Fatal("a bare expression expectation aggregates")
	}
	judge := newFakeJudge(VerdictContinue, VerdictYes)
	m := newMatcher(judge, true)
	enqueue(m.stream, map[string]any{keyType: EventExpression, "text": "<<excited>>"},
		map[string]any{keyType: EventExpression, "text": "[[laughter]]"})
	if f := mustMatch(t, m, judged, 1000); f != nil {
		t.Fatalf("failure = %v", f)
	}
	if !slices.Equal(judge.segments, []string{"<<excited>>", "[[laughter]]"}) {
		t.Fatalf("segments = %v, want each expression put to the judge", judge.segments)
	}
}

// A call's arguments compare the way equality reads them, whichever text format
// each side was decoded from.
func TestCallArgumentsCompareAcrossNumberTypes(t *testing.T) {
	event := map[string]any{
		keyType: EventFunctionCall, "name": "book",
		"args": map[string]any{"party": 2.0, "times": []any{19.0, 20.0}},
	}
	if !callMatches(event, "book", map[string]any{"party": 2, "times": []any{19, 20}}) {
		t.Fatal("2 and 2.0 compare different")
	}
	if callMatches(event, "book", map[string]any{"party": 3}) {
		t.Fatal("2 and 3 compare equal")
	}
}

// A metrics expectation takes the next report holding its kind of metric.
func TestMetricsExpectation(t *testing.T) {
	t.Run("it takes the report holding its metric", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.stream.Append(metricsEvent(map[string]any{"processing": measurement("STT#0", 0.01)}))
		m.stream.Append(metricsEvent(map[string]any{"ttfb": measurement("LLM#0", 0.4)}))
		m.stream.Append(metricsEvent(map[string]any{
			"ttfb":       measurement("DeepgramSTTService#0", 0.31),
			"processing": measurement("DeepgramSTTService#0", 0.02),
		}))
		if f := mustMatch(t, m, Expectation{Event: EventMetrics, Metric: "ttfb", Processor: "STT"}, 1000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		var kept map[string]any
		if err := json.Unmarshal([]byte(m.LastMatchText), &kept); err != nil {
			t.Fatal(err)
		}
		if want := map[string]any{"ttfb": measurement("DeepgramSTTService#0", 0.31)}; !reflect.DeepEqual(kept, want) {
			t.Fatalf("matched %v, want %v", kept, want)
		}
	})
	t.Run("the zero reported at start is not a measurement", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.stream.Append(metricsEvent(map[string]any{"ttfb": measurement("STT#0", 0)}))
		if _, err := match(t, m, Expectation{Event: EventMetrics, Metric: "ttfb"}, 50); !errors.Is(err, ErrTimeout) {
			t.Fatalf("err = %v, want a timeout", err)
		}
	})
	t.Run("a bare expectation takes any report", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.stream.Append(metricsEvent(map[string]any{
			"tokens": []any{map[string]any{"prompt_tokens": 12.0, "completion_tokens": 3.0}},
		}))
		if f := mustMatch(t, m, Expectation{Event: EventMetrics}, 1000); f != nil {
			t.Fatalf("failure = %v", f)
		}
	})
	t.Run("a report from earlier in the turn is found from turn start", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.ResetTurn()
		m.stream.Append(metricsEvent(map[string]any{"ttfb": measurement("STT#0", 0.3)}))
		m.stream.Append(map[string]any{keyType: EventUserTranscription, keyTranscript: "hello"})
		if f := mustMatch(t, m, Expectation{Event: EventUserTranscription}, 1000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		lookingBack := Expectation{Event: EventMetrics, Metric: "ttfb", CheckFrom: CheckFromTurnStart}
		if f, err := m.Match(t.Context(), &lookingBack, time.Now(), 1000, 0, 1); f != nil || err != nil {
			t.Fatalf("failure = %v, err = %v", f, err)
		}
	})
}

// A server_message expectation reads the bot's data as text.
func TestServerMessageExpectation(t *testing.T) {
	serverMessage := func(t *testing.T, m *ExpectationMatcher, data any) {
		t.Helper()
		m.stream.Append(m.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeServerMessage, data)))
	}
	t.Run("it reads the data and records it", func(t *testing.T) {
		m := newMatcher(nil, false)
		data := map[string]any{"kind": "result", "ttfb": 0.31}
		serverMessage(t, m, data)
		exp := Expectation{Event: EventServerMessage, TextContains: `"kind": "result"`}
		if f := mustMatch(t, m, exp, 1000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		var recorded map[string]any
		if err := json.Unmarshal([]byte(m.LastMatchText), &recorded); err != nil || !reflect.DeepEqual(recorded, data) {
			t.Fatalf("recorded %q, want the data", m.LastMatchText)
		}
	})
	t.Run("a content check judges the next message", func(t *testing.T) {
		// As for any event: the next one of the type is the one checked.
		m := newMatcher(nil, false)
		serverMessage(t, m, map[string]any{"kind": "progress"})
		f := mustMatch(t, m, Expectation{Event: EventServerMessage, TextContains: `"kind": "result"`}, 1000)
		if f == nil || f.Kind != FailureTextMismatch {
			t.Fatalf("failure = %v, want a text mismatch", f)
		}
	})
}
