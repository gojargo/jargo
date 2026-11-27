package eval

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"sort"
	"strings"
	"time"
)

// Matching a scenario's expectations against the bot's events.
//
// ExpectationMatcher waits on the run's EventStream for the event an
// expectation names and verifies it: a payload check, a judge verdict, the
// aggregation of a reply across its segments, the any-order matching of a
// turn's function calls, and the inverted absent: check.

// JudgeNoGrace is how long a judge's "no" waits for more of the reply once the
// bot has stopped speaking: the transcription of its last sentence lands after
// the stop.
const JudgeNoGrace = 2 * time.Second

// judgeNoGrace is JudgeNoGrace, which a test shortens.
//
//nolint:gochecknoglobals // a test seam for a fixed duration
var judgeNoGrace = JudgeNoGrace

// The outcomes of checking the accumulated reply text.
const (
	aggregatePass      = "pass"
	aggregateFail      = "fail"
	aggregateContinue  = "continue"
	aggregateNoVerdict = "no_verdict"
)

// errNoJudge is the reason of an eval: with no judge to put it to.
const errNoJudge = "scenario uses 'eval:' but no judge could be built"

// callMatches reports whether a call event is the one a calls: entry names,
// with the arguments it gives. An empty name matches any call, and nil args any
// arguments.
func callMatches(event map[string]any, name string, args map[string]any) bool {
	if name != "" && event[keyName] != name {
		return false
	}
	if args == nil {
		return true
	}
	actual, _ := event[keyArgs].(map[string]any)
	for k, v := range args {
		got, ok := actual[k]
		if !ok || !sameValue(got, v) {
			return false
		}
	}
	return true
}

// sameValue compares a value the bot reported against the one a scenario
// expects, the way equality reads both: the two are decoded from different text
// formats, JSON with one number type and YAML with integers apart, so 3 and 3.0
// are equal, inside lists and mappings too.
func sameValue(actual, want any) bool {
	if af, ok := toFloat(actual); ok {
		wf, ok := toFloat(want)
		return ok && af == wf
	}
	switch a := actual.(type) {
	case []any:
		w, ok := want.([]any)
		return ok && slices.EqualFunc(a, w, sameValue)
	case map[string]any:
		w, ok := want.(map[string]any)
		if !ok || len(a) != len(w) {
			return false
		}
		for k, v := range a {
			wv, ok := w[k]
			if !ok || !sameValue(v, wv) {
				return false
			}
		}
		return true
	}
	return reflect.DeepEqual(actual, want)
}

// toFloat reports a numeric value as a float64, whichever numeric type carries
// it, and whether it was numeric at all.
func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint64:
		return float64(n), true
	default:
		return 0, false
	}
}

// ExpectationMatcher matches one expectation at a time against the event
// stream.
//
// Expected events must appear in order, with unmatched events allowed in
// between. A reply with a content check aggregates its segments and re-checks
// on each, so an interim "Let me check" is rolled past rather than taken for
// the answer. A turn's function calls match by name in any order.
type ExpectationMatcher struct {
	stream *EventStream
	judge  Judge
	trace  *Trace

	// LastMatchText is the text content of the most recently matched event
	// (the bot's response, or a user transcript), surfaced to verbose progress.
	// Empty for events with no text (llm_started, speaking events).
	LastMatchText string
	// lastMatchAt is when the most recent expectation matched, so an absence
	// check can tell a reply that began after it from the rest of the one it
	// matched.
	lastMatchAt time.Time
}

// NewExpectationMatcher builds a matcher over stream. judge answers eval:
// assertions, or is nil when the scenario has none.
func NewExpectationMatcher(stream *EventStream, judge Judge, trace *Trace) *ExpectationMatcher {
	return &ExpectationMatcher{stream: stream, judge: judge, trace: trace}
}

// ResetTurn forgets the previous turn's events.
func (m *ExpectationMatcher) ResetTurn() { m.stream.BeginTurn() }

// Match waits for the expected event and verifies it. Its budget is budgetMS
// measured from anchor; turnIdx and expIdx are where the failure is recorded.
//
// Stale bot output is not filtered here: the stream drops it on an
// interruption and the driver before each send.
//
// It returns the failure, or nil when the expectation is satisfied. The error
// is ErrTimeout when no matching event arrives at all (so the caller can report
// that none arrived; a response that arrives but never satisfies the content
// check is a failure instead), ErrBotEnded when the bot ends the call before
// the expectation is met (an absent: expectation is met instead, since the
// event can no longer arrive), a *HarnessError, or ctx's error.
func (m *ExpectationMatcher) Match(
	ctx context.Context, exp *Expectation, anchor time.Time, budgetMS, turnIdx, expIdx int,
) (*AssertionFailure, error) {
	deadline := anchor.Add(time.Duration(budgetMS) * time.Millisecond)
	m.LastMatchText = ""

	if exp.Absent {
		return m.matchAbsent(ctx, exp, deadline, budgetMS, turnIdx, expIdx)
	}
	// Where in the turn the expectation looks for its event from: only what
	// arrives from here on, or back from the start of the turn.
	since := m.stream.TurnPosition()
	if exp.CheckFrom == CheckFromTurnStart {
		since = 0
	}
	if exp.Aggregates() {
		return m.matchAggregating(ctx, exp, deadline, budgetMS, turnIdx, expIdx, since)
	}
	if slices.Contains(functionCallEvents, exp.Event) {
		// A call expectation holds the set of calls the turn should make; it
		// completes only when all are found, in any order (a response arriving
		// does not short-circuit it).
		return m.matchFunctionCalls(ctx, exp, deadline, turnIdx, expIdx, since)
	}
	if exp.Event == EventMetrics {
		return nil, m.matchMetrics(ctx, exp, deadline, since)
	}
	return m.matchOne(ctx, exp, deadline, turnIdx, expIdx, since)
}

// matchMetrics matches the next metrics report holding the expected kind of
// metric.
//
// A report without it is passed over, as an event of another type is: a bot
// sends several per turn, and which one comes first is not the bot's behavior.
func (m *ExpectationMatcher) matchMetrics(
	ctx context.Context, exp *Expectation, deadline time.Time, since int,
) error {
	wanted := cmp.Or(exp.Metric, "any")
	if exp.Processor != "" {
		wanted += " from " + pyRepr(exp.Processor)
	}
	m.trace.Log(fmt.Sprintf("match: waiting for 'metrics' (%s)", wanted))
	event, err := m.stream.NextEvent(ctx, EventMetrics, deadline, since, func(event map[string]any) bool {
		return len(MatchingMetrics(event, exp.Metric, exp.Processor)) > 0
	})
	if err != nil {
		return err
	}
	kept, err := json.Marshal(MatchingMetrics(event, exp.Metric, exp.Processor))
	if err != nil {
		return err
	}
	m.LastMatchText = string(kept)
	m.lastMatchAt = time.Now()
	return nil
}

// MatchingMetrics is the measurements of a metrics event that an expectation's
// filters take, by kind: metric is the kind to keep, or empty for every kind,
// and processor text the measuring processor's name must contain, or empty for
// any processor. It is empty when none is kept.
//
// The zero a pipeline reports for every processor as it starts is a
// placeholder, not a measurement, and is left out.
func MatchingMetrics(event map[string]any, metric, processor string) map[string][]any {
	kept := map[string][]any{}
	data, _ := event[keyData].(map[string]any)
	for kind, entries := range data {
		if metric != "" && kind != metric {
			continue
		}
		list, _ := entries.([]any)
		for _, entry := range list {
			measurement, ok := entry.(map[string]any)
			if !ok {
				continue
			}
			if value, ok := toFloat(measurement["value"]); ok && value == 0 {
				continue
			}
			if processor != "" {
				name := ""
				if v, ok := measurement["processor"]; ok {
					name = pyStr(v)
				}
				if !strings.Contains(name, processor) {
					continue
				}
			}
			kept[kind] = append(kept[kind], measurement)
		}
	}
	return kept
}

// matchOne matches a single event and checks its payload and judge assertion.
func (m *ExpectationMatcher) matchOne(
	ctx context.Context, exp *Expectation, deadline time.Time, turnIdx, expIdx, since int,
) (*AssertionFailure, error) {
	m.trace.Log(fmt.Sprintf("match: waiting for %q", exp.Event))
	event, err := m.stream.NextEvent(ctx, exp.Event, deadline, since, nil)
	if err != nil {
		return nil, err
	}
	if f := m.checkPayload(event, exp, turnIdx, expIdx); f != nil {
		return f, nil
	}
	f := m.checkJudge(ctx, event, exp, turnIdx, expIdx)
	if f == nil {
		m.LastMatchText = matchSummary(event)
		m.lastMatchAt = time.Now()
	}
	return f, nil
}

// matchAggregating accumulates reply segments until the content check passes
// or fails.
//
//nolint:cyclop,nilnil // a nil failure with no error is a met expectation
func (m *ExpectationMatcher) matchAggregating(
	ctx context.Context, exp *Expectation, deadline time.Time, budgetMS, turnIdx, expIdx, since int,
) (*AssertionFailure, error) {
	if exp.Eval != "" && m.judge == nil {
		return failure(exp, turnIdx, expIdx, errNoJudge, FailureNoJudge), nil
	}

	var checks []string
	if exp.TextContains != "" {
		checks = append(checks, "text_contains")
	}
	if exp.Eval != "" {
		checks = append(checks, "eval")
	}
	m.trace.Log(fmt.Sprintf("match: waiting for %q (%s)", exp.Event, strings.Join(checks, "+")))
	var (
		aggregate  string
		lastReason string
		seenAny    bool
		pending    map[string]any
	)
	for {
		var event map[string]any
		if pending != nil {
			event, pending = pending, nil
		} else {
			var err error
			event, err = m.stream.NextEvent(ctx, exp.Event, deadline, since, nil)
			if errors.Is(err, ErrTimeout) && seenAny {
				return m.unsatisfied(exp, turnIdx, expIdx, budgetMS, lastReason), nil
			}
			if err != nil {
				// No response at all: the caller reports the missing event.
				return nil, err
			}
		}

		seenAny = true
		delta := eventText(event)
		aggregate += delta
		if f := textExcluded(aggregate, exp, turnIdx, expIdx); f != nil {
			return f, nil
		}
		// Feed each segment to the judge as its own assistant message, so it
		// judges the bot's reply in the conversation's context (the cumulative
		// aggregate is kept only for text_contains and the match summary).
		if exp.Eval != "" && m.judge != nil {
			m.judge.AddAssistantMessage(delta)
		}
		status, reason := m.evaluateAggregate(ctx, aggregate, exp)
		m.trace.Log(fmt.Sprintf("eval: %s (aggregate=%q) %s", status, strings.TrimSpace(aggregate), reason))
		switch status {
		case aggregateNoVerdict:
			return failure(exp, turnIdx, expIdx, reason, FailureJudgeNoVerdict), nil
		case aggregatePass:
			if f := textUnmatched(aggregate, exp, turnIdx, expIdx); f != nil {
				return f, nil
			}
			m.LastMatchText = aggregate
			m.lastMatchAt = time.Now()
			return nil, nil
		case aggregateFail:
			// Only the judge can affirmatively fail an aggregate, and only once
			// the reply is complete: a "no" on a reply still being spoken, or
			// whose last sentence is still being transcribed, is a "continue"
			// that the next segment may turn into a "yes".
			pending = m.restOfReply(ctx, exp.Event, deadline, since)
			if pending == nil {
				return failure(exp, turnIdx, expIdx, reason, FailureJudgeNo), nil
			}
			m.trace.Log("eval: no, but the reply goes on: judging the rest")
		}
		// Continue: wait for the next segment, separated by a space so
		// sentences do not run together ("...that. The weather...").
		aggregate += " "
		lastReason = reason
	}
}

// restOfReply is the next segment of a reply the judge rejected, or nil when the
// reply is over.
//
// While the bot is speaking the next segment is awaited within the turn's
// budget; once it has stopped, only for the grace its last sentence's
// transcription needs.
func (m *ExpectationMatcher) restOfReply(
	ctx context.Context, eventType string, deadline time.Time, since int,
) map[string]any {
	until := deadline
	if !m.stream.BotSpeaking() {
		if grace := time.Now().Add(judgeNoGrace); grace.Before(until) {
			until = grace
		}
	}
	event, err := m.stream.NextEvent(ctx, eventType, until, since, nil)
	if err != nil {
		return nil
	}
	return event
}

// unsatisfied is the failure of a reply that never satisfied its check within
// the budget. With eval: the judge kept saying continue; without it the only way
// to be unsatisfied is a missing substring.
func (m *ExpectationMatcher) unsatisfied(
	exp *Expectation, turnIdx, expIdx, budgetMS int, lastReason string,
) *AssertionFailure {
	m.trace.Log("eval: timeout, not satisfied: " + lastReason)
	kind := FailureTextMismatch
	if exp.Eval != "" {
		kind = FailureJudgeContinue
	}
	return failure(exp, turnIdx, expIdx, fmt.Sprintf("not satisfied within %dms: %s", budgetMS, lastReason), kind)
}

// matchAbsent passes when no event of the type arrives before the deadline; an
// arriving one fails at once, with its content.
//
// A reply reaches the stream in segments, one per pause in the bot's speech, so
// a response that continues the reply an earlier expectation matched is not a
// new one: only a response the bot began after that match counts.
//
//nolint:nilnil // a nil failure with no error is a met expectation
func (m *ExpectationMatcher) matchAbsent(
	ctx context.Context, exp *Expectation, deadline time.Time, budgetMS, turnIdx, expIdx int,
) (*AssertionFailure, error) {
	m.trace.Log(fmt.Sprintf("match: expecting NO %q for %dms", exp.Event, budgetMS))
	var event map[string]any
	for {
		var err error
		event, err = m.stream.NextEvent(ctx, exp.Event, deadline, FromNow, nil)
		switch {
		case errors.Is(err, ErrTimeout):
			// The quiet window held: absence confirmed.
			m.LastMatchText = fmt.Sprintf("no %q for %dms", exp.Event, budgetMS)
			return nil, nil
		case errors.Is(err, ErrBotEnded):
			m.LastMatchText = fmt.Sprintf("no %q before the bot ended the call", exp.Event)
			return nil, nil
		case err != nil:
			return nil, err
		}
		if exp.Event == EventResponse && !m.replyBeganAfterLastMatch() {
			m.trace.Log(fmt.Sprintf("absent: the matched reply goes on, not a new one: %q", matchSummary(event)))
			continue
		}
		break
	}
	return failure(exp, turnIdx, expIdx, fmt.Sprintf("expected no %q within %dms, but one arrived: %s",
		exp.Event, budgetMS, matchSummary(event)), FailureUnexpectedEvent), nil
}

// replyBeganAfterLastMatch reports whether the bot started a reply since the
// most recent match, by its LLM or its speech.
func (m *ExpectationMatcher) replyBeganAfterLastMatch() bool {
	var began time.Time
	for _, eventType := range []string{EventLLMStarted, EventBotStartedSpeaking} {
		if at, ok := m.stream.LatestEventTime(eventType); ok && at.After(began) {
			began = at
		}
	}
	return began.After(m.lastMatchAt)
}

// matchFunctionCalls matches every call in the expectation, in any order,
// within the budget; else it fails naming the call that was missing or whose
// args did not match.
//
// The calls are looked for from since on, so they may arrive in any order, and
// a call the LLM corrects and repeats still satisfies it. With eval:, each
// matched call is also put to the judge, and the first one it rejects fails the
// expectation.
//
//nolint:cyclop,gocognit,nilnil // a nil failure with no error is a met expectation
func (m *ExpectationMatcher) matchFunctionCalls(
	ctx context.Context, exp *Expectation, deadline time.Time, turnIdx, expIdx, since int,
) (*AssertionFailure, error) {
	if exp.Eval != "" && m.judge == nil {
		return failure(exp, turnIdx, expIdx, errNoJudge, FailureNoJudge), nil
	}

	var matched []string
	for _, spec := range exp.Calls {
		want := spec.Args
		if len(want) == 0 {
			want = nil
		}
		m.trace.Log(fmt.Sprintf("match: waiting for %q (%s)", exp.Event, spec.Signature()))
		event, err := m.stream.NextEvent(ctx, exp.Event, deadline, since, func(ev map[string]any) bool {
			return callMatches(ev, spec.Name, want)
		})
		if errors.Is(err, ErrTimeout) {
			// A call of the right name with the wrong arguments is a different
			// failure from the call never being made, and the bot's arguments are
			// what the reader needs to see.
			var near []any
			for _, ev := range m.stream.UnclaimedSince(since) {
				if ev[keyType] == exp.Event && (spec.Name == "" || ev[keyName] == spec.Name) {
					near = append(near, ev[keyArgs])
				}
			}
			if want != nil && len(near) > 0 {
				return failure(exp, turnIdx, expIdx, fmt.Sprintf("no %q call had args %v (saw %v)",
					spec.Name, want, near), FailureFunctionArgsMismatch), nil
			}
			missing := spec.Name
			if missing == "" {
				missing = "any function"
			}
			seen := "none"
			if len(matched) > 0 {
				seen = strings.Join(matched, ", ")
			}
			return failure(exp, turnIdx, expIdx, fmt.Sprintf("function call %q not seen (matched: %s)",
				missing, seen), FailureMissingFunctionCall), nil
		}
		if err != nil {
			return nil, err
		}
		if f := m.checkCallJudge(ctx, event, exp, turnIdx, expIdx); f != nil {
			return f, nil
		}
		matched = append(matched, fmt.Sprint(event[keyName]))
	}

	m.LastMatchText = strings.Join(matched, ", ")
	if m.LastMatchText == "" {
		m.LastMatchText = "function call"
	}
	m.lastMatchAt = time.Now()
	return nil, nil
}

// checkCallJudge puts a matched call to the judge when the expectation carries
// an eval:.
//
// The judge is asked about the call by name and arguments, in the light of the
// spoken conversation so far. A call is not a partial reply, so a continue
// fails it like a no.
func (m *ExpectationMatcher) checkCallJudge(
	ctx context.Context, event map[string]any, exp *Expectation, turnIdx, expIdx int,
) *AssertionFailure {
	if exp.Eval == "" {
		return nil
	}
	// matchFunctionCalls fails before matching when there is no judge.
	name, _ := event[keyName].(string)
	if name == "" {
		name = "?"
	}
	args, _ := event[keyArgs].(map[string]any)
	if args == nil {
		args = map[string]any{}
	}
	verdict := m.judge.EvaluateCall(withEvalPipeline(ctx, PipelineJudge), name, args, exp.Eval)
	m.trace.Log(fmt.Sprintf("eval: %s (%s) %s", verdict.Verdict, matchSummary(event), verdict.Reason))
	if verdict.Passed() {
		return nil
	}
	return failure(exp, turnIdx, expIdx, fmt.Sprintf("eval %q on %s: judge said %s: %s",
		exp.Eval, matchSummary(event), verdict.Verdict, verdict.Reason), judgeFailureKind(verdict))
}

// evaluateAggregate checks the accumulated reply text: pass, fail, or continue
// for more text.
//
// A missing substring is continue; only the judge can fail, and a failed judge
// call is no_verdict.
func (m *ExpectationMatcher) evaluateAggregate(
	ctx context.Context, aggregate string, exp *Expectation,
) (status, reason string) {
	if exp.TextContains != "" && !textContains(aggregate, exp.TextContains) {
		return aggregateContinue, fmt.Sprintf("does not contain %q", exp.TextContains)
	}
	if exp.Eval != "" {
		if strings.TrimSpace(aggregate) == "" {
			return aggregateContinue, "no response text yet"
		}
		// The reply segments were added to the judge's conversation in the
		// aggregation loop; the judge evaluates that context, not aggregate.
		verdict := m.judge.Evaluate(withEvalPipeline(ctx, PipelineJudge), exp.Eval, exp.AllowContinue)
		switch verdict.Verdict {
		case VerdictNone:
			return aggregateNoVerdict, "judge said none: " + verdict.Reason
		case VerdictNo:
			return aggregateFail, "judge said no: " + verdict.Reason
		case VerdictContinue:
			return aggregateContinue, "judge said continue: " + verdict.Reason
		}
		return aggregatePass, "judge said yes: " + verdict.Reason
	}
	return aggregatePass, ""
}

// checkPayload applies the payload-level checks to a matched event, returning
// the first failure or nil.
func (m *ExpectationMatcher) checkPayload(
	event map[string]any, exp *Expectation, turnIdx, expIdx int,
) *AssertionFailure {
	content := eventText(event)
	if exp.TextContains != "" && !textContains(content, exp.TextContains) {
		return failure(exp, turnIdx, expIdx,
			fmt.Sprintf("text %q does not contain %q", content, exp.TextContains), FailureTextMismatch)
	}
	if f := textExcluded(content, exp, turnIdx, expIdx); f != nil {
		return f
	}
	if f := textUnmatched(content, exp, turnIdx, expIdx); f != nil {
		return f
	}
	if exp.Marker != "" {
		kind, _ := event["kind"].(string)
		wanted := []string{exp.Marker}
		if exp.Marker == MarkerIncomplete {
			wanted = []string{MarkerShort, MarkerLong}
		}
		if !slices.Contains(wanted, kind) {
			if kind == "" {
				kind = "of no known kind"
			}
			return failure(exp, turnIdx, expIdx,
				fmt.Sprintf("marker %q is %s, expected %s", content, kind, exp.Marker), FailureMarkerMismatch)
		}
	}
	if problem := checkMarkerFormat(event, exp); problem != "" {
		return failure(exp, turnIdx, expIdx, problem, FailureMarkerFormat)
	}
	return nil
}

// checkMarkerFormat is what is wrong with the shape of the response's raw text,
// or "" when nothing is.
//
// The checks read the raw text the marker event carries against every marker
// the bot recognizes, so they see what the LLM wrote before the bot held any of
// it back.
func checkMarkerFormat(event map[string]any, exp *Expectation) string {
	if exp.MarkerFirst == nil && exp.Markers == nil && exp.TextAfter == nil {
		return ""
	}
	raw, _ := event["raw"].(string)
	firstAt, first, count := findMarkers(raw, knownMarkers(event[keyMarkers]))
	excerpt := raw
	if r := []rune(excerpt); len(r) > 80 {
		excerpt = string(r[:80])
	}
	excerpt = fmt.Sprintf("%q", excerpt)
	if exp.Markers != nil && count != *exp.Markers {
		return fmt.Sprintf("raw text holds %d marker(s), expected %d: %s", count, *exp.Markers, excerpt)
	}
	if exp.MarkerFirst != nil {
		isFirst := firstAt >= 0 && strings.TrimSpace(raw[:firstAt]) == ""
		if isFirst != *exp.MarkerFirst {
			return fmt.Sprintf("raw text %s with a marker, expected it %s: %s",
				pick(isFirst, "starts", "does not start"), pick(*exp.MarkerFirst, "to", "not to"), excerpt)
		}
	}
	if exp.TextAfter == nil {
		return ""
	}
	if firstAt < 0 {
		return "raw text holds no marker to check text after: " + excerpt
	}
	if hasAfter := strings.TrimSpace(raw[firstAt+len(first):]) != ""; hasAfter != *exp.TextAfter {
		return fmt.Sprintf("raw text has %s after the marker, expected %s: %s",
			pick(hasAfter, "text", "nothing"), pick(*exp.TextAfter, "text", "nothing"), excerpt)
	}
	return ""
}

// knownMarkers is every marker a marker event says the bot recognizes.
func knownMarkers(v any) []string {
	switch k := v.(type) {
	case []string:
		return k
	case []any:
		var known []string
		for _, m := range k {
			if s, ok := m.(string); ok {
				known = append(known, s)
			}
		}
		return known
	}
	return nil
}

// findMarkers is where the earliest of the known markers sits in raw and which
// it is (-1 when raw holds none), and how many markers raw holds.
func findMarkers(raw string, known []string) (firstAt int, first string, count int) {
	firstAt = -1
	for _, marker := range known {
		if marker == "" {
			continue
		}
		count += strings.Count(raw, marker)
		at := strings.Index(raw, marker)
		if at >= 0 && (firstAt < 0 || at < firstAt || at == firstAt && marker < first) {
			firstAt, first = at, marker
		}
	}
	return firstAt, first, count
}

// pick is yes when cond holds, no otherwise.
func pick(cond bool, yes, no string) string {
	if cond {
		return yes
	}
	return no
}

// checkJudge runs the judge assertion when the expectation carries an eval:.
func (m *ExpectationMatcher) checkJudge(
	ctx context.Context, event map[string]any, exp *Expectation, turnIdx, expIdx int,
) *AssertionFailure {
	if exp.Eval == "" {
		return nil
	}
	if m.judge == nil {
		return failure(exp, turnIdx, expIdx, errNoJudge, FailureNoJudge)
	}
	content := eventText(event)
	if content == "" {
		return failure(exp, turnIdx, expIdx,
			fmt.Sprintf("event has no text/transcript to judge: %v", event), FailureNoContent)
	}
	m.judge.AddAssistantMessage(content)
	verdict := m.judge.Evaluate(withEvalPipeline(ctx, PipelineJudge), exp.Eval, exp.AllowContinue)
	if verdict.Passed() {
		return nil
	}
	said := VerdictNo
	if verdict.Verdict == VerdictNone {
		said = VerdictNone
	}
	return failure(exp, turnIdx, expIdx,
		fmt.Sprintf("eval %q: judge said %s: %s", exp.Eval, said, verdict.Reason), judgeFailureKind(verdict))
}

// failure is the failure record for exp.
func failure(exp *Expectation, turnIdx, expIdx int, reason, kind string) *AssertionFailure {
	return &AssertionFailure{
		TurnIndex: turnIdx, ExpectationIndex: expIdx, EventName: exp.Event, Reason: reason, Kind: kind,
	}
}

// matchSummary is a short label for a matched event: the call signature for a
// function call, the event's text otherwise.
func matchSummary(event map[string]any) string {
	if event[keyType] != EventFunctionCall {
		return eventText(event)
	}
	name, _ := event[keyName].(string)
	if name == "" {
		name = "?"
	}
	args, _ := event[keyArgs].(map[string]any)
	keys := slices.Collect(maps.Keys(args))
	sort.Strings(keys) // a map iterates in random order, and this goes in a report
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, args[k]))
	}
	return name + "(" + strings.Join(parts, ", ") + ")"
}

// eventText is the text an event carries: reply events use text,
// user_transcription transcript.
func eventText(event map[string]any) string {
	if text, _ := event[keyText].(string); text != "" {
		return text
	}
	transcript, _ := event[keyTranscript].(string)
	return transcript
}

// textExcluded is the failure for text_excludes when content holds it, else
// nil.
func textExcluded(content string, exp *Expectation, turnIdx, expIdx int) *AssertionFailure {
	if exp.TextExcludes == "" || !textContains(content, exp.TextExcludes) {
		return nil
	}
	return failure(exp, turnIdx, expIdx,
		fmt.Sprintf("text %q contains %q", strings.TrimSpace(content), exp.TextExcludes), FailureTextPresent)
}

// textUnmatched is the failure for the first text_matches pattern content
// misses, else nil.
func textUnmatched(content string, exp *Expectation, turnIdx, expIdx int) *AssertionFailure {
	pattern, missed := exp.TextMatches.firstUnmatched(content)
	if !missed {
		return nil
	}
	text := strings.TrimSpace(content)
	if r := []rune(text); len(r) > 200 {
		text = string(r[:200])
	}
	return failure(exp, turnIdx, expIdx, fmt.Sprintf("text %q does not match %q", text, pattern), FailureTextUnmatched)
}

// textContains reports whether needle occurs in content, whatever the spacing.
func textContains(content, needle string) bool {
	return strings.Contains(strings.Join(strings.Fields(content), " "), strings.Join(strings.Fields(needle), " "))
}

// judgeFailureKind is the kind of a check the judge did not pass: a no, or no
// verdict when its call failed.
func judgeFailureKind(verdict JudgeVerdict) string {
	if verdict.Verdict == VerdictNone {
		return FailureJudgeNoVerdict
	}
	return FailureJudgeNo
}
