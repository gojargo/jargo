package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/tts"
)

// defaultTimeout is the latency budget for an expectation that sets no
// within_ms. It is generous so that an expectation without one waits long
// enough for a slow LLM or TTS response, and for a function-call round trip,
// rather than failing on latency. Set within_ms to assert on timing.
const defaultTimeout = 60 * time.Second

// botReadyTimeout is how long the bot has to announce readiness. It is its own,
// much shorter budget: a bot that never answers the handshake is not a valid
// eval target, and there is nothing to wait for.
const botReadyTimeout = 10 * time.Second

// errBotNotReady is returned when the bot never completes the RTVI handshake.
//
//nolint:gochecknoglobals // sentinel error
var errBotNotReady = errors.New("eval: bot did not become ready")

// eventBotReady is the translated bot-ready message. The handshake waits for it;
// a scenario cannot assert on it, so it is not in knownEvents.
const eventBotReady = "bot_ready"

const (
	// sendAfterMaxWait is how long a turn waits for the event it is scheduled
	// from before giving up. It is not the expectation budget: nothing is being
	// asserted yet, only waited for.
	sendAfterMaxWait = 30 * time.Second
	// sendAfterPoll is how often the wait re-checks whether the event has been
	// seen.
	sendAfterPoll = 10 * time.Millisecond
)

// errSendAfterNotSeen is returned when a turn's schedule never fires.
//
//nolint:gochecknoglobals // sentinel error
var errSendAfterNotSeen = errors.New("send_after never fired")

// event is a friendly, translated view of an RTVI server message: the level a
// scenario asserts on.
type event struct {
	kind     string         // one of the Event* constants
	text     string         // llm_response: the bot's reply text, joined
	function string         // function_call: the tool name
	args     map[string]any // function_call: the arguments the model produced
}

// session drives one scenario against a connected bot.
type session struct {
	client   *client
	scenario *Scenario
	judge    Judge
	userTTS  *tts.Base // non-nil in audio mode: synthesizes each user turn

	// events carries the translated events the matcher works on. The reader
	// goroutine is the only writer.
	events chan event

	// The reader goroutine owns everything below, and nothing else reads it.
	// Upstream translates on its single event loop, which is what makes the
	// interruption discard safe there; keeping translation on one goroutine is
	// how jargo gets the same guarantee.
	//
	// llmBuf accumulates bot-llm-text between bot-llm-started and bot-llm-stopped.
	llmBuf strings.Builder
	// awaitingLLMRestart is set on an interruption and cleared at the next
	// bot-llm-started. While set, llm_response segments are dropped: the
	// interrupted response can still flush a trailing token after the
	// interruption event, generated before the interrupt propagated, and that
	// straggler must not be attributed to the new turn. The genuinely new
	// response begins at the next bot-llm-started.
	awaitingLLMRestart bool

	// pendingCalls holds function_call events popped while matching a different
	// call, so a turn's calls can be claimed by name in any order. Reset per turn.
	// The matcher goroutine owns it.
	pendingCalls []event

	// seenMu guards seenAt, which the reader writes and a turn's schedule reads.
	seenMu sync.Mutex
	// seenAt is when each kind of event was last seen. A turn scheduled from an
	// event anchors on this, so an event seen earlier in the run fires the
	// schedule at once. It survives the interruption discard: it is a record of
	// what happened, not a queue of what is still to be matched.
	seenAt map[string]time.Time
}

// eventBuffer is how many translated events the reader may run ahead by. It is
// generous so events emitted while the harness is busy streaming a turn's audio
// are not lost before the matcher reads them.
const eventBuffer = 256

// run completes the handshake, then plays every turn.
func (s *session) run(ctx context.Context) (Result, error) {
	res := Result{Scenario: s.scenario.Name}

	s.events = make(chan event, eventBuffer)
	s.seenAt = make(map[string]time.Time)
	go s.readLoop(ctx)

	if err := s.handshake(ctx); err != nil {
		return res, err
	}
	for i, turn := range s.scenario.Turns {
		failures, err := s.runTurn(ctx, turn, i+1)
		if err != nil {
			return res, err
		}
		res.Failures = append(res.Failures, failures...)
		if len(failures) > 0 {
			// Fail fast: a failed turn leaves the conversation in an unknown state,
			// so running the rest just burns another budget per turn. A broken
			// greeting turn should not cost the full budget here and again on the
			// question that follows.
			break
		}
	}
	return res, nil
}

// readLoop translates the bot's RTVI messages into friendly events and hands
// them to the matcher, until the socket closes or ctx is canceled.
func (s *session) readLoop(ctx context.Context) {
	defer close(s.events)
	for {
		select {
		case in, ok := <-s.client.incoming:
			if !ok {
				return
			}
			if ev := s.translate(in); ev != nil {
				// Record the sighting before queueing, so a turn scheduled from
				// this event can anchor on it even if the discard drops the event
				// itself before the matcher gets to it.
				s.seenMu.Lock()
				s.seenAt[ev.kind] = time.Now()
				s.seenMu.Unlock()
				select {
				case s.events <- *ev:
				case <-ctx.Done():
					return
				}
			}
		case <-ctx.Done():
			return
		}
	}
}

// discardInterruptedOutput drops the bot's interrupted, unmatched output. It
// clears the response buffer and drains what the matcher has not taken yet, so a
// greeting, or any prior bot output, that the user just interrupted cannot be
// matched against this turn. A user transcription is kept: a keypress emits its
// transcription immediately before the turn-start interruption, and that
// transcription is the turn's input, not the stale bot output this is meant to
// clear.
//
// It runs on the reader goroutine, the only one that translates, so the buffers
// it clears are never touched concurrently.
func (s *session) discardInterruptedOutput() {
	s.llmBuf.Reset()
	var preserved []event
	for drained := true; drained; {
		select {
		case ev := <-s.events:
			if ev.kind == EventUserTranscription {
				preserved = append(preserved, ev)
			}
		default:
			drained = false
		}
	}
	for _, ev := range preserved {
		select {
		case s.events <- ev:
		default: // the buffer refilled behind us; the event is gone either way
		}
	}
}

// handshake sends client-ready, waits for the bot's bot-ready reply, then asks
// the bot to report as much of a tool call as this scenario asserts on.
func (s *session) handshake(ctx context.Context) error {
	ready := rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: messageID,
		Data: map[string]any{"version": rtvi.ProtocolVersion},
	}
	if err := s.client.send(ctx, ready); err != nil {
		return err
	}
	// bot-ready is a hard gate: a bot that never announces readiness either is
	// not a valid eval target or has not finished starting, and firing turns at a
	// half-started bot produces flaky, hard-to-read failures.
	if _, ok := s.nextMatching(ctx, eventBotReady, time.Now().Add(botReadyTimeout)); !ok {
		return fmt.Errorf("%w within %s", errBotNotReady, botReadyTimeout)
	}
	// Raise the bot's function-call report level for the duration of this eval
	// only, so a bot keeps the secure default for its own clients.
	if level, ok := s.requiredReportLevel(); ok {
		return s.client.send(ctx, configureMessage(level))
	}
	return nil
}

// requiredReportLevel is the least a bot has to report for this scenario's
// assertions to be checkable: everything when a call's arguments are asserted,
// the name when only the name is, and nothing more than the default when the
// scenario merely asserts that a call happened.
func (s *session) requiredReportLevel() (rtvi.FunctionCallReportLevel, bool) {
	needsName := false
	for _, turn := range s.scenario.Turns {
		for _, exp := range turn.Expect {
			for _, call := range exp.Calls {
				if call.Args != nil {
					return rtvi.ReportFull, true
				}
				if call.Name != "" {
					needsName = true
				}
			}
		}
	}
	return rtvi.ReportName, needsName
}

// runTurn sends the user's text and matches the turn's expectations in order.
// Every expectation's budget is anchored at the send, so a stalled turn fails
// within a single budget rather than one per expectation.
func (s *session) runTurn(ctx context.Context, turn Turn, turnNum int) ([]Failure, error) {
	// A turn's tool calls are matched by name in any order; start each turn with
	// an empty buffer so a prior turn's calls cannot carry over.
	s.pendingCalls = nil

	if turn.SendAfter != nil {
		if err := s.waitSendAfter(ctx, *turn.SendAfter); err != nil {
			// The turn's input never went out, so none of its expectations can be
			// judged. Report the schedule itself, at expectation 0.
			name := turn.SendAfter.Event
			if name == "" {
				name = "send_after"
			}
			return []Failure{{
				Turn: turnNum, Expectation: 0, Event: name,
				Reason: fmt.Sprintf("%v", err),
			}}, nil
		}
	}

	if s.userTTS != nil {
		if err := s.sendUserAudio(ctx, turn.User); err != nil {
			return nil, err
		}
	} else if err := s.client.send(ctx, sendText(turn.User)); err != nil {
		return nil, err
	}
	anchor := time.Now()

	// Record the user turn in the judge's conversation, so a later reply is
	// judged in context (a terse "that's four" answering this question, say).
	if s.judge != nil {
		s.judge.AddUserMessage(turn.User)
	}

	var failures []Failure
	for j, exp := range turn.Expect {
		budget := defaultTimeout
		if exp.WithinMS > 0 {
			budget = time.Duration(exp.WithinMS) * time.Millisecond
		}
		fail, absent := s.matchExpectation(ctx, exp, anchor.Add(budget), budget, turnNum, j+1)
		if fail != nil {
			failures = append(failures, *fail)
		}
		if absent {
			// Nothing of the expectation's kind ever arrived, so no later
			// expectation of this turn will match either.
			break
		}
	}
	return failures, nil
}

// waitSendAfter blocks until the turn's schedule fires: until its event has
// been seen and the delay has elapsed on top. An event seen earlier in the run
// anchors the delay at that earlier sighting, so the wait may be over already.
//
// With no event it is a pure time delay, measured from now, which is the
// previous turn's send.
func (s *session) waitSendAfter(ctx context.Context, sa SendAfter) error {
	delay := time.Duration(sa.DelayMS) * time.Millisecond
	if sa.Event == "" {
		return sleep(ctx, delay)
	}

	deadline := time.Now().Add(sendAfterMaxWait)
	ticker := time.NewTicker(sendAfterPoll)
	defer ticker.Stop()
	for {
		if seen, ok := s.eventSeenAt(sa.Event); ok {
			return sleep(ctx, time.Until(seen.Add(delay)))
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("%w: event %q not seen within %s",
				errSendAfterNotSeen, sa.Event, sendAfterMaxWait)
		}
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// eventSeenAt reports when an event of the given kind was last seen, and
// whether one has been at all.
func (s *session) eventSeenAt(kind string) (time.Time, bool) {
	s.seenMu.Lock()
	defer s.seenMu.Unlock()
	seen, ok := s.seenAt[kind]
	return seen, ok
}

// sleep waits out d, or returns early if ctx is canceled. A non-positive d
// returns at once.
func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// matchExpectation waits for an event matching exp, skipping unrelated events,
// until deadline. It returns a failure, or nil on a match, and whether no event
// of the expectation's kind arrived at all, which ends the turn.
//
// Most events match a single event and are checked once. An llm_response
// carrying a content check instead aggregates: it accumulates the text of
// successive response segments and re-checks on each new one, until the check
// passes, the judge rejects, or the budget expires. That is what rolls past an
// interim filler ("Let me check on that.") or an on-connect greeting rather than
// mistaking it for the turn's answer.
func (s *session) matchExpectation(
	ctx context.Context, exp Expectation, deadline time.Time, budget time.Duration, turnNum, expNum int,
) (*Failure, bool) {
	fail := func(reason string) *Failure {
		return &Failure{Turn: turnNum, Expectation: expNum, Event: exp.Event, Reason: reason}
	}
	missing := func() *Failure {
		return fail(fmt.Sprintf("no matching %s event arrived within %s", exp.Event, budget))
	}

	if exp.Absent {
		return s.matchAbsent(ctx, exp, deadline, budget, fail), false
	}
	if exp.Event == EventFunctionCall {
		// A function_call expectation holds the set of calls the turn should make;
		// it completes only when all are found, in any order. A call that never
		// arrives is a failure rather than the end of the turn, so a turn missing
		// both a call and a reply reports both within one budget.
		return s.matchFunctionCalls(ctx, exp, deadline, fail), false
	}
	if !s.aggregates(exp) {
		ev, ok := s.nextMatching(ctx, exp.Event, deadline)
		if !ok {
			return missing(), true
		}
		if f := checkPayload(ev, exp, fail); f != nil {
			return f, false
		}
		return s.checkJudge(ctx, ev, exp, fail), false
	}
	return s.matchAggregate(ctx, exp, deadline, budget, fail, missing)
}

// aggregates reports whether an expectation accumulates successive response
// segments rather than checking the first one. Only a content check on the
// bot's own text does; a bare event, or a check on a user transcription,
// matches once.
func (s *session) aggregates(exp Expectation) bool {
	return exp.Event == EventLLMResponse && (exp.TextContains != "" || exp.Judge != "")
}

// matchAggregate accumulates response segments and re-checks the expectation on
// each one until it is satisfied, affirmatively rejected, or the budget expires.
func (s *session) matchAggregate(
	ctx context.Context, exp Expectation, deadline time.Time, budget time.Duration,
	fail func(string) *Failure, missing func() *Failure,
) (*Failure, bool) {
	if exp.Judge != "" && s.judge == nil {
		return fail("scenario uses 'judge:' but no judge could be built"), false
	}
	var (
		aggregate  string
		lastReason string
		seenAny    bool
	)
	for {
		ev, ok := s.nextMatching(ctx, exp.Event, deadline)
		if !ok {
			if !seenAny {
				return missing(), true // no response at all
			}
			return fail(fmt.Sprintf("not satisfied within %s: %s", budget, lastReason)), false
		}
		seenAny = true
		// Feed each segment to the judge as its own reply message, so it judges
		// the bot's reply in the conversation's context. The cumulative aggregate
		// is kept for the substring check.
		aggregate += ev.text
		if exp.Judge != "" && s.judge != nil {
			s.judge.AddAssistantMessage(ev.text)
		}
		status, reason := s.evaluateAggregate(ctx, aggregate, exp)
		switch status {
		case statusPass:
			return nil, false
		case statusFail:
			return fail(reason), false
		}
		// Continue: wait for the next segment, separated by a space so sentences
		// do not run together ("...that. The weather...").
		aggregate += " "
		lastReason = reason
	}
}

// The outcomes of evaluating the accumulated response text.
const (
	statusPass     = "pass"
	statusFail     = "fail"
	statusContinue = "continue"
)

// evaluateAggregate evaluates the accumulated response text. text_contains is
// monotonic, so a missing substring is a continue and more text may still
// arrive; only the judge can affirmatively fail.
func (s *session) evaluateAggregate(ctx context.Context, aggregate string, exp Expectation) (string, string) {
	if exp.TextContains != "" && !strings.Contains(aggregate, exp.TextContains) {
		return statusContinue, fmt.Sprintf("does not contain %q", exp.TextContains)
	}
	if exp.Judge != "" {
		if strings.TrimSpace(aggregate) == "" {
			return statusContinue, "no response text yet"
		}
		// matchAggregate guarantees a judge exists before aggregating a criterion.
		// The judge evaluates the conversation it was fed, not this aggregate.
		verdict := s.judge.Evaluate(ctx, exp.Judge)
		switch verdict.Verdict {
		case VerdictNo:
			return statusFail, "judge said no: " + verdict.Reason
		case VerdictContinue:
			return statusContinue, "judge said continue: " + verdict.Reason
		}
		return statusPass, "judge said yes: " + verdict.Reason
	}
	return statusPass, ""
}

// matchAbsent is the inverted match: it passes when no event of the
// expectation's type arrives before the deadline. The budget is the whole point
// here, so the expectation holds the turn open for the full window; an arriving
// event fails immediately, and its content goes into the reason so a
// duplicate-output regression shows what the bot said.
func (s *session) matchAbsent(
	ctx context.Context, exp Expectation, deadline time.Time, budget time.Duration,
	fail func(string) *Failure,
) *Failure {
	ev, ok := s.nextMatching(ctx, exp.Event, deadline)
	if !ok {
		return nil // the quiet window held: absence confirmed
	}
	return fail(fmt.Sprintf(
		"expected no %s within %s, but one arrived: %s", exp.Event, budget, ev.summary()))
}

// matchFunctionCalls claims one call for each of the expectation's expected
// calls, from the turn's calls already buffered and those still arriving. It
// passes only when all are claimed within the budget; otherwise it names the
// call that was missing, or whose arguments did not match.
func (s *session) matchFunctionCalls(
	ctx context.Context, exp Expectation, deadline time.Time, fail func(string) *Failure,
) *Failure {
	var matched []string
	for _, want := range exp.Calls {
		ev, ok := s.nextFunctionCall(ctx, want.Name, deadline)
		if !ok {
			name := want.Name
			if name == "" {
				name = "any function"
			}
			seen := "none"
			if len(matched) > 0 {
				seen = strings.Join(matched, ", ")
			}
			return fail(fmt.Sprintf("function call %q not seen (matched: %s)", name, seen))
		}
		if missing := missingArgs(want.Args, ev.args); len(missing) > 0 {
			return fail(fmt.Sprintf("call %q args %v missing expected %v", ev.function, ev.args, missing))
		}
		matched = append(matched, ev.function)
	}
	return nil
}

// nextFunctionCall claims a function_call event for name, where an empty name
// claims any call. A turn's calls can arrive in any order, so it looks first in
// the buffer of calls seen but not yet claimed, then reads on; a call that does
// not match is buffered so another expected call can claim it. Events of other
// kinds are dropped, as in matchExpectation.
func (s *session) nextFunctionCall(ctx context.Context, name string, deadline time.Time) (event, bool) {
	matches := func(ev event) bool { return name == "" || ev.function == name }

	for i, ev := range s.pendingCalls {
		if matches(ev) {
			s.pendingCalls = append(s.pendingCalls[:i], s.pendingCalls[i+1:]...)
			return ev, true
		}
	}
	for {
		ev, ok := s.next(ctx, deadline)
		if !ok {
			return event{}, false
		}
		if ev.kind != EventFunctionCall {
			continue
		}
		if matches(ev) {
			return ev, true
		}
		s.pendingCalls = append(s.pendingCalls, ev)
	}
}

// missingArgs is the subset check on a call's arguments: the entries of want
// that got is missing or disagrees with. Arguments got carries that want does
// not list are ignored.
func missingArgs(want, got map[string]any) map[string]any {
	var missing map[string]any
	for k, v := range want {
		if actual, ok := got[k]; ok && sameArg(actual, v) {
			continue
		}
		if missing == nil {
			missing = make(map[string]any, len(want))
		}
		missing[k] = v
	}
	return missing
}

// sameArg compares an argument the model produced against the one a scenario
// expects. Both sides are decoded from a text format, but not the same one: JSON
// has a single number type where YAML distinguishes integers, so 3 and 3.0 have
// to compare equal. Everything else compares by value.
func sameArg(actual, want any) bool {
	af, aok := toFloat(actual)
	wf, wok := toFloat(want)
	if aok && wok {
		return af == wf
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

// summary is a short label for an event, for a failure reason: the call
// signature for a tool call, and the text content for everything else.
func (e event) summary() string {
	if e.kind != EventFunctionCall {
		return e.text
	}
	name := e.function
	if name == "" {
		name = "?"
	}
	args := make([]string, 0, len(e.args))
	for k, v := range e.args {
		args = append(args, fmt.Sprintf("%s=%v", k, v))
	}
	sort.Strings(args) // a map iterates in random order, and this goes in a report
	return name + "(" + strings.Join(args, ", ") + ")"
}

// checkPayload applies the payload-level checks to a matched event, returning
// the first failure or nil. It is the single-event path; an expectation that
// aggregates goes through evaluateAggregate instead.
func checkPayload(ev event, exp Expectation, fail func(string) *Failure) *Failure {
	if exp.TextContains != "" && !strings.Contains(ev.text, exp.TextContains) {
		return fail(fmt.Sprintf("text %q does not contain %q", ev.text, exp.TextContains))
	}
	return nil
}

// checkJudge runs the judge assertion when the expectation carries one. Today
// that never happens on this path: a criterion is only valid on llm_response,
// and llm_response with a criterion aggregates. It becomes live as soon as the
// scenario format grows another event whose text a judge can read.
func (s *session) checkJudge(
	ctx context.Context, ev event, exp Expectation, fail func(string) *Failure,
) *Failure {
	if exp.Judge == "" {
		return nil
	}
	if s.judge == nil {
		return fail("scenario uses 'judge:' but no judge could be built")
	}
	if ev.text == "" {
		return fail(fmt.Sprintf("event has no text to judge: %s", ev.kind))
	}
	s.judge.AddAssistantMessage(ev.text)
	verdict := s.judge.Evaluate(ctx, exp.Judge)
	if !verdict.Passed() {
		return fail(fmt.Sprintf("judge %q: judge said no: %s", exp.Judge, verdict.Reason))
	}
	return nil
}

// translate converts an RTVI server message into a friendly event, or nil if it
// carries no event on its own (bot-llm-text is accumulated until the response
// ends). It runs on the reader goroutine.
func (s *session) translate(in rtvi.Incoming) *event {
	switch in.Type {
	case rtvi.TypeBotReady:
		return &event{kind: eventBotReady}
	case rtvi.TypeUserStartedSpeaking:
		// A new user turn in audio mode. Drop any leftover bot output from a prior
		// turn so it is not aggregated into this one.
		s.discardInterruptedOutput()
		s.awaitingLLMRestart = true
		return &event{kind: EventUserStartedSpeaking}
	case rtvi.TypeBotInterrupted:
		// The bot's in-flight output was cut off, by a VAD barge-in or by a
		// run-immediately text interrupt. Drop it so only what the bot says after
		// the interruption is matched.
		s.discardInterruptedOutput()
		s.awaitingLLMRestart = true
		return &event{kind: EventBotInterrupted}
	case rtvi.TypeUserStoppedSpeaking:
		return &event{kind: EventUserStoppedSpeaking}
	case rtvi.TypeUserTranscription:
		var d rtvi.UserTranscriptionData
		if json.Unmarshal(in.Data, &d) != nil || !d.Final {
			return nil // only final transcriptions are turn-level events
		}
		return &event{kind: EventUserTranscription, text: d.Text}
	case rtvi.TypeBotLLMStarted, rtvi.TypeBotLLMText, rtvi.TypeBotLLMStopped:
		return s.translateLLM(in)
	case rtvi.TypeLLMFunctionCall:
		var d rtvi.LLMFunctionCallData
		_ = json.Unmarshal(in.Data, &d)
		var args map[string]any
		_ = json.Unmarshal(d.Arguments, &args)
		return &event{kind: EventFunctionCall, function: d.FunctionName, args: args}
	default:
		return nil
	}
}

// next reads the next translated event, blocking until one arrives, the deadline
// passes, or ctx is canceled.
func (s *session) next(ctx context.Context, deadline time.Time) (event, bool) {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return event{}, false
	}
	timer := time.NewTimer(remaining)
	defer timer.Stop()
	select {
	case ev, ok := <-s.events:
		return ev, ok
	case <-timer.C:
		return event{}, false
	case <-ctx.Done():
		return event{}, false
	}
}

// translateLLM handles the bot's LLM response boundary messages, joining the
// text of one response and emitting it as a single llm_response.
func (s *session) translateLLM(in rtvi.Incoming) *event {
	switch in.Type {
	case rtvi.TypeBotLLMStarted:
		// The genuinely new response begins here, so stragglers from an
		// interrupted prior response are now behind us.
		s.awaitingLLMRestart = false
		s.llmBuf.Reset()
		return &event{kind: EventLLMStarted}
	case rtvi.TypeBotLLMText:
		if s.awaitingLLMRestart {
			return nil
		}
		var d rtvi.TextData
		if json.Unmarshal(in.Data, &d) == nil {
			s.llmBuf.WriteString(d.Text)
		}
		return nil
	default: // rtvi.TypeBotLLMStopped
		// A stopped that arrives before the post-interruption restart is the tail
		// of the interrupted response; drop it instead of emitting it as this
		// turn's llm_response.
		if s.awaitingLLMRestart {
			s.llmBuf.Reset()
			return nil
		}
		text := s.llmBuf.String()
		s.llmBuf.Reset()
		return &event{kind: EventLLMResponse, text: text}
	}
}

// nextMatching pops events until one of kind arrives. Events that do not match
// are dropped, so a scenario does not have to enumerate every event the bot
// emits. It reports false once the deadline passes.
func (s *session) nextMatching(ctx context.Context, kind string, deadline time.Time) (event, bool) {
	for {
		ev, ok := s.next(ctx, deadline)
		if !ok {
			return event{}, false
		}
		if ev.kind == kind {
			return ev, true
		}
	}
}

// sendText builds a send-text message that appends the user's turn and runs the
// LLM immediately, without a spoken audio response.
func sendText(content string) rtvi.Message {
	runNow, noAudio := true, false
	return rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeSendText, ID: messageID,
		Data: rtvi.SendTextData{
			Content: content,
			Options: &rtvi.SendTextOptions{RunImmediately: &runNow, AudioResponse: &noAudio},
		},
	}
}
