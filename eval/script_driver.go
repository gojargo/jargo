package eval

import (
	"context"
	"errors"
	"fmt"
	"time"
)

const (
	// SendAfterMaxWait is how long a turn waits for the event its send_after
	// names before giving up. It is not the expectation budget: nothing is
	// being asserted yet, only waited for.
	SendAfterMaxWait = 30 * time.Second
	// BotQuietMaxWait is how long a turn waits for the bot to finish speaking
	// before it is sent. A reply is still spoken after its first sentence
	// satisfied the previous turn; a turn sent over it would have that tail
	// transcribed as its own reply, or discarded together with the answer that
	// followed it in the same breath.
	BotQuietMaxWait = 30 * time.Second
	// SendAfterPoll is how often the send_after wait checks whether its event
	// has been seen.
	SendAfterPoll = 10 * time.Millisecond
)

// ScriptDriver plays a scenario's turns in order and matches each turn's
// expectations.
//
// A failed turn ends the scenario by default, since the conversation is in an
// unknown state from there on; stop_on_failure: false drives every turn
// regardless.
type ScriptDriver struct {
	baseDriver

	scenario         *ScriptScenario
	defaultTimeoutMS int
	matcher          *ExpectationMatcher
	// Turns is one record per turn, filled in as the driver runs. They start as
	// not run and stay that way on every path that ends the run early, so the
	// result always says which turns were actually scored.
	Turns []ScriptTurnResult
}

// newScriptDriver builds the driver for scenario. defaultTimeoutMS is the
// latency budget for expectations without their own within_ms; progress is told
// as turns and expectations resolve.
func newScriptDriver(
	scenario *ScriptScenario, defaultTimeoutMS int, client turnSender, stream *EventStream,
	judge Judge, trace *Trace, progress func(Progress),
) *ScriptDriver {
	turns := make([]ScriptTurnResult, len(scenario.Turns))
	for i := range turns {
		turns[i] = ScriptTurnResult{TurnIndex: i, Status: TurnNotRun}
	}
	if progress == nil {
		progress = func(Progress) {}
	}
	return &ScriptDriver{
		client:           client,
		stream:           stream,
		judge:            judge,
		trace:            trace,
		progress:         progress,
		scenario:         scenario,
		defaultTimeoutMS: defaultTimeoutMS,
		matcher:          NewExpectationMatcher(stream, judge, trace),
		Turns:            turns,
	}
}

// run drives the scenario's turns in order, filling in their records.
func (d *ScriptDriver) run(ctx context.Context) ([]AssertionFailure, error) {
	var failures []AssertionFailure
	for turnIdx := range d.scenario.Turns {
		turn := &d.scenario.Turns[turnIdx]
		d.trace.SetTurn(turnIdx)
		d.trace.Log(fmt.Sprintf("--- turn %d: %q", turnIdx, turn.User))
		started := time.Now()
		turnFailures, err := d.runTurn(ctx, turn, turnIdx)
		if err != nil {
			return failures, err
		}
		record := &d.Turns[turnIdx]
		record.Status = TurnPassed
		if len(turnFailures) > 0 {
			record.Status = TurnFailed
		}
		record.Failures = turnFailures
		record.DurationMS = int(time.Since(started).Milliseconds())
		failures = append(failures, turnFailures...)
		if len(turnFailures) == 0 {
			continue
		}
		// By default a failed turn ends the scenario: it leaves the
		// conversation in an unknown state, so running the rest just burns
		// another timeout per turn. A scenario whose turns are scored
		// independently sets stop_on_failure: false and drives all of them.
		if d.scenario.stopsOnFailure() {
			d.trace.Log(fmt.Sprintf("turn %d failed; stopping scenario (stop_on_failure)", turnIdx))
			break
		}
		d.trace.Log(fmt.Sprintf("turn %d failed; continuing (stop_on_failure: false)", turnIdx))
	}
	return failures, nil
}

// result is the scenario's result: passed only if nothing failed and nothing
// was skipped.
func (d *ScriptDriver) result(
	failures []AssertionFailure, durationMS int, eventsSeen []map[string]any, debugLog []string, skipped string,
) ScriptResult {
	return ScriptResult{
		ScenarioName: d.scenario.Name,
		Failures:     failures,
		Turns:        d.Turns,
		DurationMS:   durationMS,
		EventsSeen:   eventsSeen,
		DebugLog:     debugLog,
		Skipped:      skipped,
		BotMetrics:   d.botMetrics(),
	}
}

// recordFailure scores a run-level failure against the turn it interrupted, if
// any. Before any turn started the failure's turn is -1 and every turn stays
// not run.
func (d *ScriptDriver) recordFailure(f AssertionFailure) {
	if f.TurnIndex >= 0 && f.TurnIndex < len(d.Turns) {
		record := &d.Turns[f.TurnIndex]
		record.Status = TurnFailed
		record.Failures = append(record.Failures, f)
	}
}

// runTurn drives one turn: it honors send_after, sends the input, and matches
// the expectations.
func (d *ScriptDriver) runTurn(ctx context.Context, turn *ScriptTurn, turnIdx int) ([]AssertionFailure, error) {
	// An expectation may look back to the start of its turn, never further.
	d.matcher.ResetTurn()

	switch {
	case turn.SendAfter != nil:
		if f := d.awaitSendAfter(ctx, *turn.SendAfter, turnIdx); f != nil {
			return []AssertionFailure{*f}, nil
		}
	case turn.User != "" || turn.DTMF != "":
		// A caller waits for the bot to finish; a turn that means to talk over
		// it says so with send_after.
		d.awaitBotQuiet(ctx)
	case turnIdx > 0:
		d.awaitPreviousReply(ctx)
	}

	if err := d.sendTurn(ctx, turn); err != nil {
		return nil, err
	}
	input := turn.User
	if input == "" {
		input = turn.DTMF
	}
	d.progress(Progress{TurnIndex: turnIdx, ExpectationIndex: -1, EventName: input, Status: StatusTurn})
	return d.matchExpectations(ctx, turn, turnIdx)
}

// awaitBotQuiet holds the send while the bot is speaking, and until its speech
// is transcribed, up to BotQuietMaxWait.
//
// What the bot said before the send belongs to the turn before, so the send
// waits for the last of it to be transcribed and closed into a turn: a
// transcript can land seconds after its audio, and one landing after the send
// would otherwise be taken for the reply.
func (d *ScriptDriver) awaitBotQuiet(ctx context.Context) {
	if d.stream.BotSpeaking() {
		d.trace.Log("send: waiting for the bot to finish speaking")
		if !d.stream.WaitBotQuiet(ctx, BotQuietMaxWait) {
			d.trace.Log(fmt.Sprintf("send: bot still speaking after %s, sending", BotQuietMaxWait))
			return
		}
	}
	if d.stream.BotTranscribed() {
		return
	}
	d.trace.Log("send: waiting for the bot's speech to be transcribed")
	if !d.stream.WaitBotTranscribed(ctx, BotQuietMaxWait) {
		d.trace.Log(fmt.Sprintf("send: bot speech still being transcribed after %s (%s), sending",
			BotQuietMaxWait, d.stream.BotSpeechOutstanding()))
	}
}

// awaitPreviousReply lets a reply the bot is still speaking, or that is still
// being transcribed, end before an observing turn starts.
//
// A turn that sends nothing waits for what the bot says next, so what the bot
// is still saying belongs to the turn before it, however late its
// transcription lands. The first turn keeps the bot's greeting.
func (d *ScriptDriver) awaitPreviousReply(ctx context.Context) {
	if !d.stream.BotSpeaking() && d.stream.BotTranscribed() {
		return
	}
	d.trace.Log("observe: waiting for the bot to finish the previous reply")
	d.stream.WaitBotQuiet(ctx, BotQuietMaxWait)
	d.stream.WaitBotTranscribed(ctx, BotQuietMaxWait)
	d.stream.DropPendingBotOutput("the previous reply")
	d.stream.ObservingBoundary()
}

// awaitSendAfter holds the turn's send until its send_after fires; a failure if
// it never does.
func (d *ScriptDriver) awaitSendAfter(ctx context.Context, sendAfter SendAfter, turnIdx int) *AssertionFailure {
	err := d.waitSendAfter(ctx, sendAfter)
	var harnessErr *HarnessError
	switch {
	case err == nil:
		return nil
	case errors.Is(err, ErrBotEnded):
		reason := fmt.Sprintf("the bot ended the call before %q fired", sendAfter.Event)
		return d.sendAfterFailure(sendAfter, turnIdx, reason, FailureBotEnded)
	case errors.As(err, &harnessErr):
		return d.sendAfterFailure(sendAfter, turnIdx, harnessErr.Reason, FailureError)
	default:
		return d.sendAfterFailure(sendAfter, turnIdx, err.Error(), FailureSendAfterTimeout)
	}
}

// errSendAfterNotSeen is a send_after whose event never arrived.
var errSendAfterNotSeen = errors.New("send_after never fired")

// sendAfterFailure records a turn whose send_after never fired, and returns its
// failure.
func (d *ScriptDriver) sendAfterFailure(sendAfter SendAfter, turnIdx int, reason, kind string) *AssertionFailure {
	// Only the event-anchored wait can fail; the pure-delay form just sleeps.
	eventName := sendAfter.Event
	if eventName == "" {
		eventName = "send_after"
	}
	d.trace.Log(fmt.Sprintf("FAIL: %s: %s", eventName, reason))
	status := StatusFailed
	if kind == FailureSendAfterTimeout {
		status = StatusTimeout
	}
	d.progress(Progress{TurnIndex: turnIdx, ExpectationIndex: -1, EventName: eventName, Status: status, Detail: reason})
	return &AssertionFailure{TurnIndex: turnIdx, ExpectationIndex: -1, EventName: eventName, Reason: reason, Kind: kind}
}

// waitSendAfter waits until the send_after's event has been seen and its
// delay_ms more have passed. With no event it is a plain delay from the
// previous turn's send.
//
// It fails with an error naming the event when it never arrives, ErrBotEnded
// when the bot ends the call first, and a *HarnessError when the harness's own
// pipeline fails first.
func (d *ScriptDriver) waitSendAfter(ctx context.Context, sendAfter SendAfter) error {
	delay := time.Duration(sendAfter.DelayMS) * time.Millisecond
	if sendAfter.Event == "" {
		d.trace.Log(fmt.Sprintf("send_after: waiting %dms", sendAfter.DelayMS))
		return sleep(ctx, delay)
	}

	deadline := time.Now().Add(SendAfterMaxWait)
	d.trace.Log(fmt.Sprintf("send_after: waiting for %q + %dms", sendAfter.Event, sendAfter.DelayMS))
	for {
		if seenAt, ok := d.stream.LatestEventTime(sendAfter.Event); ok {
			return sleep(ctx, time.Until(seenAt.Add(delay)))
		}
		if d.stream.BotHasEnded() {
			return ErrBotEnded
		}
		if reason, failed := d.stream.HarnessError(); failed {
			return &HarnessError{Reason: reason}
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("%w: event %q not seen within %dms",
				errSendAfterNotSeen, sendAfter.Event, SendAfterMaxWait.Milliseconds())
		}
		if err := sleep(ctx, SendAfterPoll); err != nil {
			return err
		}
	}
}

// sendTurn sends the turn's input: its image, if any, then its utterance or
// keypresses. A turn that sends nothing only observes, so the bot's pending
// output is kept for it.
func (d *ScriptDriver) sendTurn(ctx context.Context, turn *ScriptTurn) error {
	// Register the turn's image before the user input, so the bot can serve it
	// when it requests a user image during the turn.
	if turn.Image != "" {
		if err := d.client.SendImage(turn.Image); err != nil {
			return err
		}
	}
	switch {
	case turn.User != "":
		return d.say(ctx, turn.User, turn.Audio, turn.Parts)
	case turn.DTMF != "":
		d.press(turn.DTMF)
	}
	return nil
}

// matchExpectations matches the turn's expectations in order, all within one
// deadline anchored at the send.
//
// A stalled turn then fails within a single within_ms budget rather than one
// budget per expectation, and a turn whose bot ended the call fails at once.
func (d *ScriptDriver) matchExpectations(
	ctx context.Context, turn *ScriptTurn, turnIdx int,
) ([]AssertionFailure, error) {
	var failures []AssertionFailure
	anchor := time.Now()
	for expIdx := range turn.Expect {
		exp := &turn.Expect[expIdx]
		budgetMS := exp.WithinMS
		if budgetMS == 0 {
			budgetMS = d.defaultTimeoutMS
		}
		f, err := d.matcher.Match(ctx, exp, anchor, budgetMS, turnIdx, expIdx)
		var harnessErr *HarnessError
		switch {
		case errors.Is(err, ErrTimeout):
			reason := fmt.Sprintf("no matching %q event arrived within %dms", exp.Event, budgetMS)
			return append(failures, d.unmatched(turnIdx, expIdx, exp, reason, FailureTimeout)), nil
		case errors.Is(err, ErrBotEnded):
			reason := fmt.Sprintf("the bot ended the call before a matching %q event", exp.Event)
			return append(failures, d.unmatched(turnIdx, expIdx, exp, reason, FailureBotEnded)), nil
		case errors.As(err, &harnessErr):
			return append(failures, d.unmatched(turnIdx, expIdx, exp, harnessErr.Reason, FailureError)), nil
		case err != nil:
			return failures, err
		}

		resolved := &d.Turns[turnIdx].Expectations
		if f != nil {
			failures = append(failures, *f)
			*resolved = append(*resolved, ExpectationResult{ExpectationIndex: expIdx, EventName: exp.Event})
			d.trace.Log(fmt.Sprintf("FAIL: %s: %s", exp.Event, f.Reason))
			d.progress(Progress{
				TurnIndex: turnIdx, ExpectationIndex: expIdx, EventName: exp.Event, Status: StatusFailed, Detail: f.Reason,
			})
			continue
		}
		matched := d.matcher.LastMatchText
		*resolved = append(*resolved, ExpectationResult{
			ExpectationIndex: expIdx, EventName: exp.Event, Passed: true, Matched: matched,
		})
		d.progress(Progress{
			TurnIndex: turnIdx, ExpectationIndex: expIdx, EventName: exp.Event, Status: StatusMatched, Detail: matched,
		})
	}
	return failures, nil
}

// unmatched records an expectation whose event never arrived, and returns its
// failure.
func (d *ScriptDriver) unmatched(turnIdx, expIdx int, exp *Expectation, reason, kind string) AssertionFailure {
	d.Turns[turnIdx].Expectations = append(d.Turns[turnIdx].Expectations,
		ExpectationResult{ExpectationIndex: expIdx, EventName: exp.Event})
	d.trace.Log(fmt.Sprintf("FAIL: %s: %s", exp.Event, reason))
	status := StatusFailed
	if kind == FailureTimeout {
		status = StatusTimeout
	}
	d.progress(Progress{
		TurnIndex: turnIdx, ExpectationIndex: expIdx, EventName: exp.Event, Status: status, Detail: reason,
	})
	return AssertionFailure{TurnIndex: turnIdx, ExpectationIndex: expIdx, EventName: exp.Event, Reason: reason, Kind: kind}
}
