package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/pytext"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/text"
)

// The simulation driver: lets the persona hold the conversation, then judges the
// whole of it.

const (
	// EventEndCall is the event that reports the persona's end_call, once the
	// reply that made the call has gone out to the bot.
	EventEndCall = "end_call"
	// ClosingReplyWait is how long the run waits, after the persona hangs up on
	// its own last line, for the bot's reply to it. The wait runs only while
	// neither side is speaking, by the bot's reports, and starts over when one
	// stops: the persona's last line plays out after the call is made, and the
	// bot's reply can be long. A flow makes its final tool calls in that turn,
	// and a caller who hangs up mid-sentence is not what the simulation tests.
	ClosingReplyWait = 15 * time.Second
	// BotEndedWait is how long the run waits, after the bot hangs up, for what
	// it said last to close into a turn. The harness hears the bot stop only in
	// the silence that follows, and the turn model may wait out its 3 s silence
	// fallback.
	BotEndedWait = 5 * time.Second
)

// The failure kinds of a measured metric.
const (
	// MetricOutOfRange is a measure outside its bounds.
	MetricOutOfRange = "out_of_range"
	// MetricFunctionCalls is a function_calls measure whose calls did not
	// match the list.
	MetricFunctionCalls = "function_calls"
)

// RunEventJudge is where a judge that gave no verdict on the goal is reported.
const RunEventJudge = "<judge>"

// The keys of the persona's end_call claim, and how an empty list of expected
// calls reads.
const (
	keySuccess = "success"
	keyReason  = "reason"
	noCalls    = "none"
)

// personaClient is what the simulation driver needs of the connection to the
// bot: the Client.
type personaClient interface {
	// HangUp ends the persona's part of the conversation.
	HangUp()
	// ReportEndCall appends the persona's end_call event once the reply that
	// made the call has gone out.
	ReportEndCall(event map[string]any)
}

// specMatches reports whether a call the bot made is the one a calls: entry
// describes: the same name, and the entry's args, when given, all present in the
// call's arguments with the same values.
func specMatches(spec, call FunctionCall) bool {
	if spec.Name != call.Name {
		return false
	}
	for k, v := range spec.Args {
		if !sameValue(call.Args[k], v) {
			return false
		}
	}
	return true
}

// SimulationDriver lets the persona LLM hold the conversation, then judges the
// whole of it.
//
// The persona answers the bot on its own inside the client's pipeline. The
// driver reports each line as progress and watches for the end: the persona's
// end_call, the bot hanging up, the turn cap, the time cap, a lull neither side
// breaks, or a failure of the harness's own pipeline. Then one judge call over
// the whole transcript scores every bot turn on every criterion and decides the
// goal.
type SimulationDriver struct {
	client     personaClient
	stream     *EventStream
	judge      RunJudge
	trace      *Trace
	progress   func(SimulationProgress)
	simulation *SimulationScenario
	persona    *Persona

	// mu guards what the end_call handler sets, from the persona LLM's
	// goroutine.
	mu      sync.Mutex
	endCall map[string]any

	turns   int
	endedBy string
	// endedAt is when the run ended, on the stream's clock, for a reply still
	// owed then; nil until it has.
	endedAt           *float64
	durationS         float64
	succeeded         bool
	reason            string
	goalConfidence    *float64
	goalProbabilities map[string]float64
	metrics           []SimulationMetricScore
	// botSaid is the event that carries the bot's finished response: in audio
	// mode the harness's transcription of what it said, in text mode its LLM
	// text.
	botSaid string
	// conversation is the conversation as the events tell it, built as they
	// arrive, and observed how far into the stream's events the build has
	// read.
	conversation *Conversation
	observed     int
	// closingReplyWait is ClosingReplyWait, or what a test shortens it to.
	closingReplyWait time.Duration
}

// newSimulationDriver builds the driver of simulation. persona is the simulated
// caller, client the connection to the bot, judge the judge for the goal and
// the quality criteria (or nil), and progress is told each line of the
// conversation, and once when it ends.
func newSimulationDriver(
	simulation *SimulationScenario, persona *Persona, client personaClient, stream *EventStream,
	judge RunJudge, trace *Trace, progress func(SimulationProgress),
) *SimulationDriver {
	if progress == nil {
		progress = func(SimulationProgress) {}
	}
	botSaid := EventLLMResponse
	if simulation.BotAudio {
		botSaid = EventResponse
	}
	return &SimulationDriver{
		client:       client,
		stream:       stream,
		judge:        judge,
		trace:        trace,
		progress:     progress,
		simulation:   simulation,
		persona:      persona,
		botSaid:      botSaid,
		conversation: NewConversation(),

		closingReplyWait: ClosingReplyWait,
	}
}

// close closes the judge, once the run has ended.
func (d *SimulationDriver) close(ctx context.Context) {
	if d.judge != nil {
		closeJudge(ctx, d.judge)
	}
}

// recordFailure notes a run-level failure the session raised: a simulation
// keeps no per-turn records, so there is nothing to note.
func (*SimulationDriver) recordFailure(AssertionFailure) {}

// seconds is s seconds as a duration.
func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

// run watches the conversation until it ends, then judges it.
//
//nolint:cyclop,gocognit // one loop over the ways a conversation ends
func (d *SimulationDriver) run(ctx context.Context) ([]AssertionFailure, error) {
	d.persona.OnEndCall(d.onEndCall)
	simulation := d.simulation
	started := time.Now()
	deadline := started.Add(seconds(simulation.MaxDurationS))
	maxSilence := seconds(simulation.MaxSilenceS)
	// A lull is measured from the stream's last activity, which a token or a
	// spoken sentence refreshes without making an event, so a reply in
	// progress on either side is never silence, only nothing happening is.
	d.stream.Touch()
	var failures []AssertionFailure
	d.trace.Log(fmt.Sprintf("persona: listening (up to %d turn(s), %ss, %ss of silence)",
		simulation.MaxTurns, formatG(simulation.MaxDurationS), formatG(simulation.MaxSilenceS)))
	// Set when the persona hangs up on its own last line: the run then waits
	// for the bot's reply to it, or this long, before ending.
	var closingEnds time.Time
	// Whether the bot hears the persona speaking, by its own reports.
	botHearsPersona := false
	for d.endedBy == "" {
		wait := minTime(deadline, d.stream.LastActivity().Add(maxSilence))
		if !closingEnds.IsZero() {
			wait = minTime(wait, closingEnds)
		}
		event, err := d.stream.NextAny(ctx, wait)
		if err != nil {
			if !errors.Is(err, ErrTimeout) {
				return nil, err
			}
			now := time.Now()
			if !closingEnds.IsZero() {
				// The persona already hung up; the bot's reply was a courtesy.
				d.trace.Log("persona: no closing reply from the bot; ending the call")
				d.endedBy = SimulationEndedByEndCall
				break
			}
			if !now.Before(deadline) {
				d.endedBy = SimulationEndedByMaxDuration
				break
			}
			if d.stream.LastActivity().Add(maxSilence).After(now) {
				continue
			}
			d.endedBy = SimulationEndedBySilence
			break
		}
		d.observeNewEvents()
		eventText, _ := event[keyText].(string)
		switch event[keyType] {
		case EventEndCall:
			// The bot still gets its turn if the persona hung up on its own
			// last line.
			if !d.conversation.UserSpokeLast() {
				d.endedBy = SimulationEndedByEndCall
			} else {
				d.trace.Log("persona: hung up; waiting for the bot's closing turn")
				closingEnds = d.closingReplyDeadline(botHearsPersona)
			}
		case EventUserStartedSpeaking, EventUserStoppedSpeaking:
			botHearsPersona = event[keyType] == EventUserStartedSpeaking
			if !closingEnds.IsZero() {
				closingEnds = d.closingReplyDeadline(botHearsPersona)
			}
		case EventBotStartedSpeaking, EventBotStoppedSpeaking:
			if !closingEnds.IsZero() {
				closingEnds = d.closingReplyDeadline(botHearsPersona)
			}
		case EventBotEnded:
			d.endedBy = SimulationEndedByBot
		case EventHarnessError:
			d.endedBy = SimulationEndedByError
			failures = append(failures, d.harnessFailure(eventText))
		case EventPersonaTurn:
			d.turns++
			d.report("user", eventText)
			// A persona that hung up on this turn has its end_call event next,
			// which leaves the bot its closing turn.
			if d.turns >= simulation.MaxTurns && d.endCallClaim() == nil {
				d.endedBy = SimulationEndedByMaxTurns
			}
		case d.botSaid:
			if eventText != "" {
				d.report("bot", eventText)
				if !closingEnds.IsZero() {
					d.endedBy = SimulationEndedByEndCall
				}
			}
		}
	}
	endedAt := d.stream.Elapsed()
	d.endedAt = &endedAt
	// The persona has said its last word either way: nothing the bot says from
	// here on gets an answer.
	d.client.HangUp()
	d.durationS = pytext.Round(time.Since(started).Seconds(), 3)
	d.trace.Log(fmt.Sprintf("persona: ended by %s after %d turn(s)", d.endedBy, d.turns))
	if d.endedBy == SimulationEndedByBot {
		d.reportLastWords(ctx)
	}
	// Whatever the bot said last closes the conversation.
	d.observeNewEvents()
	d.conversation.EndBotTurn()
	d.report("ended", d.endedBy)
	if len(failures) > 0 {
		return failures, nil
	}
	return d.judgeConversation(ctx), nil
}

// minTime is the earlier of a and b.
func minTime(a, b time.Time) time.Time {
	if b.Before(a) {
		return b
	}
	return a
}

// result is the run's result; a run-level failure makes it an error, not a goal
// failure.
func (d *SimulationDriver) result(
	failures []AssertionFailure, durationMS int, eventsSeen []map[string]any, debugLog []string, skipped string,
) *SimulationResult {
	errText := skipped
	if errText == "" && len(failures) > 0 {
		reasons := make([]string, 0, len(failures))
		for _, f := range failures {
			reasons = append(reasons, f.Reason)
		}
		errText = strings.Join(reasons, "; ")
	}
	reason := d.reason
	if errText != "" {
		reason = errText
	}
	endedBy := d.endedBy
	if endedBy == "" {
		endedBy = SimulationEndedByError
	}
	var failureKind string
	if len(failures) > 0 {
		failureKind = failures[0].Kind
	}
	return &SimulationResult{
		SimulationName: d.simulation.Name,
		Succeeded:      d.succeeded && errText == "",
		Reason:         reason,
		Confidence:     d.goalConfidence,
		Probabilities:  d.goalProbabilities,
		Error:          errText,
		Metrics:        d.metrics,
		Messages:       d.Conversation(),
		Conversation:   d.conversation.Messages(),
		Turns:          d.turns,
		EndedBy:        endedBy,
		EndCall:        d.endCallClaim(),
		DurationMS:     durationMS,
		EventsSeen:     eventsSeen,
		DebugLog:       debugLog,
		Success:        d.simulation.Success,
		FailureKind:    failureKind,
		BotMetrics:     botMetricsOf(d.stream),
	}
}

// Timeline is the conversation as the events told it, each line with the tool
// calls made by then (under "evidence").
//
// A bot turn is everything the bot said between two persona turns, however turn
// detection or a function call split it; a turn in which the bot said nothing is
// not a turn.
func (d *SimulationDriver) Timeline() []map[string]any {
	var (
		lines    []map[string]any
		evidence []string
	)
	for _, m := range d.conversation.Messages() {
		if m[keyRole] == ConversationRoleToolCall {
			evidence = append(evidence, CallLines(m)...)
			continue
		}
		lines = append(lines, map[string]any{
			keyRole: m[keyRole], keyContent: m[keyContent], "evidence": slices.Clone(evidence),
		})
	}
	return lines
}

// Transcript is the conversation as the judge sees it: the lines, each tool call
// in place before the line it preceded, as a "tool" line.
//
// Deprecated: No replacement. The judge keeps the conversation it judges.
func (d *SimulationDriver) Transcript() []map[string]any {
	var entries []map[string]any
	for _, m := range d.conversation.Messages() {
		if m[keyRole] == ConversationRoleToolCall {
			for _, line := range CallLines(m) {
				entries = append(entries, map[string]any{keyRole: roleTool, keyContent: line})
			}
			continue
		}
		entries = append(entries, map[string]any{keyRole: m[keyRole], keyContent: m[keyContent]})
	}
	return entries
}

// Conversation is the conversation with the persona as user and the bot as
// assistant, without the tool calls.
func (d *SimulationDriver) Conversation() []map[string]any {
	lines := []map[string]any{}
	for _, m := range d.conversation.Messages() {
		if m[keyRole] != ConversationRoleToolCall {
			lines = append(lines, map[string]any{keyRole: m[keyRole], keyContent: m[keyContent]})
		}
	}
	return lines
}

// ToolCalls are the bot's function calls in order, one line each; a canceled
// call is listed as canceled.
func (d *SimulationDriver) ToolCalls() []string {
	var lines []string
	for _, m := range d.conversation.Messages() {
		if m[keyRole] == ConversationRoleToolCall {
			lines = append(lines, CallLines(m)...)
		}
	}
	return lines
}

// closingReplyDeadline is when the closing reply wait runs out: never while
// either side is speaking.
func (d *SimulationDriver) closingReplyDeadline(botHearsPersona bool) time.Time {
	if botHearsPersona || d.stream.BotSpeaking() {
		return forever
	}
	return time.Now().Add(d.closingReplyWait)
}

// reportLastWords waits for what the bot said before hanging up to close into
// a turn, and reports it.
func (d *SimulationDriver) reportLastWords(ctx context.Context) {
	d.stream.WaitBotTranscribed(ctx, BotEndedWait)
	events := d.stream.EventsSeen()
	for _, event := range events[min(d.observed, len(events)):] {
		if text, _ := event[keyText].(string); event[keyType] == d.botSaid && text != "" {
			d.report("bot", text)
		}
	}
}

// report emits one line of the conversation, or its end, as progress.
func (d *SimulationDriver) report(status, text string) {
	d.progress(SimulationProgress{Status: status, Text: text, Turn: d.turns})
}

// harnessFailure is the failure a harness pipeline error becomes: the run's
// error, scored against the current turn.
func (d *SimulationDriver) harnessFailure(text string) AssertionFailure {
	d.trace.Log("error: harness pipeline: " + text)
	return AssertionFailure{
		TurnIndex: d.turns, ExpectationIndex: -1, EventName: RunEventHarness, Reason: text, Kind: FailureError,
	}
}

// endCallClaim is the persona's end_call claim, or nil before it made one.
func (d *SimulationDriver) endCallClaim() map[string]any {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.endCall
}

// onEndCall is the persona's end_call: it notes the claim and ends the
// conversation.
func (d *SimulationDriver) onEndCall(ctx context.Context, params llm.FunctionCallParams) error {
	var arguments map[string]any
	if len(params.Arguments) > 0 {
		_ = json.Unmarshal(params.Arguments, &arguments)
	}
	reason := ""
	if v, ok := arguments[keyReason]; ok {
		reason = pyStr(v)
	}
	claim := map[string]any{keySuccess: truthy(arguments[keySuccess]), keyReason: reason}
	d.mu.Lock()
	d.endCall = claim
	d.mu.Unlock()
	// The call is the persona's last word: no follow-up response.
	runLLM := false
	if params.Result != nil {
		if err := params.Result(ctx, `{"status":"call ended"}`,
			&frames.FunctionCallResultProperties{RunLLM: &runLLM}); err != nil {
			return err
		}
	}
	// The persona answers nothing more. The run hears of the call once the
	// reply that made it has gone out, so that reply's turn comes first.
	d.client.HangUp()
	d.client.ReportEndCall(map[string]any{
		keyType: EventEndCall, keyText: reason, keySuccess: claim[keySuccess], keyReason: reason,
	})
	return nil
}

// observeNewEvents reads the stream's events not yet read into the
// conversation.
func (d *SimulationDriver) observeNewEvents() {
	events := d.stream.EventsSeen()
	for d.observed < len(events) {
		event := events[d.observed]
		d.observed++
		name, _ := event[keyName].(string)
		if name == "" {
			name = "?"
		}
		args, _ := event[keyArgs].(map[string]any)
		eventText, _ := event[keyText].(string)
		switch event[keyType] {
		case d.botSaid:
			if eventText != "" {
				d.conversation.AddAssistantMessage(eventText)
			}
		case EventPersonaTurn:
			d.conversation.AddUserMessage(eventText)
		case EventFunctionCall:
			d.conversation.AddToolCall(name, args)
		case EventFunctionCallStopped:
			if truthy(args[keyCanceled]) {
				d.conversation.CancelToolCall(name)
			}
		}
	}
}

// judgeConversation scores the measured metrics, then asks the judge once about
// the goal and every judged criterion.
//
// A judge that answers nothing usable about the goal is the run's error, not a
// verdict on the bot.
func (d *SimulationDriver) judgeConversation(ctx context.Context) []AssertionFailure {
	measured := map[string]SimulationMetricScore{}
	for _, m := range d.simulation.Metrics {
		if m.Measure != "" {
			measured[m.Name] = d.measure(m)
		}
	}
	if d.judge == nil {
		for _, m := range d.simulation.Metrics {
			if score, ok := measured[m.Name]; ok {
				d.metrics = append(d.metrics, score)
			}
		}
		d.reason = "no judge configured"
		return nil
	}
	// A judge passed in is never built, so judging cannot fail.
	judgement, _ := NewSimulationJudge(d.simulation, d.judge, nil, d.trace).Judge(ctx, d.conversation.Messages())
	judged := map[string]SimulationMetricScore{}
	for _, m := range judgement.Metrics {
		judged[m.Name] = m
	}
	for _, m := range d.simulation.Metrics {
		if score, ok := measured[m.Name]; ok {
			d.metrics = append(d.metrics, score)
		} else {
			d.metrics = append(d.metrics, judged[m.Name])
		}
	}
	goal := judgement.Goal
	d.succeeded = goal.Verdict == VerdictYes
	d.reason = goal.Reason
	d.goalConfidence = goal.Confidence
	d.goalProbabilities = goal.Probabilities
	if goal.Verdict == VerdictNone {
		d.trace.Log("judge: no verdict on the goal: " + d.reason)
		return []AssertionFailure{{
			TurnIndex: d.turns, ExpectationIndex: -1, EventName: RunEventJudge,
			Reason: d.reason, Kind: FailureJudgeNoVerdict,
		}}
	}
	outcome := "not achieved"
	if d.succeeded {
		outcome = "achieved"
	}
	d.trace.Log(fmt.Sprintf("judge: goal %s: %s", outcome, d.reason))
	return nil
}

// measure is a measured metric's outcome: its value against its range, or the
// bot's calls against the list.
func (d *SimulationDriver) measure(metric SimulationMetric) SimulationMetricScore {
	var (
		score, value        *float64
		passed              bool
		reason, failureKind string
	)
	if metric.Measure == MeasureFunctionCalls {
		score, value, passed, reason, failureKind = d.callsScore(metric)
	} else {
		score, value, passed, reason, failureKind = d.rangeScore(metric)
	}
	scored := unscored
	if score != nil {
		scored = fmt.Sprintf("%.2f", *score)
	}
	failed := ""
	if !passed {
		failed = " (failed)"
	}
	d.trace.Log(fmt.Sprintf("measure: %s = %s%s: %s", metric.Name, scored, failed, reason))
	return SimulationMetricScore{
		Name: metric.Name, Score: score, Passed: passed, Reason: reason, Value: value, FailureKind: failureKind,
	}
}

// callsScore is a function_calls measure's outcome: the bot's calls against
// the list.
func (d *SimulationDriver) callsScore(
	metric SimulationMetric,
) (score, value *float64, passed bool, reason, failureKind string) {
	count, passed, reason := d.callsOutcome(metric)
	s := 0.0
	if passed {
		s = 1
	} else {
		failureKind = MetricFunctionCalls
	}
	return &s, &count, passed, reason, failureKind
}

// rangeScore is a measure's outcome: its value against its range.
func (d *SimulationDriver) rangeScore(
	metric SimulationMetric,
) (score, value *float64, passed bool, reason, failureKind string) {
	value, described := d.measurement(metric.Measure)
	low, high := metric.MinValue, metric.MaxValue
	var bound string
	switch {
	case low != nil && high != nil:
		bound = fmt.Sprintf("between %s and %s", formatG(*low), formatG(*high))
	case high != nil:
		bound = "at most " + formatG(*high)
	default:
		bound = "at least " + formatG(derefOr(low, 0))
	}
	reason = described + ", " + bound
	if value == nil {
		return nil, nil, false, reason, FailureError
	}
	inside := (low == nil || *value >= *low) && (high == nil || *value <= *high)
	s := 0.0
	if inside {
		s = 1
	} else {
		failureKind = MetricOutOfRange
	}
	return &s, value, inside, reason, failureKind
}

// derefOr is *p, or def when p is nil.
func derefOr(p *float64, def float64) float64 {
	if p == nil {
		return def
	}
	return *p
}

// measurement is a measure's value for this run, or nil when there is nothing
// to measure, and the phrase that reports it.
func (d *SimulationDriver) measurement(measure string) (*float64, string) {
	switch measure {
	case MeasureTurns:
		v := float64(d.turns)
		return &v, fmt.Sprintf("%d persona turn(s)", d.turns)
	case MeasureDuration:
		v := d.durationS
		return &v, fmt.Sprintf("%.1f s", v)
	case MeasureWords:
		longest := -1
		for _, line := range d.Conversation() {
			if line[keyRole] == ConversationRoleAssistant {
				content, _ := line[keyContent].(string)
				longest = max(longest, text.CountSpokenWords(content))
			}
		}
		if longest < 0 {
			return nil, "no reply to measure"
		}
		v := float64(longest)
		return &v, fmt.Sprintf("longest reply %d words", longest)
	case MeasureLatency:
		latencies, unanswered := d.replyLatencies()
		if unanswered != nil {
			latencies = append(latencies, *unanswered)
		}
		if len(latencies) == 0 {
			return nil, "no reply to time"
		}
		slowest := slices.Max(latencies)
		start := "the persona's send"
		if d.simulation.UserAudio {
			start = "the persona stopping"
		}
		end := "the first token"
		if d.simulation.BotAudio {
			end = "the bot starting to speak"
		}
		reason := fmt.Sprintf("slowest reply %.2f s from %s to %s", slowest, start, end)
		if unanswered != nil && slowest == *unanswered {
			reason += ", the last turn unanswered when the run ended by " + d.endedBy
		}
		return &slowest, reason
	}
	return nil, fmt.Sprintf("unknown measure %s", pyRepr(measure))
}

// replyLatencies are the seconds from each persona turn to the start of the
// bot's reply, and the wait of one never answered, or nil.
//
// Each end follows its own side's modality. The start is the persona's send for
// text input, and the bot noticing the persona stop for spoken input, which the
// bot reports only for audio and only once its turn strategy has decided the
// persona is done, so the bot's end-of-turn wait is left out. The end is the
// first LLM token of a reply that has text, for a text judge, and the bot
// starting to speak, for an audio judge. An LLM response that only calls a
// function has no text, so it is not the reply. The bot's report of its first
// spoken word isn't used as the end: it lags the audio by about 0.1 s with a TTS
// that reports word timestamps, and by the whole first sentence with an HTTP TTS
// that doesn't, which sends a sentence's text after its audio.
//
// A turn still unanswered when the run ends by silence or its time limit was
// owed a reply, so the wait up to the end is returned as a lower bound on its
// latency. Under the time limit that bound can be short. After an end_call, the
// turn limit, or the bot leaving, no reply was owed, and nothing is returned for
// it.
func (d *SimulationDriver) replyLatencies() ([]float64, *float64) {
	sent := EventPersonaTurn
	if d.simulation.UserAudio {
		sent = EventUserStoppedSpeaking
	}
	replied, needsText := EventLLMResponse, true
	if d.simulation.BotAudio {
		replied, needsText = EventBotStartedSpeaking, false
	}
	var (
		latencies []float64
		sentAt    *float64
	)
	for _, event := range d.stream.EventsSeen() {
		eventText, _ := event[keyText].(string)
		switch {
		case event[keyType] == sent:
			sentAt = floatOf(event["at"])
		case event[keyType] == replied && sentAt != nil && (eventText != "" || !needsText):
			firstWordValue, ok := event["started_at"]
			if !ok {
				firstWordValue = event["at"]
			}
			if firstWord := floatOf(firstWordValue); firstWord != nil {
				latencies = append(latencies, pytext.Round(math.Max(0, *firstWord-*sentAt), 3))
			}
			sentAt = nil
		}
	}
	var unanswered *float64
	if sentAt != nil && d.endedAt != nil &&
		(d.endedBy == SimulationEndedBySilence || d.endedBy == SimulationEndedByMaxDuration) {
		v := pytext.Round(math.Max(0, *d.endedAt-*sentAt), 3)
		unanswered = &v
	}
	return latencies, unanswered
}

// floatOf is a numeric event field as a float, or nil when it has none.
func floatOf(v any) *float64 {
	if f, ok := toFloat(v); ok {
		return &f
	}
	return nil
}

// callsOutcome is the function_calls measure: the count, whether the set
// matches, and a reason naming the calls made against the list.
func (d *SimulationDriver) callsOutcome(metric SimulationMetric) (float64, bool, string) {
	made := d.callsMade()
	expected := metric.Calls
	missing := slices.ContainsFunc(expected, func(spec FunctionCall) bool {
		return !slices.ContainsFunc(made, func(c FunctionCall) bool { return specMatches(spec, c) })
	})
	unlisted := slices.ContainsFunc(made, func(c FunctionCall) bool {
		return !slices.ContainsFunc(expected, func(spec FunctionCall) bool { return specMatches(spec, c) })
	})
	described := signatures(made)
	if described == "" {
		described = "no calls"
	}
	wanted := signatures(expected)
	if wanted == "" {
		wanted = noCalls
	}
	return float64(len(made)), !missing && !unlisted, described + ", expected " + wanted
}

// signatures are the calls' signatures, comma-separated.
func signatures(calls []FunctionCall) string {
	parts := make([]string, 0, len(calls))
	for _, c := range calls {
		parts = append(parts, c.Signature())
	}
	return strings.Join(parts, ", ")
}

// callsMade are the bot's function calls in order, less the ones it canceled.
func (d *SimulationDriver) callsMade() []FunctionCall {
	var calls []FunctionCall
	for _, m := range d.conversation.Messages() {
		if m[keyRole] != ConversationRoleToolCall || truthy(m[keyCanceled]) {
			continue
		}
		name, _ := m[keyName].(string)
		args, _ := m[keyArguments].(map[string]any)
		calls = append(calls, FunctionCall{Name: name, Args: args})
	}
	return calls
}
