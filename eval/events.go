package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// The bot's output as the harness sees it.
//
// EventStream turns the frames the bot-facing transport produces into the
// events scenarios assert on, queues them for the matcher, and keeps the
// bookkeeping the rest of the harness reads: every event seen, when each event
// type last arrived (for send_after), and whether the bot's next output is still
// the tail of an interrupted response.
//
// The events, and the RTVI server messages the bot emits them as:
//
//	user_started_speaking      user-started-speaking
//	user_stopped_speaking      user-stopped-speaking
//	vad_user_started_speaking  vad-user-started-speaking (raw VAD, ungated by turn detection)
//	vad_user_stopped_speaking  vad-user-stopped-speaking (raw VAD, ungated by turn detection)
//	user_transcription         user-input of type transcription (final only)
//	llm_started                bot-llm-started
//	llm_response               the LLM text: bot-llm-text joined at bot-llm-stopped
//	llm_marker                 bot-llm-marker: the sideband marker the LLM emitted in a
//	                           response, such as a turn-completion marker or the
//	                           silent mark of a reply dropped unspoken, with its
//	                           kind and the response's raw text; sent when the
//	                           response ends, only on request
//	tts_response               the TTS's spoken text, one segment per bot-tts-text
//	                           (audio modality only)
//	response                   the harness's own transcription of the bot's audio
//	                           (audio modality only); llm_response in text modality
//	function_call              llm-function-call-in-progress, which the bot also sends
//	                           for a call made outside its pipeline and reported
//	                           into it, such as a backend's
//	function_call_stopped      llm-function-call-stopped; its args carry tool_call_id
//	                           and whether it was canceled, so a scenario can tell
//	                           work that was stopped from work that finished on its own
//	image                      eval-bot-image: an image the bot output, with its
//	                           width, height and format; sent only on request
//	bot_started_speaking       bot-started-speaking
//	bot_stopped_speaking       bot-stopped-speaking
//	bot_backchannel            bot-output of type backchannel, once spoken to the end:
//	                           a short acknowledgment the bot said while the user talked
//	user_backchannel           user-input of type backchannel: a short acknowledgment
//	                           the user said while the bot talked, which neither
//	                           interrupted it nor reached the LLM
//	metrics                    metrics: what the bot's processors measured (TTFB,
//	                           processing time, usage), as sent, under "data";
//	                           kept in Metrics and out of the events seen
//	server_message             server-message: a message of the bot's own making,
//	                           with its "data" and, as "text", the data as JSON
//	flow_node                  flow-node-entered: the bot's conversation flow
//	                           entered a node, with the message's "data" and, as
//	                           "text", the node's name
//	expression                 bot-output in expressive conversation: the expression
//	                           the bot's TTS performed, in the tags the LLM writes
//
// An event is a map with its type under "type" and its payload beside it, the
// way a results record lists it.

// The event types the stream adds beyond the ones a scenario names in script.go.
const (
	// EventBotStartedSpeaking fires when the bot reports it started speaking.
	EventBotStartedSpeaking = "bot_started_speaking"
	// EventBotStoppedSpeaking fires when the bot reports it stopped speaking.
	EventBotStoppedSpeaking = "bot_stopped_speaking"
	// EventBotBackchannel is a short acknowledgment the bot spoke to the end
	// while the user talked, such as "Mm-hmm".
	EventBotBackchannel = "bot_backchannel"
	// EventUserBackchannel is a short acknowledgment the user said while the
	// bot talked, which neither interrupted it nor reached its LLM.
	EventUserBackchannel = "user_backchannel"
	// EventBotEnded is appended when the bot ends the call by closing the
	// connection.
	EventBotEnded = "bot_ended"
	// EventHarnessError is appended when the harness's own pipeline reported
	// an error: a service in it failed, not the bot.
	EventHarnessError = "harness_error"
)

// interruptionEvents are the bot's reports that end its current turn: what it
// produced so far is not its reply to what comes next.
//
//nolint:gochecknoglobals // fixed table, read only
var interruptionEvents = []string{EventUserStartedSpeaking, EventBotInterrupted}

// The ways a wait on the stream ends without its event.
var (
	// ErrTimeout is a wait whose deadline passed before its event arrived.
	ErrTimeout = errors.New("eval: timed out")
	// ErrBotEnded is a wait for an event that will never arrive: the bot ended
	// the call first.
	ErrBotEnded = errors.New("eval: the bot ended the call")
)

// HarnessError is a wait abandoned because a service in the harness's own
// pipeline failed, so what the bot does next is no evidence.
type HarnessError struct {
	// Reason is the pipeline's error.
	Reason string
}

func (e *HarnessError) Error() string { return e.Reason }

// FromNow is the since of a NextEvent that looks only at what arrives from now
// on, not back into the turn's log.
const FromNow = -1

// forever stands for a time the bot has not reached yet: talking over it lasts
// until it starts speaking again.
//
//nolint:gochecknoglobals // a fixed instant, read only
var forever = time.Unix(math.MaxInt32, 0)

// EventStream is the bot's output as events: a queue the drivers read, fed by
// the pipeline.
//
// The sink hands it the bot's RTVI messages, each mapped to its event (see
// RTVIMessageToEvent), and the harness appends events of its own; the drivers
// pop events as they wait for them. One rule runs through it: what the
// bot produced before the user's latest send is not the reply to it. An LLM
// response still streaming at the send, or cut by an interruption, is dropped
// until the bot's next llm-started, and a spoken turn that began before the send
// is dropped when it ends.
//
// It is safe for concurrent use: the pipeline feeds it and the driver reads it.
type EventStream struct {
	botAudio bool
	trace    *Trace

	mu sync.Mutex
	// changed is closed and replaced whenever the stream's state changes, so a
	// wait can sleep until there is something new to look at.
	changed chan struct{}
	queue   []map[string]any
	// The events popped during the current turn, in order, and which of them an
	// expectation has claimed. An expectation looks for its event from the
	// point the turn had reached when it began, or from the start of the turn
	// (check_from: turn_start), so an event popped on the way to another
	// expectation's is not lost to one that may look back for it.
	turnLog []map[string]any
	claimed map[int]bool
	// Every event in arrival order, for the result's diagnostics.
	eventsSeen []map[string]any
	// Every metrics event the bot sent, in arrival order. Kept apart from
	// eventsSeen: a bot reports several per turn, and they are measurements to
	// read rather than the behavior a failed run is diagnosed from.
	metrics []map[string]any
	// When each event type last arrived, for send_after anchoring.
	latestEventTimes map[string]time.Time
	// The bot's LLM text for the current response, emitted as one llm_response
	// segment when the response ends.
	textBuffer []string
	// Events are stamped in seconds since t0 ("at"), and a reply with the
	// arrival of its first token ("started_at"), for the timing measures.
	t0 time.Time
	// When either side last did anything: an event, or a frame that only
	// buffers toward one.
	lastActivity time.Time
	llmTextAt    *float64
	// The delivery the bot last reported speaking in during the current
	// response, so an expression event marks only where the delivery changes,
	// and the event and pause segments already reported.
	expressionDelivery []string
	expressionSegments map[any]bool
	// The reply rule's flag: set by a send or an interruption, cleared at the
	// bot's next llm-started. While set, LLM text is dropped: an interrupted
	// response can still flush a trailing token after the interruption, and
	// that straggler must not be attributed to the new turn.
	awaitingReply bool
	// When the driver last sent user input, and, when that send talked over the
	// bot, when the bot started speaking again (forever until it does): its
	// speech from before then is what it was saying, not its reply, however
	// late its transcription lands (audio mode).
	inputSentAt     time.Time
	talkedOverUntil *time.Time
	// Whether the turn sends nothing and waits for the bot's next speech, which
	// is its reply whether or not the bot's LLM produced it. Cleared when the
	// bot starts speaking.
	observing bool
	// Whether the bot is quiet, by its own report.
	botQuiet bool
	// Whether everything the bot has said is transcribed and closed into a
	// turn (audio mode): a segment of its speech the VAD still hears, segments
	// still to be transcribed, and whether a spoken turn is open.
	botSegmentOpen     bool
	botSegmentsPending int
	botTurnOpen        bool
	botTranscribed     bool
	// Whether the bot closed the connection. Kept apart from its queued
	// bot_ended event, which a drop of the bot's pending output discards.
	botEnded bool
	// The harness's first pipeline error, kept apart from its queued
	// harness_error event for the same reason.
	harnessError *string
}

// NewEventStream builds the stream of a run. botAudio says whether the scenario
// runs in audio mode, where tts_response is emitted; trace is where each event
// is logged as it arrives.
func NewEventStream(botAudio bool, trace *Trace) *EventStream {
	now := time.Now()
	return &EventStream{
		botAudio:           botAudio,
		trace:              trace,
		changed:            make(chan struct{}),
		claimed:            map[int]bool{},
		latestEventTimes:   map[string]time.Time{},
		t0:                 now,
		lastActivity:       now,
		expressionSegments: map[any]bool{},
		botQuiet:           true,
		botTranscribed:     true,
	}
}

// signalLocked wakes whatever waits on the stream. The caller holds mu.
func (s *EventStream) signalLocked() {
	close(s.changed)
	s.changed = make(chan struct{})
}

// waitFor waits up to timeout for cond, read under the lock, to hold, and
// reports whether it did.
func (s *EventStream) waitFor(ctx context.Context, timeout time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for {
		s.mu.Lock()
		ok := cond()
		ch := s.changed
		s.mu.Unlock()
		if ok {
			return true
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return false
		}
		timer := time.NewTimer(remaining)
		select {
		case <-ch:
			timer.Stop()
		case <-timer.C:
			// Read once more: the deadline check then ends the wait.
		case <-ctx.Done():
			timer.Stop()
			return false
		}
	}
}

// BotSpeaking reports whether the bot reports itself speaking right now.
func (s *EventStream) BotSpeaking() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.botQuiet
}

// TalkedOverUntil is when the bot started speaking again after the latest turn
// began. The bot's speech that began before then belongs to the turn before:
// what it was saying when a send talked over it, or the reply an observing turn
// waited out. It is far in the future until the bot starts speaking again, and
// ok is false when the latest send did not talk over it.
func (s *EventStream) TalkedOverUntil() (until time.Time, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.talkedOverUntil == nil {
		return time.Time{}, false
	}
	return *s.talkedOverUntil, true
}

// BotTranscribed reports whether everything the bot has said is transcribed and
// closed into a turn.
func (s *EventStream) BotTranscribed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.botTranscribed
}

// BotHasEnded reports whether the bot ended the call (see BotEnded).
func (s *EventStream) BotHasEnded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.botEnded
}

// HarnessError is the harness's first pipeline error, and whether there was
// one (see HarnessFailed).
func (s *EventStream) HarnessError() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.harnessError == nil {
		return "", false
	}
	return *s.harnessError, true
}

// BotSpeechOutstanding is what of the bot's speech is not yet accounted for,
// for the trace.
func (s *EventStream) BotSpeechOutstanding() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var parts []string
	if s.botSegmentOpen {
		parts = append(parts, "a segment still being heard")
	}
	if s.botSegmentsPending > 0 {
		parts = append(parts, fmt.Sprintf("%d segment(s) awaiting transcription", s.botSegmentsPending))
	}
	if s.botTurnOpen {
		parts = append(parts, "a turn still open")
	}
	if len(parts) == 0 {
		return "nothing"
	}
	return strings.Join(parts, ", ")
}

// WaitBotTranscribed waits up to timeout for everything the bot has said to be
// transcribed and closed into a turn, and reports whether it was.
func (s *EventStream) WaitBotTranscribed(ctx context.Context, timeout time.Duration) bool {
	return s.waitFor(ctx, timeout, func() bool { return s.botTranscribed })
}

// BotSegmentStarted notes that the VAD hears a segment of the bot's speech.
func (s *EventStream) BotSegmentStarted() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botSegmentOpen = true
	s.botTranscribed = false
	s.signalLocked()
}

// BotSegmentEnded notes that a segment of the bot's speech ended and awaits
// transcription.
func (s *EventStream) BotSegmentEnded() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botSegmentOpen = false
	s.botSegmentsPending++
	s.botTranscribed = false
	s.signalLocked()
}

// BotSegmentTranscribed notes that a segment of the bot's speech was
// transcribed, whether or not it yielded text.
func (s *EventStream) BotSegmentTranscribed() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botSegmentsPending = max(0, s.botSegmentsPending-1)
	s.updateBotTranscribedLocked()
}

// BotTurnStarted notes that the bot began a spoken turn.
func (s *EventStream) BotTurnStarted() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botTurnOpen = true
	s.botTranscribed = false
	s.signalLocked()
}

// BotEnded notes that the bot closed the connection: it sends nothing more, and
// is no longer speaking.
func (s *EventStream) BotEnded() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botEnded = true
	s.botQuiet = true
	s.signalLocked()
}

// HarnessFailed notes that a service in the harness's own pipeline failed; the
// first error is kept.
func (s *EventStream) HarnessFailed(err string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.harnessError == nil {
		s.harnessError = &err
	}
	s.signalLocked()
}

func (s *EventStream) updateBotTranscribedLocked() {
	if !s.botSegmentOpen && s.botSegmentsPending == 0 && !s.botTurnOpen {
		s.botTranscribed = true
	}
	s.signalLocked()
}

// WaitBotQuiet waits up to timeout for the bot to stop speaking, and reports
// whether it is quiet.
func (s *EventStream) WaitBotQuiet(ctx context.Context, timeout time.Duration) bool {
	return s.waitFor(ctx, timeout, func() bool { return s.botQuiet })
}

// Append queues an event for the drivers. It stamps its arrival time as "at"
// (seconds since the stream began) unless it has one, records it, and logs it.
// A metrics event is recorded in Metrics rather than EventsSeen.
func (s *EventStream) Append(event map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.appendLocked(event)
}

func (s *EventStream) appendLocked(event map[string]any) {
	if _, ok := event["at"]; !ok {
		event["at"] = s.elapsedLocked()
	}
	s.lastActivity = time.Now()
	eventType, _ := event[keyType].(string)
	var preview string
	if eventType == EventMetrics {
		s.metrics = append(s.metrics, event)
		kinds, _ := event[keyData].(map[string]any)
		preview = strings.Join(slices.Sorted(maps.Keys(kinds)), ", ")
	} else {
		s.eventsSeen = append(s.eventsSeen, event)
		preview = eventPreview(event)
	}
	s.latestEventTimes[eventType] = time.Now()
	line := "event: " + eventType
	if preview != "" {
		line += fmt.Sprintf("  %q", preview)
	}
	s.trace.Log(line)
	s.queue = append(s.queue, event)
	s.signalLocked()
}

// eventPreview is what of an event the trace shows beside its type.
func eventPreview(event map[string]any) string {
	for _, key := range []string{keyText, keyTranscript, keyName} {
		if v, ok := event[key].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// NextAny pops the next queued event, whatever its type, waiting for one until
// deadline. It returns ErrTimeout when none arrives in time.
func (s *EventStream) NextAny(ctx context.Context, deadline time.Time) (map[string]any, error) {
	if time.Until(deadline) <= 0 {
		return nil, ErrTimeout
	}
	for {
		s.mu.Lock()
		if len(s.queue) > 0 {
			event := s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			return event, nil
		}
		ch := s.changed
		s.mu.Unlock()
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return nil, ErrTimeout
		}
		timer := time.NewTimer(remaining)
		select {
		case <-ch:
			timer.Stop()
		case <-timer.C:
			return nil, ErrTimeout
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		}
	}
}

// TurnPosition is how many events the turn has popped so far: where an
// expectation beginning now looks from.
func (s *EventStream) TurnPosition() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.turnLog)
}

// BeginTurn forgets the events popped during the previous turn.
func (s *EventStream) BeginTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnLog = nil
	s.claimed = map[int]bool{}
}

// UnclaimedSince is the events popped from position since on that no
// expectation has claimed.
func (s *EventStream) UnclaimedSince(since int) []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []map[string]any
	for i, event := range s.turnLog {
		if i >= since && !s.claimed[i] {
			out = append(out, event)
		}
	}
	return out
}

// NextEvent is the next unclaimed event of eventType that accept takes, looking
// from since.
//
// The turn's log is searched first, from since on (FromNow looks only at what
// arrives from now on), then the queue is popped until such an event arrives.
// Events popped on the way stay in the log, unclaimed: a scenario need not list
// every event the bot emits, and an expectation that looks back from an earlier
// point can still claim them. accept, when not nil, is a further check on an
// event of the type, such as a call's name; one it turns down is left for
// another expectation.
//
// It returns ErrTimeout when none arrives before deadline, ErrBotEnded when the
// bot ended the call and none is left to arrive (the harness's transcription of
// what the bot said last, in audio mode, is waited for first, since it can land
// after the bot is gone), and a *HarnessError when the harness's own pipeline
// failed.
func (s *EventStream) NextEvent(
	ctx context.Context, eventType string, deadline time.Time, since int, accept func(map[string]any) bool,
) (map[string]any, error) {
	wanted := func(event map[string]any) bool {
		return event[keyType] == eventType && (accept == nil || accept(event))
	}
	if since >= 0 {
		s.mu.Lock()
		for i := since; i < len(s.turnLog); i++ {
			if !s.claimed[i] && wanted(s.turnLog[i]) {
				s.claimed[i] = true
				event := s.turnLog[i]
				s.mu.Unlock()
				s.trace.Log(fmt.Sprintf("match: %q from earlier in the turn", eventType))
				return event, nil
			}
		}
		s.mu.Unlock()
	}
	for {
		s.mu.Lock()
		harnessError, ended, empty := s.harnessError, s.botEnded, len(s.queue) == 0
		s.mu.Unlock()
		if harnessError != nil {
			return nil, &HarnessError{Reason: *harnessError}
		}
		if ended && empty {
			if !s.WaitBotTranscribed(ctx, time.Until(deadline)) {
				return nil, ErrTimeout
			}
			// A transcription that yields text queues its event as it closes.
			s.mu.Lock()
			empty = len(s.queue) == 0
			s.mu.Unlock()
			if empty {
				return nil, ErrBotEnded
			}
		}
		event, err := s.NextAny(ctx, deadline)
		if err != nil {
			return nil, err
		}
		s.mu.Lock()
		s.turnLog = append(s.turnLog, event)
		if wanted(event) {
			s.claimed[len(s.turnLog)-1] = true
			s.mu.Unlock()
			return event, nil
		}
		s.mu.Unlock()
	}
}

// DropPendingBotOutput drops the bot's queued output, so a later turn cannot
// match it.
//
// A queued user_transcription stays: a DTMF keypress reports its transcription
// just before the turn starts, and that is the turn's input, not stale bot
// output. The events seen and the latest event times are left as they are. why
// says what prompted the drop, for the trace.
func (s *EventStream) DropPendingBotOutput(why string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dropPendingBotOutputLocked(why)
}

func (s *EventStream) dropPendingBotOutputLocked(why string) {
	s.textBuffer = nil
	var preserved []map[string]any
	dropped := 0
	for _, event := range s.queue {
		if event[keyType] == EventUserTranscription {
			preserved = append(preserved, event)
		} else {
			dropped++
		}
	}
	s.queue = preserved
	if dropped > 0 {
		s.trace.Log(fmt.Sprintf("discard: dropped %d queued event(s) %s", dropped, why))
	}
}

// Elapsed is the seconds since the stream began, the clock the events' "at" is
// on, to the millisecond.
func (s *EventStream) Elapsed() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.elapsedLocked()
}

func (s *EventStream) elapsedLocked() float64 {
	return math.Round(time.Since(s.t0).Seconds()*1000) / 1000
}

// Touch notes activity that makes no event, so a lull is measured from it.
func (s *EventStream) Touch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastActivity = time.Now()
}

// LastActivity is when either side last did anything: an event, or a frame
// that only buffers toward one.
func (s *EventStream) LastActivity() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastActivity
}

// EventsSeen is every event in arrival order, for the result's diagnostics.
func (s *EventStream) EventsSeen() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.eventsSeen)
}

// Metrics is every metrics event the bot sent, in arrival order.
func (s *EventStream) Metrics() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.metrics)
}

// LatestEventTime is when an event of eventType last arrived, and whether one
// has at all.
func (s *EventStream) LatestEventTime(eventType string) (time.Time, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	at, ok := s.latestEventTimes[eventType]
	return at, ok
}

// InputSentAt is when the user's latest input was sent; the bot's speech from
// before it is not the reply.
func (s *EventStream) InputSentAt() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inputSentAt
}

// RTVIMessageToEvent translates one of the bot's RTVI messages, as the client
// received it, into the event it maps to, or nil; many map to none.
//
// The LLM text is buffered and emitted as one llm_response when the response
// ends; a marker is its own llm_marker as soon as it arrives; tts_response is
// emitted only in audio mode; the bot's reports, about its own speaking and
// interruptions and about the harness (its VAD and transcription messages),
// become their events. The audio-mode response is not made here: the turn
// aggregator appends it when the bot's spoken turn ends.
func (s *EventStream) RTVIMessageToEvent(message []byte) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastActivity = time.Now()
	event := s.eventForRTVIMessageLocked(message)
	if event == nil {
		return nil
	}
	switch eventType, _ := event[keyType].(string); {
	case slices.Contains(interruptionEvents, eventType):
		s.interruptedLocked()
	case eventType == EventBotStartedSpeaking:
		s.botStartedSpeakingLocked()
	case eventType == EventBotStoppedSpeaking:
		s.botQuiet = true
		s.signalLocked()
	}
	return event
}

// The keys of an event's payload.
const (
	keyText       = "text"
	keyTranscript = "transcript"
	keyArgs       = "args"
	// keyData is what a metrics or server_message event carries, as the bot
	// sent it: the data of its message.
	keyData = "data"
	// keyCanceled says, in a call event's args, whether the call was canceled,
	// as the protocol spells it.
	keyCanceled = "cancelled" //nolint:misspell // the protocol's spelling
)

// optionalName is a call's name as the bot reported it, or nil when it
// reported none: the name may be absent at the bot's default report level.
func optionalName(data map[string]any) any {
	if name := data["function_name"]; truthy(name) {
		return name
	}
	return nil
}

// BotTurnStopped appends the bot's finished spoken turn as a response, unless it
// is stale.
//
// Speech the bot was saying when a send talked over it never reaches a turn
// (the client drops it by its audio's time). A turn is still stale while the
// bot's LLM has not restarted since the send: the bot going on with what it was
// saying, which must not pass for the reply. Nothing is appended for an empty
// text.
func (s *EventStream) BotTurnStopped(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.botTurnOpen = false
	s.signalLocked()
	s.updateBotTranscribedLocked()
	if text == "" {
		return
	}
	if s.awaitingReply {
		s.trace.Log(fmt.Sprintf("discard: bot turn from before the reply %q", text))
		return
	}
	s.appendLocked(map[string]any{keyType: EventResponse, keyText: text})
}

// WaitBotTurnClosed waits up to timeout for no turn of the bot's to be open, and
// reports whether none was.
//
// The bot-turn aggregator reports a turn's start and stop on a goroutine of its
// own, after it has handed the turn on, so whatever answers the turn can be
// ready before the stop is in the stream. A caller about to record an answer
// waits here first, so the bot's turn always comes before the answer to it.
func (s *EventStream) WaitBotTurnClosed(ctx context.Context, timeout time.Duration) bool {
	return s.waitFor(ctx, timeout, func() bool { return !s.botTurnOpen })
}

// InputSent marks the user's input as sent: the bot's reply is what it says
// from now on.
func (s *EventStream) InputSent() { s.TurnBoundary() }

// TurnBoundary starts a new turn: only what the bot says from now on belongs to
// it.
//
// When the bot is speaking at this point, the turn talks over it, and what the
// bot says until it starts speaking again, however late its transcription
// lands, is the previous turn's and is discarded.
func (s *EventStream) TurnBoundary() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnBoundaryLocked()
}

func (s *EventStream) turnBoundaryLocked() {
	s.awaitingReply = true
	s.inputSentAt = time.Now()
	s.talkedOverUntil = nil
	if !s.botQuiet {
		until := forever
		s.talkedOverUntil = &until
	}
	s.observing = false
	s.signalLocked()
}

// ObservingBoundary starts a turn that sends nothing: what the bot says next
// belongs to it.
//
// What the bot says until it starts speaking again, however late its
// transcription lands, is the previous turn's and is discarded. What it says
// from then on is the turn's, even when its LLM did not produce it.
func (s *EventStream) ObservingBoundary() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnBoundaryLocked()
	until := forever
	s.talkedOverUntil = &until
	s.observing = true
}

// interruptedLocked handles the bot reporting an interruption: its unfinished
// response is dropped and the stream waits for a fresh reply.
//
// The queue is left as it is. The bot reports the user's turn starting just
// ahead of the interruption, and a driver may not have read those reports yet.
// The interrupted response's text and spoken turn that still arrive are held
// back until the bot's next response starts (see awaitingReply), and a send
// drops whatever is left before it.
func (s *EventStream) interruptedLocked() {
	s.textBuffer = nil
	s.awaitingReply = true
	s.botQuiet = true
	s.signalLocked()
}

// botStartedSpeakingLocked handles the bot reporting it started speaking: it is
// no longer quiet, and its speech now is the turn's.
func (s *EventStream) botStartedSpeakingLocked() {
	s.botQuiet = false
	if s.talkedOverUntil != nil && s.talkedOverUntil.Equal(forever) {
		now := time.Now()
		s.talkedOverUntil = &now
	}
	if s.observing {
		s.observing = false
		s.awaitingReply = false
	}
	s.signalLocked()
}

// eventForRTVIMessageLocked maps one of the bot's reports, about its own output
// or about the harness, to its event, if any. raw is the message as it arrived.
//
//nolint:cyclop // one case per message the bot sends
func (s *EventStream) eventForRTVIMessageLocked(raw []byte) map[string]any {
	var message map[string]any
	if err := json.Unmarshal(raw, &message); err != nil {
		return nil
	}
	msgType, _ := message[keyType].(string)
	data, _ := message[keyData].(map[string]any)
	switch msgType {
	case rtvi.TypeBotStartedSpeaking:
		return map[string]any{keyType: EventBotStartedSpeaking}
	case rtvi.TypeBotStoppedSpeaking:
		return map[string]any{keyType: EventBotStoppedSpeaking}
	case rtvi.TypeBotTTSText:
		if s.botAudio {
			text, _ := data[keyText].(string)
			return segmentEvent(EventTTSResponse, text)
		}
		return nil
	case rtvi.TypeBotLLMStarted, rtvi.TypeBotLLMText, rtvi.TypeBotLLMMarker, rtvi.TypeBotLLMStopped:
		return s.llmReportToEventLocked(msgType, data)
	case rtvi.TypeLLMFunctionCall:
		// The name and arguments may be absent at the bot's default report
		// level.
		args, _ := data["arguments"].(map[string]any)
		if args == nil {
			args = map[string]any{}
		}
		return map[string]any{keyType: EventFunctionCall, keyName: optionalName(data), keyArgs: args}
	case rtvi.TypeLLMFunctionCallStop:
		// How the call ended is the assertable part, so whether it was canceled
		// sits in args beside the id: a scenario matches both through the same
		// calls:/args: check a function_call uses.
		toolCallID, _ := data["tool_call_id"].(string)
		return map[string]any{
			keyType: EventFunctionCallStopped,
			keyName: optionalName(data),
			keyArgs: map[string]any{"tool_call_id": toolCallID, keyCanceled: truthy(data[keyCanceled])},
		}
	case rtvi.TypeUserStartedSpeaking:
		return map[string]any{keyType: EventUserStartedSpeaking}
	case rtvi.TypeBotInterrupted:
		return map[string]any{keyType: EventBotInterrupted}
	case rtvi.TypeUserStoppedSpeaking:
		return map[string]any{keyType: EventUserStoppedSpeaking}
	case rtvi.TypeVADUserStarted:
		return map[string]any{keyType: EventVADUserStartedSpeaking}
	case rtvi.TypeVADUserStopped:
		return map[string]any{keyType: EventVADUserStoppedSpeaking}
	case rtvi.TypeBotOutput:
		if event := botBackchannelEvent(data); event != nil {
			return event
		}
		return s.expressionEventLocked(data)
	case rtvi.TypeUserInput:
		return userInputEvent(data)
	case rtvi.TypeMetrics:
		if data == nil {
			data = map[string]any{}
		}
		return map[string]any{keyType: EventMetrics, keyData: data}
	case rtvi.TypeServerMessage:
		return serverMessageEvent(message[keyData], raw)
	case rtvi.TypeFlowNodeEntered:
		node, _ := data["node"].(string)
		return map[string]any{keyType: EventFlowNode, keyData: maps.Clone(data), keyText: node}
	case evalws.BotImageType:
		return map[string]any{
			keyType: EventImage, "width": data["width"], "height": data["height"], "format": data["format"],
		}
	}
	return nil
}

// llmReportToEventLocked is the event for one of the bot's bot-llm-* reports,
// buffering its text until the response ends.
func (s *EventStream) llmReportToEventLocked(msgType string, data map[string]any) map[string]any {
	if msgType == rtvi.TypeBotLLMStarted {
		s.awaitingReply = false
		s.textBuffer = nil
		s.llmTextAt = nil
		s.expressionDelivery = nil
		return map[string]any{keyType: EventLLMStarted}
	}
	if s.awaitingReply {
		// A straggler of a response from before the turn's reply.
		if msgType == rtvi.TypeBotLLMStopped {
			s.textBuffer = nil
		}
		return nil
	}
	switch msgType {
	case rtvi.TypeBotLLMText:
		if len(s.textBuffer) == 0 {
			at := s.elapsedLocked()
			s.llmTextAt = &at
		}
		text, _ := data[keyText].(string)
		s.textBuffer = append(s.textBuffer, text)
		return nil
	case rtvi.TypeBotLLMMarker:
		text, _ := data[keyText].(string)
		raw, _ := data["raw"].(string)
		markers := knownMarkers(data[keyMarkers])
		if markers == nil {
			markers = []string{}
		}
		return map[string]any{
			keyType: EventLLMMarker, keyText: text, "kind": data["kind"], "raw": raw, keyMarkers: markers,
		}
	}
	event := segmentEvent(EventLLMResponse, strings.Join(s.textBuffer, ""))
	if s.llmTextAt != nil {
		event["started_at"] = *s.llmTextAt
	}
	return event
}

// serverMessageEvent is a message of the bot's own making, from a
// server-message: data, in any JSON shape, and raw, the message it came in.
//
// Its text is what a scenario's content checks read and what a match records:
// the data itself when it is a string, else the data as JSON, laid out with a
// space after each comma and colon and its keys in the order the bot sent
// them.
func serverMessageEvent(data any, raw []byte) map[string]any {
	text, ok := data.(string)
	if !ok {
		var message struct {
			Data json.RawMessage `json:"data"`
		}
		_ = json.Unmarshal(raw, &message)
		text = jsonText(message.Data)
	}
	return map[string]any{keyType: EventServerMessage, keyData: data, keyText: text}
}

// jsonText is raw JSON laid out with a space after each comma and colon, its
// keys in the order they came and its text as it is, unescaped beyond what
// JSON requires. Empty raw JSON is null.
func jsonText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "null"
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var b strings.Builder
	if err := writeJSONText(&b, dec); err != nil {
		return string(raw)
	}
	return b.String()
}

// writeJSONText writes the next JSON value dec holds to b, as jsonText lays it
// out.
func writeJSONText(b *strings.Builder, dec *json.Decoder) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	switch t := tok.(type) {
	case json.Delim:
		b.WriteRune(rune(t))
		for i := 0; dec.More(); i++ {
			if i > 0 {
				b.WriteString(", ")
			}
			if t == '{' {
				key, err := dec.Token()
				if err != nil {
					return err
				}
				keyText, _ := key.(string)
				b.WriteString(jsonString(keyText) + ": ")
			}
			if err := writeJSONText(b, dec); err != nil {
				return err
			}
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		b.WriteRune(rune(end.(json.Delim))) //nolint:forcetypeassert // the decoder closes what it opened
	case string:
		b.WriteString(jsonString(t))
	case json.Number:
		b.WriteString(t.String())
	case bool:
		b.WriteString(strconv.FormatBool(t))
	case nil:
		b.WriteString("null")
	}
	return nil
}

// jsonString is s as a JSON string, escaping only what JSON requires.
func jsonString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20:
			b.WriteString(`\u` + hex4(r))
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// userInputEvent is the user's final transcription or backchannel, from a
// user-input.
func userInputEvent(data map[string]any) map[string]any {
	text, _ := data[keyText].(string)
	switch data["input_type"] {
	case "transcription":
		if truthy(data["final"]) {
			return map[string]any{keyType: EventUserTranscription, keyTranscript: text}
		}
	case "backchannel":
		return map[string]any{keyType: EventUserBackchannel, keyText: text}
	}
	return nil
}

// botBackchannelEvent is the bot's backchannel, from the bot-output reporting it
// was spoken to the end.
func botBackchannelEvent(data map[string]any) map[string]any {
	if data["output_type"] != "backchannel" || data["spoken_status"] != "completed" {
		return nil
	}
	progress, _ := data["spoken_progress"].(map[string]any)
	text, _ := progress["accumulated_text"].(string)
	if text == "" {
		text, _ = data[keyText].(string)
	}
	return map[string]any{keyType: EventBotBackchannel, keyText: text}
}

// expressionEventLocked is the expression the bot performed, from a bot-output,
// written as the LLM writes it.
//
// Only expressive conversation reports it: an event or pause segment, once, or
// spoken text carrying a delivery, where it differs from the delivery before it
// in the response. Each is reported as it is spoken rather than announced, so
// they keep the order they were performed in.
func (s *EventStream) expressionEventLocked(data map[string]any) map[string]any {
	textType, _ := data["text_type"].(string)
	segmentID := data["segment_id"]
	if !truthy(data["will_be_spoken"]) || data["spoken_status"] == "new" {
		return nil
	}
	if textType == string(frames.TextTypeExpressionSound) || textType == string(frames.TextTypeExpressionPause) {
		if s.expressionSegments[segmentID] {
			return nil
		}
		s.expressionSegments[segmentID] = true
		text, _ := data[keyText].(string)
		tag := "[[" + text + "]]"
		if textType == string(frames.TextTypeExpressionPause) {
			tag = "[[pause " + text + "]]"
		}
		return map[string]any{keyType: EventExpression, keyText: tag}
	}
	raw, ok := data["delivery"].([]any)
	if !ok {
		return nil
	}
	delivery := make([]string, 0, len(raw))
	for _, d := range raw {
		if str, ok := d.(string); ok {
			delivery = append(delivery, str)
		}
	}
	if slices.Equal(delivery, s.expressionDelivery) {
		return nil
	}
	s.expressionDelivery = delivery
	text := strings.Join(delivery, ", ")
	if text == "" {
		text = "neutral"
	}
	return map[string]any{keyType: EventExpression, keyText: "<<" + text + ">>"}
}

// segmentEvent is one response segment; the text may be empty, as after an
// interruption.
func segmentEvent(eventType, text string) map[string]any {
	return map[string]any{keyType: eventType, keyText: text}
}
