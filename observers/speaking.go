package observers

import (
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// SpeechEventKind is what happened, to whom, and at which layer.
//
// The user appears at two layers, and they answer different questions.
// UserSpeech* is the speech itself, as the voice activity detector heard it:
// speech that never becomes a turn (a cough, a false start, a pause
// mid-sentence) appears only here. UserTurn* is the turn strategy's ruling on
// that speech, which is what the rest of the pipeline acts on, and follows the
// speech by however long the ruling took.
type SpeechEventKind string

// The moments a conversation is made of.
const (
	UserSpeechStarted SpeechEventKind = "user_speech_started"
	UserSpeechStopped SpeechEventKind = "user_speech_stopped"
	UserTurnStarted   SpeechEventKind = "user_turn_started"
	UserTurnStopped   SpeechEventKind = "user_turn_stopped"
	BotSpeechStarted  SpeechEventKind = "bot_speech_started"
	BotSpeechStopped  SpeechEventKind = "bot_speech_stopped"
	Interruption      SpeechEventKind = "interruption"
)

// SpeechEvent is one moment in the conversation's speaking lifecycle.
type SpeechEvent struct {
	// Kind is what happened, to whom, and at which layer.
	Kind SpeechEventKind
	// Timestamp is the wall-clock time of the moment itself. Speech is timed to
	// when it began and ended, not to when the detector confirmed it, so an
	// interval drawn from these matches what was said.
	Timestamp time.Time
	// StartedAt is when the matching stretch of speech began, on the moments
	// that end one, so a stretch reads as an interval without pairing it with
	// the moment that opened it. The zero value means the stretch began before
	// the observer was watching.
	StartedAt time.Time
}

// SpeakingConfig configures a Speaking observer.
type SpeakingConfig struct {
	// MaxFrames is unused.
	//
	// Deprecated: the observer is told about each frame once, so it keeps no
	// window of the frames it has seen.
	MaxFrames int
	// Now reads the current time. Nil uses time.Now. Supplying one lets a test
	// place moments without waiting.
	Now func() time.Time
	// OnSpeechEvent is called for each moment, as a SpeechEvent.
	OnSpeechEvent func(e SpeechEvent)
}

// Speaking reports the speaking lifecycle of a conversation.
//
// A conversation is a sequence of people taking the floor and occasionally
// taking it from each other. Every moment is reported as it happens, and the
// moments that close a stretch of speech name where it began, so an interval
// reads whole from one record: a stretch whose closing moment never arrives
// stays open rather than quietly joining itself to the next one.
//
// What a turn is stays with the reader. A turn built here would freeze one
// definition into every record, where the moments themselves can be grouped
// again later, differently, over the same history.
type Speaking struct {
	cfg SpeakingConfig

	mu sync.Mutex
	// open is when each open stretch of speech began, so the moment that closes
	// one can carry it.
	open map[SpeechEventKind]time.Time
}

// NewSpeaking builds a Speaking observer.
func NewSpeaking(cfg SpeakingConfig) *Speaking {
	return &Speaking{
		cfg:  cfg,
		open: map[SpeechEventKind]time.Time{},
	}
}

// now reads the clock the observer was configured with.
func (o *Speaking) now() time.Time {
	if o.cfg.Now != nil {
		return o.cfg.Now()
	}
	return time.Now()
}

// speechMoment is what a frame represents, before the observer has decided what
// to do with it: the kind of moment, when it happened, and which stretch it
// opens or closes.
type speechMoment struct {
	kind SpeechEventKind
	at   time.Time
	// opens reports that this moment begins a stretch of speech.
	opens bool
	// closes is the kind that opened the stretch this moment ends, and "" on a
	// moment that ends nothing.
	closes SpeechEventKind
}

// momentOf builds the moment a frame represents, reporting false for a frame
// that is not part of the speaking lifecycle. It reads no state, so a frame that
// turns out to be a duplicate has changed nothing by reaching here.
func momentOf(f frames.Frame, now time.Time) (speechMoment, bool) {
	switch f := f.(type) {
	case *frames.VADUserStartedSpeakingFrame:
		// The detector's account of when speech began, which precedes its
		// confirmation by the time it needed to be sure.
		return speechMoment{kind: UserSpeechStarted, at: speechStart(f, now), opens: true}, true
	case *frames.VADUserStoppedSpeakingFrame:
		return speechMoment{kind: UserSpeechStopped, at: speechStop(f, now), closes: UserSpeechStarted}, true
	case *frames.UserStartedSpeakingFrame:
		return speechMoment{kind: UserTurnStarted, at: now, opens: true}, true
	case *frames.UserStoppedSpeakingFrame:
		return speechMoment{kind: UserTurnStopped, at: now, closes: UserTurnStarted}, true
	case *frames.BotStartedSpeakingFrame:
		return speechMoment{kind: BotSpeechStarted, at: now, opens: true}, true
	case *frames.BotStoppedSpeakingFrame:
		return speechMoment{kind: BotSpeechStopped, at: now, closes: BotSpeechStarted}, true
	case *frames.InterruptionFrame:
		return speechMoment{kind: Interruption, at: now}, true
	default:
		return speechMoment{}, false
	}
}

// ObserveEveryPush implements processor.EveryPushObserver: a moment is reported
// once, on the first push of the frame that represents it.
func (o *Speaking) ObserveEveryPush() bool { return false }

// OnPushFrame implements processor.Observer. It reports the moment a frame
// represents.
func (o *Speaking) OnPushFrame(data processor.FramePushed) {
	// An interruption is broadcast, arriving as two frames, each pushed for the
	// first time once. Read the downstream one.
	if skipBroadcastSibling(data.Frame, data.Direction) {
		return
	}

	o.mu.Lock()
	m, ok := momentOf(data.Frame, o.now())
	if !ok {
		o.mu.Unlock()
		return
	}

	e := SpeechEvent{Kind: m.kind, Timestamp: m.at}
	switch {
	case m.opens:
		o.open[m.kind] = m.at
	case m.closes != "":
		// A stretch that began before the observer was watching closes without a
		// start rather than borrowing one from another stretch.
		e.StartedAt = o.open[m.closes]
		delete(o.open, m.closes)
	}
	o.mu.Unlock()

	if o.cfg.OnSpeechEvent != nil {
		o.cfg.OnSpeechEvent(e)
	}
}

// Compile-time interface check.
var _ processor.Observer = (*Speaking)(nil)
