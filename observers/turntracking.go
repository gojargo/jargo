package observers

import (
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// TurnTrackingConfig configures a TurnTracking observer.
type TurnTrackingConfig struct {
	// MaxFrames is how many recent frame ids the observer remembers to
	// recognize one it has already counted; 0 uses 100.
	MaxFrames int
	// TurnEndTimeout is how long after the bot stops speaking a turn ends; 0 uses
	// 2.5s. The delay lets a turn survive a brief gap between bot utterances (an
	// HTTP TTS boundary, a function call) without splitting into two turns.
	TurnEndTimeout time.Duration
	// OnTurnStarted is called when a turn begins, with the 1-based turn number.
	OnTurnStarted func(turn int)
	// OnTurnEnded is called when a turn ends, with the turn number, its duration,
	// and whether it was cut short by an interruption.
	OnTurnEnded func(turn int, duration time.Duration, interrupted bool)
}

// TurnTracking tracks conversational turns. The first turn starts with the
// pipeline; a turn ends when the bot finishes speaking (after TurnEndTimeout) or
// is interrupted by the user, at which point the next turn starts.
//
// A turn is timed on the pipeline clock, from the frame that opened it to the
// frame that closed it. A turn ended by the timeout is therefore measured to the
// moment the bot fell silent, not to the moment the timer fired: the wait exists
// to tell a pause apart from an ending, and it is not part of the turn.
type TurnTracking struct {
	cfg TurnTrackingConfig

	mu       sync.Mutex
	dd       deduper
	active   bool
	botTalk  bool
	botSpoke bool
	count    int
	start    time.Duration
	timer    *time.Timer
	// ended are the listeners added with OnTurnEnded, on top of the one the
	// config carries. A processor that needs to know where a turn ended, the
	// audio buffer recording per-turn audio say, subscribes here rather than
	// taking the single config callback away from the application.
	ended []func(turn int, duration time.Duration, interrupted bool)
}

// OnTurnEnded adds a listener called when a turn ends, alongside the one the
// config carries and any added before it.
func (o *TurnTracking) OnTurnEnded(fn func(turn int, duration time.Duration, interrupted bool)) {
	if fn == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.ended = append(o.ended, fn)
}

// NewTurnTracking builds a TurnTracking observer.
func NewTurnTracking(cfg TurnTrackingConfig) *TurnTracking {
	if cfg.TurnEndTimeout == 0 {
		cfg.TurnEndTimeout = defaultTurnEndTimeout
	}
	return &TurnTracking{cfg: cfg, dd: newDeduper(cfg.MaxFrames)}
}

// OnPushFrame implements processor.Observer.
func (o *TurnTracking) OnPushFrame(data processor.FramePushed) {
	f, dir := data.Frame, data.Direction
	if skipBroadcastSibling(f, dir) {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.dd.seenBefore(f.ID()) {
		return
	}
	switch f.(type) {
	case *frames.StartFrame:
		if o.count == 0 {
			o.startTurn(data.Timestamp)
		}
	case *frames.UserStartedSpeakingFrame:
		o.userStarted(data.Timestamp)
	case *frames.BotStartedSpeakingFrame:
		o.botTalk = true
		o.botSpoke = true
		o.cancelTimer()
	case *frames.BotStoppedSpeakingFrame:
		if o.botTalk {
			o.botTalk = false
			o.scheduleEnd(data.Timestamp)
		}
	case *frames.EndFrame, *frames.CancelFrame:
		if o.active {
			o.cancelTimer()
			o.endTurn(data.Timestamp, true)
		}
	}
}

// userStarted advances the turn state when the user starts speaking, ending the
// current turn (as an interruption while the bot speaks, or normally after it
// has spoken) and beginning a new one.
func (o *TurnTracking) userStarted(at time.Duration) {
	switch {
	case o.botTalk:
		o.cancelTimer()
		o.endTurn(at, true)
		o.botTalk = false
		o.startTurn(at)
	case o.active && o.botSpoke:
		o.cancelTimer()
		o.endTurn(at, false)
		o.startTurn(at)
	case !o.active:
		o.startTurn(at)
	}
}

func (o *TurnTracking) startTurn(at time.Duration) {
	o.active = true
	o.botSpoke = false
	o.count++
	o.start = at
	if o.cfg.OnTurnStarted != nil {
		o.cfg.OnTurnStarted(o.count)
	}
}

func (o *TurnTracking) endTurn(at time.Duration, interrupted bool) {
	if !o.active {
		return
	}
	d := at - o.start
	o.active = false
	turn := o.count
	if o.cfg.OnTurnEnded != nil {
		o.cfg.OnTurnEnded(turn, d, interrupted)
	}
	for _, fn := range o.ended {
		fn(turn, d, interrupted)
	}
}

// scheduleEnd arms the turn-end timer for the current turn, to end it as of at,
// the moment the bot fell silent. The turn-number guard makes a late fire a
// no-op if a new turn has since started.
func (o *TurnTracking) scheduleEnd(at time.Duration) {
	o.cancelTimer()
	turn := o.count
	o.timer = time.AfterFunc(o.cfg.TurnEndTimeout, func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		if o.active && !o.botTalk && o.count == turn {
			o.endTurn(at, false)
			o.timer = nil
		}
	})
}

func (o *TurnTracking) cancelTimer() {
	if o.timer != nil {
		o.timer.Stop()
		o.timer = nil
	}
}

// Compile-time interface check.
var _ processor.Observer = (*TurnTracking)(nil)
