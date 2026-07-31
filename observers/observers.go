// Package observers provides pipeline observers: components that watch the frames
// flowing through a pipeline to derive turn, latency and startup metrics, or to
// log the stream, without modifying it. Register them via
// pipeline.TaskParams.Observers.
//
// The pipeline reports frames at its two ends, so these observers track the
// turn-taking and bot-output frames that travel there (StartFrame, the
// user/bot speaking frames, TTS audio). Each observer is safe for concurrent
// use: the two ends run on separate goroutines.
package observers

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// skipBroadcastSibling reports whether a frame is the upstream half of a
// broadcast pair and should be ignored. A broadcast builds a distinct frame for
// each direction, paired by BroadcastSiblingID, so an observer that watched both
// would report one event twice. Counting only the downstream half reports it
// once, and the pairing is what makes the two halves recognizable — their ids
// deliberately differ, so the id deduper below cannot catch them.
func skipBroadcastSibling(f frames.Frame, dir processor.Direction) bool {
	_, paired := f.Base().BroadcastSiblingID()
	return paired && dir != processor.Downstream
}

// deduper drops a frame already seen, keeping a bounded window of recent ids. It
// catches the same instance arriving twice — a frame pushed on and later echoed
// back — which is distinct from the broadcast pairing above.
type deduper struct {
	seen  map[uint64]struct{}
	order []uint64
	max   int
}

func newDeduper() deduper { return deduper{seen: map[uint64]struct{}{}, max: 256} }

// seenBefore reports whether id was already observed, recording it otherwise.
func (d *deduper) seenBefore(id uint64) bool {
	if _, ok := d.seen[id]; ok {
		return true
	}
	d.seen[id] = struct{}{}
	d.order = append(d.order, id)
	if len(d.order) > d.max {
		old := d.order[0]
		d.order = d.order[1:]
		delete(d.seen, old)
	}
	return false
}

// defaultTurnEndTimeout is how long after the bot stops speaking a turn is
// considered ended, absent a new user turn.
const defaultTurnEndTimeout = 2500 * time.Millisecond

// TurnTrackingConfig configures a TurnTracking observer.
type TurnTrackingConfig struct {
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
type TurnTracking struct {
	cfg TurnTrackingConfig

	mu       sync.Mutex
	dd       deduper
	active   bool
	botTalk  bool
	botSpoke bool
	count    int
	start    time.Time
	timer    *time.Timer
}

// NewTurnTracking builds a TurnTracking observer.
func NewTurnTracking(cfg TurnTrackingConfig) *TurnTracking {
	if cfg.TurnEndTimeout == 0 {
		cfg.TurnEndTimeout = defaultTurnEndTimeout
	}
	return &TurnTracking{cfg: cfg, dd: newDeduper()}
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
			o.startTurn()
		}
	case *frames.UserStartedSpeakingFrame:
		o.userStarted()
	case *frames.BotStartedSpeakingFrame:
		o.botTalk = true
		o.botSpoke = true
		o.cancelTimer()
	case *frames.BotStoppedSpeakingFrame:
		if o.botTalk {
			o.botTalk = false
			o.scheduleEnd()
		}
	case *frames.EndFrame, *frames.CancelFrame:
		if o.active {
			o.cancelTimer()
			o.endTurn(true)
		}
	}
}

// userStarted advances the turn state when the user starts speaking, ending the
// current turn (as an interruption while the bot speaks, or normally after it
// has spoken) and beginning a new one.
func (o *TurnTracking) userStarted() {
	switch {
	case o.botTalk:
		o.cancelTimer()
		o.endTurn(true)
		o.botTalk = false
		o.startTurn()
	case o.active && o.botSpoke:
		o.cancelTimer()
		o.endTurn(false)
		o.startTurn()
	case !o.active:
		o.startTurn()
	}
}

func (o *TurnTracking) startTurn() {
	o.active = true
	o.botSpoke = false
	o.count++
	o.start = time.Now()
	if o.cfg.OnTurnStarted != nil {
		o.cfg.OnTurnStarted(o.count)
	}
}

func (o *TurnTracking) endTurn(interrupted bool) {
	if !o.active {
		return
	}
	d := time.Since(o.start)
	o.active = false
	if o.cfg.OnTurnEnded != nil {
		o.cfg.OnTurnEnded(o.count, d, interrupted)
	}
}

// scheduleEnd arms the turn-end timer for the current turn. The turn-number
// guard makes a late fire a no-op if a new turn has since started.
func (o *TurnTracking) scheduleEnd() {
	o.cancelTimer()
	turn := o.count
	o.timer = time.AfterFunc(o.cfg.TurnEndTimeout, func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		if o.active && !o.botTalk && o.count == turn {
			o.endTurn(false)
		}
	})
}

func (o *TurnTracking) cancelTimer() {
	if o.timer != nil {
		o.timer.Stop()
		o.timer = nil
	}
}

// LatencyConfig configures a UserBotLatency observer.
type LatencyConfig struct {
	// OnLatency is called with the time from the user stopping speaking to the
	// bot starting — the user-perceived response latency.
	OnLatency func(d time.Duration)
}

// UserBotLatency measures the response latency of each turn: the gap between the
// user stopping speaking and the bot starting.
type UserBotLatency struct {
	cfg LatencyConfig

	mu      sync.Mutex
	dd      deduper
	stopped time.Time
	pending bool
}

// NewUserBotLatency builds a UserBotLatency observer.
func NewUserBotLatency(cfg LatencyConfig) *UserBotLatency {
	return &UserBotLatency{cfg: cfg, dd: newDeduper()}
}

// OnPushFrame implements processor.Observer.
func (o *UserBotLatency) OnPushFrame(data processor.FramePushed) {
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
	case *frames.UserStoppedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame:
		o.stopped = time.Now()
		o.pending = true
	case *frames.BotStartedSpeakingFrame:
		if o.pending {
			o.pending = false
			if o.cfg.OnLatency != nil {
				o.cfg.OnLatency(time.Since(o.stopped))
			}
		}
	}
}

// StartupConfig configures a StartupTiming observer.
type StartupConfig struct {
	// OnStartup is called once with the time from pipeline start to the first bot
	// audio — the cold-start latency of the first response.
	OnStartup func(d time.Duration)
}

// StartupTiming measures the time from the pipeline starting to the first bot
// audio, the cold-start latency before the bot first speaks.
type StartupTiming struct {
	cfg StartupConfig

	mu      sync.Mutex
	dd      deduper
	start   time.Time
	started bool
	done    bool
}

// NewStartupTiming builds a StartupTiming observer.
func NewStartupTiming(cfg StartupConfig) *StartupTiming {
	return &StartupTiming{cfg: cfg, dd: newDeduper()}
}

// OnPushFrame implements processor.Observer.
func (o *StartupTiming) OnPushFrame(data processor.FramePushed) {
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
		if !o.started {
			o.start = time.Now()
			o.started = true
		}
	case *frames.BotStartedSpeakingFrame, *frames.TTSAudioRawFrame:
		if o.started && !o.done {
			o.done = true
			if o.cfg.OnStartup != nil {
				o.cfg.OnStartup(time.Since(o.start))
			}
		}
	}
}

// LoggerConfig configures a Logger observer.
type LoggerConfig struct {
	// Logger is the destination; slog.Default() when nil.
	Logger *slog.Logger
	// Level is the log level; the zero value is slog.LevelInfo.
	Level slog.Level
	// Filter, when set, logs only frames for which it returns true.
	Filter func(frames.Frame) bool
}

// Logger logs every frame it observes, for debugging a pipeline's frame flow.
type Logger struct {
	log    *slog.Logger
	level  slog.Level
	filter func(frames.Frame) bool
}

// NewLogger builds a Logger observer.
func NewLogger(cfg LoggerConfig) *Logger {
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	return &Logger{log: log, level: cfg.Level, filter: cfg.Filter}
}

// OnPushFrame implements processor.Observer.
func (o *Logger) OnPushFrame(data processor.FramePushed) {
	f, dir := data.Frame, data.Direction
	if o.filter != nil && !o.filter(f) {
		return
	}
	o.log.Log(context.Background(), o.level, "frame", "frame", f.String(), "dir", dir.String())
}

// Every observer here is reported each handover, so each dedups by frame id.
var (
	_ processor.Observer = (*TurnTracking)(nil)
	_ processor.Observer = (*UserBotLatency)(nil)
	_ processor.Observer = (*StartupTiming)(nil)
	_ processor.Observer = (*Logger)(nil)
)
