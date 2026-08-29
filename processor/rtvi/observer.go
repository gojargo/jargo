package rtvi

import (
	"context"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/loudness"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Observer reports pipeline events to an RTVI client. It watches every frame
// handed between processors and turns the ones a client cares about into RTVI
// messages, which it hands to a Processor to send.
//
// Only outgoing events come from here. Messages arriving from the client are
// handled by the Processor, which also carries out the handshake.
//
// Watching rather than sitting in the pipeline is what lets a client be told
// about frames that never travel the whole chain, and lets an event be reported
// from where the frame actually is: text pushed by the LLM is seen at the LLM,
// whether or not anything downstream forwards it.
type Observer struct {
	sink *Processor

	mu    sync.Mutex
	seen  map[uint64]struct{}
	order []uint64

	// paramsMu guards params, which a ConfigureObserverFrame rewrites while the
	// pipeline is running.
	paramsMu sync.Mutex
	params   ObserverParams

	// levelMu guards the volume tracking below. Audio for the two sides arrives
	// from different processors, so on different goroutines.
	levelMu      sync.Mutex
	userVolume   loudness.Tracker
	botVolume    loudness.Tracker
	lastUserSent time.Time
	lastBotSent  time.Time
}

// FunctionCallReportLevel is how much of a tool call is exposed in the RTVI
// function-call events. A call's name and its arguments can carry information a
// client has no business seeing, so what is reported is a per-function setting
// rather than a fixed payload.
type FunctionCallReportLevel string

// The report levels, in increasing order of disclosure.
const (
	// ReportDisabled emits no function-call event at all for the call.
	ReportDisabled FunctionCallReportLevel = "disabled"
	// ReportNone emits the event with the tool call id only. This is the default:
	// a client learns that a call is running, and nothing more.
	ReportNone FunctionCallReportLevel = "none"
	// ReportName adds the function's name, still without arguments or result.
	ReportName FunctionCallReportLevel = "name"
	// ReportFull adds the function's name, its arguments and its result.
	ReportFull FunctionCallReportLevel = "full"
)

// ObserverParams configures what an Observer reports.
//
// The categories a client normally wants are on unless they are turned off, so
// their fields are pointers: nil means on, and only an explicit false suppresses
// them. The ones that would flood a client that has not asked for them are off
// unless turned on, and are plain bools.
type ObserverParams struct {
	// FunctionCallReportLevel maps a function name to the level of detail its
	// events carry. The "*" key sets the level for functions not listed; when it
	// is absent too, ReportNone applies.
	FunctionCallReportLevel map[string]FunctionCallReportLevel
	// BotLLMEnabled reports what the model produced: the brackets around a
	// response and the text inside it. Nil leaves it on.
	BotLLMEnabled *bool
	// BotTTSEnabled reports what the voice is doing: the brackets around a
	// synthesis and the text as it is spoken. Nil leaves it on.
	BotTTSEnabled *bool
	// BotSpeakingEnabled reports when the bot's audio starts and stops playing,
	// and when it was cut off. Nil leaves it on.
	BotSpeakingEnabled *bool
	// UserSpeakingEnabled reports the user's turn starting and ending. This is
	// the turn as the pipeline settled it, which a strategy may gate or defer;
	// VADUserSpeakingEnabled is the raw signal behind it. Nil leaves it on.
	UserSpeakingEnabled *bool
	// UserTranscriptionEnabled reports what the user said, both the interim
	// guesses and the final transcript. Nil leaves it on.
	UserTranscriptionEnabled *bool
	// MetricsEnabled reports the timings and usage the pipeline measures. Nil
	// leaves it on.
	MetricsEnabled *bool
	// IgnoredSources are processors whose frames the observer says nothing
	// about. It keeps a secondary branch of the pipeline out of the client's
	// view: an evaluation model answering alongside the real one has a whole
	// conversation of its own, and none of it is for the client to see. Sources
	// can be added and removed while the pipeline runs with AddIgnoredSource and
	// RemoveIgnoredSource.
	IgnoredSources []processor.Processor
	// VADUserSpeakingEnabled reports the raw VAD speaking signal as well as the
	// turn-level one. The two differ whenever a turn strategy gates or defers a
	// turn, which is what makes the raw signal useful as a timing anchor. Off by
	// default, because a client wants turns rather than the signal behind them.
	VADUserSpeakingEnabled bool
	// UserAudioLevelEnabled reports how loud the user is, for a client drawing a
	// speaking meter. Off by default: it is a message every AudioLevelPeriod for
	// as long as the call lasts, which a client that draws nothing does not want.
	UserAudioLevelEnabled bool
	// BotAudioLevelEnabled reports how loud the bot is. Off by default, for the
	// same reason.
	BotAudioLevelEnabled bool
	// AudioLevelPeriod is how often an audio level is reported while it is
	// enabled; zero defaults to 150 ms. Audio arrives far more often than a
	// meter can usefully be redrawn, and measuring loudness is not free, so the
	// level is reported on a period rather than per frame.
	AudioLevelPeriod time.Duration
}

// defaultAudioLevelPeriod is how often an audio level is reported when
// AudioLevelPeriod is unset.
const defaultAudioLevelPeriod = 150 * time.Millisecond

// audioLevelPeriod is the configured reporting period, or the default.
func (p ObserverParams) audioLevelPeriod() time.Duration {
	if p.AudioLevelPeriod <= 0 {
		return defaultAudioLevelPeriod
	}
	return p.AudioLevelPeriod
}

// DefaultObserverParams is the configuration NewObserver uses: function-call
// events carry the tool call id alone.
func DefaultObserverParams() ObserverParams {
	return ObserverParams{
		FunctionCallReportLevel: map[string]FunctionCallReportLevel{"*": ReportNone},
		AudioLevelPeriod:        defaultAudioLevelPeriod,
	}
}

// seenCap bounds the ids remembered for deduplication. A frame is reported once
// per handover, so each is recognized on sight, and only a bounded window has to
// be kept: by the time an id is evicted the frame has long since left.
const seenCap = 4096

// NewObserver builds an observer that sends through sink, with the default
// parameters. Use NewObserverWithParams to report more of a tool call than its
// id.
func NewObserver(sink *Processor) *Observer {
	return NewObserverWithParams(sink, DefaultObserverParams())
}

// NewObserverWithParams builds an observer that sends through sink and reports
// what params allows.
func NewObserverWithParams(sink *Processor, params ObserverParams) *Observer {
	return &Observer{
		sink:   sink,
		seen:   make(map[uint64]struct{}, seenCap),
		params: params,
	}
}

// OnPushFrame implements processor.Observer.
func (o *Observer) OnPushFrame(data processor.FramePushed) {
	f, dir := data.Frame, data.Direction
	// A broadcast frame is pushed both ways; report only the downstream copy so
	// the client is not told twice.
	if _, broadcast := f.Base().BroadcastSiblingID(); broadcast && dir != processor.Downstream {
		return
	}
	if o.alreadySeen(f.ID()) {
		return
	}
	// A branch of the pipeline the client is not meant to see says nothing at
	// all, whatever the frame is.
	//
	// Checked after the frame has been recorded, not before, because a frame is
	// handed over more than once on its way out: a pipeline pushes what leaves
	// it onward under its own name, so passing over the branch's handover
	// without recording it would let the pipeline's report it a moment later.
	if o.ignores(data.Source) {
		return
	}
	if cfg, ok := f.(*ConfigureObserverFrame); ok {
		o.applyConfig(cfg)
		return
	}
	for _, msg := range o.messagesFor(f) {
		o.sink.Send(msg)
	}
}

// applyConfig applies a runtime reconfiguration, leaving unset fields alone.
func (o *Observer) applyConfig(f *ConfigureObserverFrame) {
	o.paramsMu.Lock()
	if f.FunctionCallReportLevel != nil {
		o.params.FunctionCallReportLevel = f.FunctionCallReportLevel
	}
	if f.VADUserSpeakingEnabled != nil {
		o.params.VADUserSpeakingEnabled = *f.VADUserSpeakingEnabled
	}
	o.paramsMu.Unlock()
	slog.Debug("RTVI observer reconfigured",
		"function_call_report_level", f.FunctionCallReportLevel,
		"vad_user_speaking", f.VADUserSpeakingEnabled)
}

// audioLevelMessageFor feeds the audio to the rolling window for its side of the
// conversation and reports the level when one is due.
//
// Every frame feeds the window, but the window is only measured when a level is
// due to be sent: audio arrives far more often than a meter can be redrawn, and
// measuring loudness costs a few hundred microseconds. The second result reports
// whether the frame was audio at all, so the dispatch can stop looking.
func (o *Observer) audioLevelMessageFor(f frames.Frame) (Message, bool, bool) {
	var (
		audio    frames.AudioRawData
		tracker  *loudness.Tracker
		lastSent *time.Time
		enabled  bool
		build    func(float64) Message
	)

	o.paramsMu.Lock()
	params := o.params
	o.paramsMu.Unlock()

	switch fr := f.(type) {
	case *frames.InputAudioRawFrame:
		audio, enabled, build = fr.AudioRawData, params.UserAudioLevelEnabled, UserAudioLevel
		tracker, lastSent = &o.userVolume, &o.lastUserSent
	case *frames.TTSAudioRawFrame:
		audio, enabled, build = fr.AudioRawData, params.BotAudioLevelEnabled, BotAudioLevel
		tracker, lastSent = &o.botVolume, &o.lastBotSent
	default:
		return Message{}, false, false
	}

	if !enabled {
		return Message{}, false, true
	}

	o.levelMu.Lock()
	defer o.levelMu.Unlock()

	tracker.Update(audio.Audio, audio.SampleRate)

	now := time.Now()
	if !lastSent.IsZero() && now.Sub(*lastSent) <= params.audioLevelPeriod() {
		return Message{}, false, true
	}
	*lastSent = now
	return build(tracker.Volume()), true, true
}

// onUnlessOff reads a category that is reported unless it was explicitly turned
// off, which is what a nil setting means.
func onUnlessOff(p *bool) bool { return p == nil || *p }

// enabled reports whether the category read by pick is exposed.
func (o *Observer) enabled(pick func(ObserverParams) *bool) bool {
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	return onUnlessOff(pick(o.params))
}

// AddIgnoredSource stops the observer reporting anything about frames pushed by
// source. See ObserverParams.IgnoredSources.
func (o *Observer) AddIgnoredSource(source processor.Processor) {
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	if !slices.Contains(o.params.IgnoredSources, source) {
		o.params.IgnoredSources = append(o.params.IgnoredSources, source)
	}
}

// RemoveIgnoredSource undoes an AddIgnoredSource. Removing one that was never
// ignored does nothing.
func (o *Observer) RemoveIgnoredSource(source processor.Processor) {
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	o.params.IgnoredSources = slices.DeleteFunc(o.params.IgnoredSources,
		func(p processor.Processor) bool { return p == source })
}

// ignores reports whether frames from source are passed over in silence.
func (o *Observer) ignores(source processor.Processor) bool {
	if source == nil {
		return false
	}
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	return slices.Contains(o.params.IgnoredSources, source)
}

// vadUserSpeakingEnabled reports whether the raw VAD speaking signal is exposed.
func (o *Observer) vadUserSpeakingEnabled() bool {
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	return o.params.VADUserSpeakingEnabled
}

// reportLevelFor is the level to report a call to name at: the function's own
// entry, else the "*" default, else ReportNone.
func (o *Observer) reportLevelFor(name string) FunctionCallReportLevel {
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	if level, ok := o.params.FunctionCallReportLevel[name]; ok {
		return level
	}
	if level, ok := o.params.FunctionCallReportLevel["*"]; ok {
		return level
	}
	return ReportNone
}

// alreadySeen reports whether the frame has been reported, recording it when it
// has not.
func (o *Observer) alreadySeen(id uint64) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if _, ok := o.seen[id]; ok {
		return true
	}
	o.seen[id] = struct{}{}
	o.order = append(o.order, id)
	if len(o.order) > seenCap {
		delete(o.seen, o.order[0])
		o.order = o.order[1:]
	}
	return false
}

// Send hands an RTVI message to the client. It is safe to call from any
// goroutine: the message is queued to the next processor like any other frame.
func (p *Processor) Send(msg Message) {
	ctx := p.sendContext()
	_ = p.PushFrame(ctx, frames.NewOutputTransportMessageUrgentFrame(msg), processor.Downstream)
}

// sendContext is the context an out-of-band send runs under.
func (p *Processor) sendContext() context.Context {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.baseCtx != nil {
		return p.baseCtx
	}
	return context.Background()
}

var _ processor.Observer = (*Observer)(nil)
