package observers

import (
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// TTFBBreakdown is one time-to-first-byte measurement, placed on the timeline of
// the reply it belongs to.
type TTFBBreakdown struct {
	// Processor is the name of the processor that reported it.
	Processor string
	// Model is the model it is attributed to, "" when unknown.
	Model string
	// StartTime is the wall-clock time the measurement started at.
	StartTime time.Time
	// Duration is the measured time to first byte.
	Duration time.Duration
}

// TextAggregationBreakdown is one text-aggregation measurement, placed on the
// timeline of the reply it belongs to.
type TextAggregationBreakdown struct {
	// Processor is the name of the processor that reported it.
	Processor string
	// StartTime is the wall-clock time the measurement started at.
	StartTime time.Time
	// Duration is the measured aggregation time.
	Duration time.Duration
}

// FunctionCallMetrics is how long one tool call took to run.
type FunctionCallMetrics struct {
	// FunctionName is the name of the tool that was called.
	FunctionName string
	// StartTime is the wall-clock time the call started at.
	StartTime time.Time
	// Duration is the time from the call starting to its result arriving.
	Duration time.Duration
}

// LatencyBreakdown accounts for one user-to-bot cycle: what each service in the
// pipeline contributed to the delay the listener heard.
//
// It is collected between the user falling silent and the bot starting to speak,
// and only when the pipeline collects metrics at all: the measurements come from
// the MetricsFrames the services emit, which they only emit when asked to.
type LatencyBreakdown struct {
	// TTFB is what each service took to produce anything at all, in the order
	// the measurements were reported.
	TTFB []TTFBBreakdown
	// TextAggregation is the first text-aggregation measurement of the cycle,
	// which is what grouping the model's tokens into sentences cost before
	// synthesis could start. It is nil when none was reported.
	TextAggregation *TextAggregationBreakdown
	// UserTurnStart is when the user's turn ended in the audio: the moment the
	// speech itself stopped, before the detector had confirmed it. The zero
	// value means no VAD stop was observed.
	UserTurnStart time.Time
	// UserTurn is how long releasing the turn took from that moment: the
	// detector's silence window, the transcriber finalizing, and any wait on an
	// end-of-turn analyzer. It is nil when the turn was never released, which is
	// what a pipeline with no turn analyzer looks like.
	UserTurn *time.Duration
	// FunctionCalls is how long each tool call of the cycle took. It is empty
	// when the reply made none.
	FunctionCalls []FunctionCallMetrics
	// Contributions are the named parts of the interval, in chronological order,
	// summing to the measured latency. A part is listed only if it happened, so
	// a bot without turn completion has no marker or wait entries.
	Contributions []LatencyContribution
	// MeasuredFrom is where the interval was anchored, so a greeting is not
	// compared with a turn. It is MeasuredFromNothing on a breakdown carrying no
	// contributions.
	MeasuredFrom MeasuredFrom
	// Total is the measured interval, which the contributions sum to.
	Total time.Duration
}

// LatencyConfig configures a UserBotLatency observer.
type LatencyConfig struct {
	// MaxFrames is unused.
	//
	// Deprecated: the observer is told about each frame once, so it keeps no
	// window of the frames it has seen.
	MaxFrames int
	// MinContribution is the shortest stretch reported in its own right; 0 uses
	// 5ms. Anything shorter is a frame hop rather than work worth naming, so it
	// is rolled into the single pipeline contribution. Set it negative to list
	// every stretch, individual frame hops included.
	MinContribution time.Duration
	// Now reads the current time. Nil uses time.Now. Supplying one lets a test
	// drive a cycle without waiting out the intervals it describes.
	Now func() time.Time
	// OnLatency is called with the time from the user stopping speaking to the
	// bot starting: the user-perceived response latency.
	OnLatency func(d time.Duration)
	// OnBreakdown is called with the per-service account of the same cycle,
	// alongside every latency the observer reports. It is empty of measurements
	// unless the pipeline collects metrics.
	OnBreakdown func(b LatencyBreakdown)
	// OnFirstBotSpeechLatency is called once, with the time from the client
	// connecting to the bot first speaking. It is not called at all when the
	// user speaks first: the figure means the greeting was slow, and there is no
	// greeting to measure once the conversation has started without one.
	OnFirstBotSpeechLatency func(d time.Duration)
}

// UserBotLatency measures the response latency of each turn: the gap between the
// user stopping speaking and the bot starting. Alongside each measurement it
// reports a LatencyBreakdown accounting for where that time went, and it reports
// separately on the first thing the bot says after a client connects.
//
// It watches downstream frames only. A frame broadcast in both directions is
// therefore counted once, on the way down.
type UserBotLatency struct {
	cfg LatencyConfig

	mu      sync.Mutex
	stopped time.Time
	// turnStart is when the user's speech actually ended, and turn is how long
	// releasing the turn took from there.
	turnStart time.Time
	turn      *time.Duration

	// clientConnected is when the client joined, and firstSpeechDone reports
	// that the first-speech measurement is over, whether it was made or
	// abandoned.
	clientConnected time.Time
	firstSpeechDone bool

	// pipelineStarted is when the StartFrame had reached every processor, which
	// is where the startup report stops measuring. A greeting is timed from
	// there so the two reports meet rather than overlap.
	pipelineStarted time.Time

	// moments are the points of the cycle, in the order they were observed, and
	// llmRequest is the one the model was asked at, kept so a metric reported
	// against that processor can be worked back to its first chunk.
	moments    []moment
	llmRequest *moment
	// markersSeen reports that this bot uses turn completion at all, which a
	// marker frame proves. It is not per-cycle: a bot that emits markers keeps
	// using them.
	markersSeen bool

	// Per-cycle accumulators, cleared whenever a cycle begins or is abandoned.
	ttfb       []TTFBBreakdown
	textAgg    *TextAggregationBreakdown
	callStarts map[string]FunctionCallMetrics
	calls      []FunctionCallMetrics
}

// now reads the clock the observer was configured with.
func (o *UserBotLatency) now() time.Time {
	if o.cfg.Now != nil {
		return o.cfg.Now()
	}
	return time.Now()
}

// minContribution is the threshold below which a stretch is rolled into the
// pipeline contribution.
func (o *UserBotLatency) minContribution() time.Duration {
	if o.cfg.MinContribution == 0 {
		return defaultMinContribution
	}
	return o.cfg.MinContribution
}

// OnPipelineStarted implements processor.PipelineStartedObserver. The StartFrame
// has reached every processor by now, which is where the startup report stops
// measuring.
func (o *UserBotLatency) OnPipelineStarted() {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.pipelineStarted.IsZero() {
		o.pipelineStarted = o.now()
	}
}

// NewUserBotLatency builds a UserBotLatency observer.
func NewUserBotLatency(cfg LatencyConfig) *UserBotLatency {
	o := &UserBotLatency{cfg: cfg}
	o.resetAccumulators()
	return o
}

// ObserveEveryPush implements processor.EveryPushObserver: a frame is counted
// once, on its first push.
func (o *UserBotLatency) ObserveEveryPush() bool { return false }

// OnPushFrame implements processor.Observer.
func (o *UserBotLatency) OnPushFrame(data processor.FramePushed) {
	if data.Direction != processor.Downstream {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()

	// The frames that only place a moment on the timeline are handled apart from
	// the ones that also move the measurement along.
	if o.markTimelineMoment(data) || o.timeToolCall(data) {
		return
	}

	switch f := data.Frame.(type) {
	case *frames.ClientConnectedFrame:
		if o.clientConnected.IsZero() {
			o.clientConnected = o.now()
		}
	case *frames.VADUserStartedSpeakingFrame:
		// A new utterance discards whatever the last one had accumulated.
		o.stopped = time.Time{}
		o.resetAccumulators()
		// The user speaking before the bot ever did means there is no greeting
		// to time, so that measurement is abandoned rather than left pending.
		o.firstSpeechDone = true
	case *frames.VADUserStoppedSpeakingFrame:
		// The detector confirms the stop only after its silence window has
		// elapsed, so the speech itself ended that much earlier. Measuring from
		// there is what makes the figure the delay the user actually heard.
		o.stopped = speechStop(f, o.now())
		o.turnStart = o.stopped
		// The moment is the determination rather than the speech: the wait the
		// timeline names is the silence the detector had to hear.
		at := f.Timestamp
		if at.IsZero() {
			at = o.now()
		}
		o.mark(moment{at: at, kind: momentVADStop}, false)
	case *frames.UserStoppedSpeakingFrame:
		if !o.stopped.IsZero() {
			d := o.now().Sub(o.stopped)
			o.turn = &d
		}
	case *frames.InterruptionFrame:
		// The measurements of a cycle that was cut short describe work nobody
		// heard, so they are dropped rather than charged to the next reply.
		o.resetAccumulators()
	case *frames.MetricsFrame:
		o.handleMetrics(f)
	case *frames.BotStartedSpeakingFrame:
		o.mark(moment{kind: momentBotSpeaking, source: sourceName(data)}, true)
		o.botStartedSpeaking()
	}
}

// timeToolCall times one tool call, from the frame that starts it to the one
// carrying its result, and reports whether it handled the frame. The caller
// holds o.mu.
func (o *UserBotLatency) timeToolCall(data processor.FramePushed) bool {
	switch f := data.Frame.(type) {
	case *frames.FunctionCallInProgressFrame:
		o.callStarts[f.ToolCallID] = FunctionCallMetrics{
			FunctionName: f.ToolName,
			StartTime:    o.now(),
		}
	case *frames.FunctionCallResultFrame:
		call, ok := o.callStarts[f.ToolCallID]
		if !ok {
			return true
		}
		delete(o.callStarts, f.ToolCallID)
		call.Duration = o.now().Sub(call.StartTime)
		o.calls = append(o.calls, call)
	default:
		return false
	}
	return true
}

// markTimelineMoment records the moment a frame marks, for the frames that do
// nothing else, and reports whether it handled the frame. The caller holds o.mu.
func (o *UserBotLatency) markTimelineMoment(data processor.FramePushed) bool {
	switch f := data.Frame.(type) {
	case *frames.TranscriptionFrame:
		// A service can finalize an utterance in several pieces. The turn is not
		// detectable until the last of them, which is also where the service
		// stops its own TTFB clock, so a later one replaces the earlier until
		// the model is asked.
		if !o.seen(momentLLMRequest) {
			o.forget(momentTranscript)
			o.mark(moment{kind: momentTranscript, source: sourceName(data)}, false)
		}
	case *frames.LLMFullResponseStartFrame:
		o.llmRequest = o.mark(moment{kind: momentLLMRequest, source: sourceName(data)}, false)
	case *frames.LLMMarkerFrame:
		// A stand-alone marker holds the turn open; one that prefixes a response
		// is the completion the pipeline was waiting for.
		o.markersSeen = true
		// A turn can be held more than once before it completes, so every
		// incomplete verdict is recorded; only the completion is kept once.
		kind := momentMarkerComplete
		if f.AppendToContextImmediately {
			kind = momentMarkerIncomplete
		}
		o.mark(moment{kind: kind}, kind == momentMarkerComplete)
	case *frames.LLMTextFrame:
		o.mark(moment{kind: momentFirstText, source: sourceName(data)}, true)
	case *frames.TTSAudioRawFrame:
		o.mark(moment{kind: momentFirstAudio, source: sourceName(data)}, true)
	default:
		return false
	}
	return true
}

// sourceName is the name of the processor that pushed a frame, and "" where the
// handover named none.
func sourceName(data processor.FramePushed) string {
	if data.Source == nil {
		return ""
	}
	return data.Source.Name()
}

// forget drops every moment of a kind from the cycle. The caller holds o.mu.
func (o *UserBotLatency) forget(kind momentKind) {
	kept := o.moments[:0]
	for _, m := range o.moments {
		if m.kind != kind {
			kept = append(kept, m)
		}
	}
	o.moments = kept
}

// botStartedSpeaking closes whichever measurements were running and reports
// them. The caller holds o.mu.
func (o *UserBotLatency) botStartedSpeaking() {
	report := false

	if !o.clientConnected.IsZero() && !o.firstSpeechDone {
		o.firstSpeechDone = true
		if o.cfg.OnFirstBotSpeechLatency != nil {
			o.cfg.OnFirstBotSpeechLatency(o.now().Sub(o.clientConnected))
		}
		report = true
	}

	if !o.stopped.IsZero() {
		d := o.now().Sub(o.stopped)
		o.stopped = time.Time{}
		if o.cfg.OnLatency != nil {
			o.cfg.OnLatency(d)
		}
		report = true
	}

	if !report {
		return
	}
	if o.cfg.OnBreakdown != nil {
		contributions := o.buildContributions()
		var total time.Duration
		for _, c := range contributions {
			total += c.Duration
		}
		measuredFrom := MeasuredFromNothing
		if len(contributions) > 0 {
			measuredFrom = MeasuredFromClientConnected
			if !o.turnStart.IsZero() {
				measuredFrom = MeasuredFromUserSilence
			}
		}
		o.cfg.OnBreakdown(LatencyBreakdown{
			TTFB:            append([]TTFBBreakdown(nil), o.ttfb...),
			TextAggregation: o.textAgg,
			UserTurnStart:   o.turnStart,
			UserTurn:        o.turn,
			FunctionCalls:   append([]FunctionCallMetrics(nil), o.calls...),
			Contributions:   contributions,
			MeasuredFrom:    measuredFrom,
			Total:           total,
		})
	}
	o.resetAccumulators()
}

// handleMetrics accumulates the measurements of a MetricsFrame into the cycle
// being timed. The caller holds o.mu.
func (o *UserBotLatency) handleMetrics(f *frames.MetricsFrame) {
	// Measurements are only worth keeping while something is being timed: a
	// user-to-bot cycle, or the wait for the bot's first words.
	waitingForFirstSpeech := !o.clientConnected.IsZero() && !o.firstSpeechDone
	if o.stopped.IsZero() && !waitingForFirstSpeech {
		return
	}

	now := o.now()
	for _, d := range f.Data {
		switch m := d.(type) {
		case frames.TTFBMetricsData:
			if m.Value <= 0 {
				continue
			}
			o.ttfb = append(o.ttfb, TTFBBreakdown{
				Processor: m.Processor,
				Model:     m.Model,
				StartTime: now.Add(-m.Value),
				Duration:  m.Value,
			})
			if o.llmRequest != nil && m.Processor == o.llmRequest.source {
				// The first chunk landed before this metric was pushed, so cap
				// it there rather than letting a derived moment sort after the
				// frames that followed it.
				at := o.llmRequest.at.Add(m.Value)
				if at.After(now) {
					at = now
				}
				o.mark(moment{at: at, kind: momentLLMChunk, source: o.llmRequest.source, derived: true}, false)
			}
		case frames.TextAggregationMetricsData:
			// Only the first is kept: it is the one that held up the start of
			// the reply, and the ones after it overlap speech already playing.
			if o.textAgg == nil {
				o.textAgg = &TextAggregationBreakdown{
					Processor: m.Processor,
					StartTime: now.Add(-m.Value),
					Duration:  m.Value,
				}
			}
		}
	}
}

// resetAccumulators clears what a cycle collected. The caller holds o.mu.
func (o *UserBotLatency) resetAccumulators() {
	o.moments = nil
	o.llmRequest = nil
	o.ttfb = nil
	o.textAgg = nil
	o.turnStart = time.Time{}
	o.turn = nil
	o.callStarts = map[string]FunctionCallMetrics{}
	o.calls = nil
}

// speechStart is the moment the speech a VAD start frame reports actually began,
// which is earlier than the determination by however long the detector needed to
// be sure. A frame carrying no timestamp is taken as having just arrived, and
// fallback is what "just now" means to the caller.
func speechStart(f *frames.VADUserStartedSpeakingFrame, fallback time.Time) time.Time {
	at := f.Timestamp
	if at.IsZero() {
		at = fallback
	}
	return at.Add(-time.Duration(f.StartSecs * float64(time.Second)))
}

// speechStop is the moment the speech a VAD stop frame reports actually ended,
// which is earlier than the determination by the detector's silence window. A
// frame carrying no timestamp is taken as having just arrived, and fallback is
// what "just now" means to the caller.
func speechStop(f *frames.VADUserStoppedSpeakingFrame, fallback time.Time) time.Time {
	at := f.Timestamp
	if at.IsZero() {
		at = fallback
	}
	return at.Add(-time.Duration(f.StopSecs * float64(time.Second)))
}

// Compile-time interface checks.
var (
	_ processor.Observer                = (*UserBotLatency)(nil)
	_ processor.PipelineStartedObserver = (*UserBotLatency)(nil)
)
