package observers

import (
	"fmt"
	"sort"
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
}

// ChronologicalEvents renders every measurement in the breakdown as a line of
// text, ordered by when it started. It is what turns the breakdown into a log of
// where a slow reply spent its time.
func (b LatencyBreakdown) ChronologicalEvents() []string {
	type event struct {
		at    time.Time
		label string
	}
	var events []event

	if !b.UserTurnStart.IsZero() && b.UserTurn != nil {
		events = append(events, event{b.UserTurnStart, fmt.Sprintf("User turn: %.3fs", b.UserTurn.Seconds())})
	}
	for _, t := range b.TTFB {
		events = append(events, event{t.StartTime, fmt.Sprintf("%s: TTFB %.3fs", t.Processor, t.Duration.Seconds())})
	}
	for _, fc := range b.FunctionCalls {
		events = append(events, event{fc.StartTime, fmt.Sprintf("%s: %.3fs", fc.FunctionName, fc.Duration.Seconds())})
	}
	if ta := b.TextAggregation; ta != nil {
		events = append(events, event{
			ta.StartTime,
			fmt.Sprintf("%s: text aggregation %.3fs", ta.Processor, ta.Duration.Seconds()),
		})
	}

	sort.SliceStable(events, func(i, j int) bool { return events[i].at.Before(events[j].at) })

	labels := make([]string, 0, len(events))
	for _, e := range events {
		labels = append(labels, e.label)
	}
	return labels
}

// LatencyConfig configures a UserBotLatency observer.
type LatencyConfig struct {
	// MaxFrames is how many recent frame ids the observer remembers to
	// recognize one it has already counted; 0 uses 100.
	MaxFrames int
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
	dd      deduper
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

	// Per-cycle accumulators, cleared whenever a cycle begins or is abandoned.
	ttfb       []TTFBBreakdown
	textAgg    *TextAggregationBreakdown
	callStarts map[string]FunctionCallMetrics
	calls      []FunctionCallMetrics
}

// NewUserBotLatency builds a UserBotLatency observer.
func NewUserBotLatency(cfg LatencyConfig) *UserBotLatency {
	o := &UserBotLatency{cfg: cfg, dd: newDeduper(cfg.MaxFrames)}
	o.resetAccumulators()
	return o
}

// OnPushFrame implements processor.Observer.
func (o *UserBotLatency) OnPushFrame(data processor.FramePushed) {
	if data.Direction != processor.Downstream {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.dd.seenBefore(data.Frame.ID()) {
		return
	}

	switch f := data.Frame.(type) {
	case *frames.ClientConnectedFrame:
		if o.clientConnected.IsZero() {
			o.clientConnected = time.Now()
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
		o.stopped = speechStop(f)
		o.turnStart = o.stopped
	case *frames.UserStoppedSpeakingFrame:
		if !o.stopped.IsZero() {
			d := time.Since(o.stopped)
			o.turn = &d
		}
	case *frames.InterruptionFrame:
		// The measurements of a cycle that was cut short describe work nobody
		// heard, so they are dropped rather than charged to the next reply.
		o.resetAccumulators()
	case *frames.FunctionCallInProgressFrame:
		o.callStarts[f.ToolCallID] = FunctionCallMetrics{
			FunctionName: f.ToolName,
			StartTime:    time.Now(),
		}
	case *frames.FunctionCallResultFrame:
		if call, ok := o.callStarts[f.ToolCallID]; ok {
			delete(o.callStarts, f.ToolCallID)
			call.Duration = time.Since(call.StartTime)
			o.calls = append(o.calls, call)
		}
	case *frames.MetricsFrame:
		o.handleMetrics(f)
	case *frames.BotStartedSpeakingFrame:
		o.botStartedSpeaking()
	}
}

// botStartedSpeaking closes whichever measurements were running and reports
// them. The caller holds o.mu.
func (o *UserBotLatency) botStartedSpeaking() {
	report := false

	if !o.clientConnected.IsZero() && !o.firstSpeechDone {
		o.firstSpeechDone = true
		if o.cfg.OnFirstBotSpeechLatency != nil {
			o.cfg.OnFirstBotSpeechLatency(time.Since(o.clientConnected))
		}
		report = true
	}

	if !o.stopped.IsZero() {
		d := time.Since(o.stopped)
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
		o.cfg.OnBreakdown(LatencyBreakdown{
			TTFB:            append([]TTFBBreakdown(nil), o.ttfb...),
			TextAggregation: o.textAgg,
			UserTurnStart:   o.turnStart,
			UserTurn:        o.turn,
			FunctionCalls:   append([]FunctionCallMetrics(nil), o.calls...),
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

	now := time.Now()
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
	o.ttfb = nil
	o.textAgg = nil
	o.turnStart = time.Time{}
	o.turn = nil
	o.callStarts = map[string]FunctionCallMetrics{}
	o.calls = nil
}

// speechStop is the moment the speech a VAD stop frame reports actually ended,
// which is earlier than the determination by the detector's silence window. A
// frame carrying no timestamp is taken as having just arrived.
func speechStop(f *frames.VADUserStoppedSpeakingFrame) time.Time {
	at := f.Timestamp
	if at.IsZero() {
		at = time.Now()
	}
	return at.Add(-time.Duration(f.StopSecs * float64(time.Second)))
}

// Compile-time interface check.
var _ processor.Observer = (*UserBotLatency)(nil)
