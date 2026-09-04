package observers_test

import (
	"encoding/json"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// latencyRecorder collects everything a UserBotLatency observer reports.
type latencyRecorder struct {
	mu          sync.Mutex
	latencies   []time.Duration
	breakdowns  []observers.LatencyBreakdown
	firstSpeech []time.Duration
}

func (r *latencyRecorder) config() observers.LatencyConfig {
	return observers.LatencyConfig{
		OnLatency: func(d time.Duration) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.latencies = append(r.latencies, d)
		},
		OnBreakdown: func(b observers.LatencyBreakdown) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.breakdowns = append(r.breakdowns, b)
		},
		OnFirstBotSpeechLatency: func(d time.Duration) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.firstSpeech = append(r.firstSpeech, d)
		},
	}
}

// vadStopped is a VAD stop frame reporting speech that ended just now, after the
// detector's default silence window.
func vadStopped() *frames.VADUserStoppedSpeakingFrame {
	return frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now())
}

// metrics wraps measurements in the frame that carries them.
func metrics(data ...frames.MetricsData) *frames.MetricsFrame {
	return frames.NewMetricsFrame(data...)
}

// ttfb is one time-to-first-byte measurement from the named processor.
func ttfb(processorName string, d time.Duration) frames.TTFBMetricsData {
	return frames.TTFBMetricsData{
		Processor: processorName,
		Value:     d,
	}
}

// TestUserBotLatencyMeasuresTheGapToTheBot covers the headline figure: how long
// the user waited between falling silent and hearing an answer.
func TestUserBotLatencyMeasuresTheGapToTheBot(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, vadStopped(), processor.Downstream)
	time.Sleep(10 * time.Millisecond)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.latencies) != 1 {
		t.Fatalf("latencies = %v, want one", r.latencies)
	}
	if r.latencies[0] <= 0 {
		t.Errorf("latency = %s, want more than nothing", r.latencies[0])
	}
}

// TestUserBotLatencyMeasuresEveryExchange covers a conversation rather than a
// single reply: each user-to-bot cycle is measured on its own.
func TestUserBotLatencyMeasuresEveryExchange(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	for range 2 {
		push(o, vadStopped(), processor.Downstream)
		push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	}

	if len(r.latencies) != 2 {
		t.Fatalf("latencies = %v, want two", r.latencies)
	}
}

// TestUserBotLatencyMeasuresNothingWithoutTheUser covers the bot speaking of its
// own accord. Nobody was waiting, so there is no response latency to report.
func TestUserBotLatencyMeasuresNothingWithoutTheUser(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.latencies) != 0 || len(r.breakdowns) != 0 {
		t.Fatalf("latencies = %v, breakdowns = %d, want neither", r.latencies, len(r.breakdowns))
	}
}

// TestUserBotLatencyBreaksTheDelayDown covers the account of where the delay
// went: each service's own contribution, in the order it was reported.
func TestUserBotLatencyBreaksTheDelayDown(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	textAgg := frames.TextAggregationMetricsData{
		Processor: "TTS#0",
		Value:     30 * time.Millisecond,
	}

	push(o, vadStopped(), processor.Downstream)
	push(o, metrics(ttfb("STT#0", 80*time.Millisecond)), processor.Downstream)
	push(o, metrics(ttfb("LLM#0", 250*time.Millisecond), textAgg), processor.Downstream)
	push(o, metrics(ttfb("TTS#0", 70*time.Millisecond)), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	b := r.breakdowns[0]
	var named []string
	for _, m := range b.TTFB {
		named = append(named, m.Processor)
	}
	if want := []string{"STT#0", "LLM#0", "TTS#0"}; !reflect.DeepEqual(named, want) {
		t.Errorf("TTFB = %v, want %v in the order they were reported", named, want)
	}
	if b.TextAggregation == nil {
		t.Fatal("TextAggregation is unset, want the measurement that was reported")
	}
	if b.TextAggregation.Duration != 30*time.Millisecond {
		t.Errorf("text aggregation = %s, want 30ms", b.TextAggregation.Duration)
	}
}

// TestUserBotLatencyDropsMeasurementsOfAnInterruptedReply covers a reply cut
// short. Its measurements describe work nobody heard, so they are dropped rather
// than charged to the reply that follows.
func TestUserBotLatencyDropsMeasurementsOfAnInterruptedReply(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, vadStopped(), processor.Downstream)
	push(o, metrics(ttfb("LLM#0", 245*time.Millisecond)), processor.Downstream)
	push(o, frames.NewInterruptionFrame(), processor.Downstream)
	push(o, metrics(ttfb("LLM#0", 224*time.Millisecond)), processor.Downstream)
	push(o, metrics(ttfb("TTS#0", 142*time.Millisecond)), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	b := r.breakdowns[0]
	if len(b.TTFB) != 2 {
		t.Fatalf("TTFB = %+v, want only the two measured after the interruption", b.TTFB)
	}
	if b.TTFB[0].Duration != 224*time.Millisecond || b.TTFB[1].Duration != 142*time.Millisecond {
		t.Errorf("TTFB = %+v, want the post-interruption measurements", b.TTFB)
	}
}

// TestUserBotLatencyKeepsTheFirstTextAggregation covers repeated aggregation
// measurements in one reply. Only the first held up the start of the speech; the
// ones after it overlap audio already playing.
func TestUserBotLatencyKeepsTheFirstTextAggregation(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	agg := func(d time.Duration) frames.TextAggregationMetricsData {
		return frames.TextAggregationMetricsData{
			Processor: "TTS#0",
			Value:     d,
		}
	}

	push(o, vadStopped(), processor.Downstream)
	push(o, metrics(agg(30*time.Millisecond)), processor.Downstream)
	push(o, metrics(agg(80*time.Millisecond)), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	if r.breakdowns[0].TextAggregation == nil {
		t.Fatal("TextAggregation is unset, want the first measurement")
	}
	if got := r.breakdowns[0].TextAggregation.Duration; got != 30*time.Millisecond {
		t.Errorf("text aggregation = %s, want the first measurement of 30ms", got)
	}
}

// TestUserBotLatencyMeasuresTheUserTurn covers what happened before the model
// was asked anything: the detector's silence window, the transcriber finalizing,
// and any wait on an end-of-turn analyzer.
func TestUserBotLatencyMeasuresTheUserTurn(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, vadStopped(), processor.Downstream)
	time.Sleep(20 * time.Millisecond)
	push(o, frames.NewUserStoppedSpeakingFrame(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	b := r.breakdowns[0]
	if b.UserTurn == nil {
		t.Fatal("UserTurn is unset, want the time the turn took to be released")
	}
	if *b.UserTurn < 20*time.Millisecond {
		t.Errorf("UserTurn = %s, want at least the 20ms the release took", *b.UserTurn)
	}
	if b.UserTurnStart.IsZero() {
		t.Error("UserTurnStart is unset, want the moment the speech ended")
	}
}

// TestUserBotLatencyReportsNoUserTurnWithoutARelease covers a pipeline with no
// turn analyzer, where the turn is never released as its own event. There is
// nothing to measure, which is distinct from measuring nothing.
func TestUserBotLatencyReportsNoUserTurnWithoutARelease(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, vadStopped(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	if r.breakdowns[0].UserTurn != nil {
		t.Errorf("UserTurn = %s, want none: the turn was never released", *r.breakdowns[0].UserTurn)
	}
}

// TestUserBotLatencyMeasuresTheGreeting covers the other latency worth knowing:
// how long after a client connected the bot first said anything.
func TestUserBotLatencyMeasuresTheGreeting(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, frames.NewClientConnectedFrame(), processor.Downstream)
	push(o, metrics(ttfb("LLM#0", 250*time.Millisecond)), processor.Downstream)
	push(o, metrics(ttfb("TTS#0", 70*time.Millisecond)), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.firstSpeech) != 1 {
		t.Fatalf("first-speech latencies = %v, want one", r.firstSpeech)
	}
	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one alongside the latency", len(r.breakdowns))
	}
	if len(r.breakdowns[0].TTFB) != 2 {
		t.Errorf("TTFB = %+v, want the two measured while waiting for the greeting", r.breakdowns[0].TTFB)
	}
}

// TestUserBotLatencyMeasuresTheGreetingOnce covers the rest of the call: the
// figure describes the greeting, and there is only ever one of those.
func TestUserBotLatencyMeasuresTheGreetingOnce(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, frames.NewClientConnectedFrame(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	push(o, vadStopped(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.firstSpeech) != 1 {
		t.Fatalf("first-speech latencies = %v, want one", r.firstSpeech)
	}
}

// TestUserBotLatencySkipsTheGreetingWhenTheUserSpeaksFirst covers a caller who
// talks before the bot does. There was no greeting, so timing one would report a
// figure for something that never happened.
func TestUserBotLatencySkipsTheGreetingWhenTheUserSpeaksFirst(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, frames.NewClientConnectedFrame(), processor.Downstream)
	push(o, frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()), processor.Downstream)
	push(o, vadStopped(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.firstSpeech) != 0 {
		t.Fatalf("first-speech latencies = %v, want none: the user spoke first", r.firstSpeech)
	}
}

// TestUserBotLatencyTimesToolCalls covers a reply that had to go and fetch
// something. The call is often the largest single part of the delay, so it is
// accounted for by name.
func TestUserBotLatencyTimesToolCalls(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	args := json.RawMessage(`{"location":"Atlanta"}`)
	push(o, vadStopped(), processor.Downstream)
	push(o, frames.NewFunctionCallInProgressFrame("call_1", "get_weather", args, true, ""), processor.Downstream)
	time.Sleep(20 * time.Millisecond)
	push(o, frames.NewFunctionCallResultFrame("call_1", "get_weather", args, "75"), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	calls := r.breakdowns[0].FunctionCalls
	if len(calls) != 1 {
		t.Fatalf("function calls = %+v, want one", calls)
	}
	if calls[0].FunctionName != "get_weather" {
		t.Errorf("function name = %q, want get_weather", calls[0].FunctionName)
	}
	if calls[0].Duration < 20*time.Millisecond {
		t.Errorf("call duration = %s, want at least the 20ms it took", calls[0].Duration)
	}
}

// TestUserBotLatencyDropsToolCallsOfAnInterruptedReply covers the tool calls of
// a reply that was cut short, which are dropped with the rest of its account.
func TestUserBotLatencyDropsToolCallsOfAnInterruptedReply(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	args := json.RawMessage(`{}`)
	push(o, vadStopped(), processor.Downstream)
	push(o, frames.NewFunctionCallInProgressFrame("call_1", "get_weather", args, true, ""), processor.Downstream)
	push(o, frames.NewFunctionCallResultFrame("call_1", "get_weather", args, ""), processor.Downstream)
	push(o, frames.NewInterruptionFrame(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)

	if len(r.breakdowns) != 1 {
		t.Fatalf("breakdowns = %d, want one", len(r.breakdowns))
	}
	if calls := r.breakdowns[0].FunctionCalls; len(calls) != 0 {
		t.Errorf("function calls = %+v, want none: they belonged to the interrupted reply", calls)
	}
}

// TestUserBotLatencyIgnoresUpstreamHandovers covers a frame broadcast in both
// directions. Counting both halves would measure every cycle twice.
func TestUserBotLatencyIgnoresUpstreamHandovers(t *testing.T) {
	var r latencyRecorder
	o := observers.NewUserBotLatency(r.config())

	push(o, vadStopped(), processor.Downstream)
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Upstream)

	if len(r.latencies) != 0 {
		t.Fatalf("latencies = %v, want none: only the downstream half is counted", r.latencies)
	}
}

// TestLatencyBreakdownOrdersItsEventsInTime covers the rendering: the point of
// the breakdown is reading a slow reply as a sequence, so the lines come out in
// the order the measurements started rather than the order they were reported.
func TestLatencyBreakdownOrdersItsEventsInTime(t *testing.T) {
	base := time.Unix(100, 0)
	at := base.Add
	userTurn := 150 * time.Millisecond

	b := observers.LatencyBreakdown{
		UserTurnStart: base,
		UserTurn:      &userTurn,
		TTFB: []observers.TTFBBreakdown{
			{Processor: "LLM#0", Model: "gpt-4o", StartTime: at(200 * time.Millisecond), Duration: 250 * time.Millisecond},
			{Processor: "STT#0", StartTime: at(50 * time.Millisecond), Duration: 80 * time.Millisecond},
			{Processor: "TTS#0", StartTime: at(500 * time.Millisecond), Duration: 70 * time.Millisecond},
		},
		FunctionCalls: []observers.FunctionCallMetrics{
			{FunctionName: "get_weather", StartTime: at(450 * time.Millisecond), Duration: 120 * time.Millisecond},
		},
		TextAggregation: &observers.TextAggregationBreakdown{
			Processor: "TTS#0", StartTime: at(480 * time.Millisecond), Duration: 30 * time.Millisecond,
		},
	}

	want := []string{
		"User turn: 0.150s",
		"STT#0: TTFB 0.080s",
		"LLM#0: TTFB 0.250s",
		"get_weather: 0.120s",
		"TTS#0: text aggregation 0.030s",
		"TTS#0: TTFB 0.070s",
	}
	if got := b.ChronologicalEvents(); !reflect.DeepEqual(got, want) {
		t.Errorf("events =\n%v\nwant\n%v", got, want)
	}
}

// TestLatencyBreakdownRendersNothingWhenEmpty covers a cycle that measured
// nothing, which is what a pipeline that does not collect metrics reports.
func TestLatencyBreakdownRendersNothingWhenEmpty(t *testing.T) {
	if got := (observers.LatencyBreakdown{}).ChronologicalEvents(); len(got) != 0 {
		t.Errorf("events = %v, want none", got)
	}
}

// TestLatencyBreakdownNeedsBothHalvesOfTheUserTurn covers a half-measured user
// turn, which cannot be placed on the timeline and so is left off it.
func TestLatencyBreakdownNeedsBothHalvesOfTheUserTurn(t *testing.T) {
	d := 150 * time.Millisecond
	if got := (observers.LatencyBreakdown{UserTurnStart: time.Unix(100, 0)}).ChronologicalEvents(); len(got) != 0 {
		t.Errorf("events = %v, want none: the turn has a start but no duration", got)
	}
	if got := (observers.LatencyBreakdown{UserTurn: &d}).ChronologicalEvents(); len(got) != 0 {
		t.Errorf("events = %v, want none: the turn has a duration but no start", got)
	}
}
