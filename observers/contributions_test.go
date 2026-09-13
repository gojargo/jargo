package observers_test

import (
	"encoding/json"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// cycleDriver drives an observer through a cycle on a clock the test controls,
// so a cycle can describe intervals of any length without waiting them out.
type cycleDriver struct {
	t *testing.T

	mu         sync.Mutex
	clock      time.Time
	breakdowns []observers.LatencyBreakdown

	// realClock leaves the observer on the wall clock, so the hops a frozen
	// clock puts at the same instant are microseconds apart instead. It is what
	// a test needs to see the briefest stretches at all: a stretch of exactly no
	// duration is not part of the timeline whatever the threshold says.
	realClock bool

	o *observers.UserBotLatency
	// sources are the processors pushing frames, kept so each name keeps one
	// identity: a metric names its processor the same way a handover does, and
	// the observer matches the two.
	sources map[string]processor.Processor

	silenceAt, spokeAt time.Time
}

// newCycleDriver starts an observer reading the driver's clock. A zero
// minContribution leaves the default threshold in place.
func newCycleDriver(t *testing.T, minContribution time.Duration) *cycleDriver {
	t.Helper()
	return newDriver(t, minContribution, false)
}

// newRealClockDriver starts one on the wall clock instead.
func newRealClockDriver(t *testing.T, minContribution time.Duration) *cycleDriver {
	t.Helper()
	return newDriver(t, minContribution, true)
}

func newDriver(t *testing.T, minContribution time.Duration, realClock bool) *cycleDriver {
	t.Helper()
	d := &cycleDriver{
		t:         t,
		clock:     time.Date(2026, time.September, 13, 12, 0, 0, 0, time.UTC),
		realClock: realClock,
		sources:   map[string]processor.Processor{},
	}
	// The wall clock is what a nil Now means, so a real-clock driver simply
	// does not supply one.
	var now func() time.Time
	if !realClock {
		now = d.now
	}
	d.o = observers.NewUserBotLatency(observers.LatencyConfig{
		Now:             now,
		MinContribution: minContribution,
		OnBreakdown: func(b observers.LatencyBreakdown) {
			d.mu.Lock()
			defer d.mu.Unlock()
			d.breakdowns = append(d.breakdowns, b)
		},
	})
	return d
}

func (d *cycleDriver) now() time.Time {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.realClock {
		return time.Now()
	}
	return d.clock
}

// wait advances the clock without sleeping.
func (d *cycleDriver) wait(dur time.Duration) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.clock = d.clock.Add(dur)
}

// source is the processor of a given name, built once so its identity is stable.
func (d *cycleDriver) source(name string) processor.Processor {
	if p, ok := d.sources[name]; ok {
		return p
	}
	p := processor.NewIdentityFilter(name)
	d.sources[name] = p
	return p
}

// name is how a processor of this name reports itself, which is what a metric
// has to say to be matched to it.
func (d *cycleDriver) name(source string) string { return d.source(source).Name() }

// push feeds one frame to the observer, as a pipeline handover would.
func (d *cycleDriver) push(f frames.Frame, source string) {
	d.o.OnPushFrame(processor.FramePushed{
		Frame:     f,
		Source:    d.source(source),
		Direction: processor.Downstream,
	})
}

// ttfb is the metrics frame a service reports its time to first byte in.
func (d *cycleDriver) ttfb(source string, value time.Duration) *frames.MetricsFrame {
	return frames.NewMetricsFrame(frames.TTFBMetricsData{
		Processor: d.name(source),
		Value:     value,
	})
}

// last is the breakdown most recently reported.
func (d *cycleDriver) last() observers.LatencyBreakdown {
	d.t.Helper()
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.breakdowns) == 0 {
		d.t.Fatal("no breakdown was reported")
	}
	return d.breakdowns[len(d.breakdowns)-1]
}

// turnOpts shapes the cycle completeTurn drives.
type turnOpts struct {
	// marker reports that the response carried a turn-completion marker.
	marker bool
	// hold holds the turn open once before the response that completes it.
	hold    bool
	holdFor time.Duration
	// detect is how long the turn strategy deliberates after the transcript.
	detect time.Duration
	// aggregation is what grouping text into a sentence cost.
	aggregation time.Duration
	// buffered is time spent waiting for a speakable token.
	buffered time.Duration
}

// aTurn is the cycle the tests vary from: a response the model completed.
func aTurn() turnOpts { return turnOpts{marker: true, holdFor: 60 * time.Millisecond} }

// completeTurn drives one whole user-to-bot cycle and returns its breakdown.
func (d *cycleDriver) completeTurn(opts turnOpts) observers.LatencyBreakdown {
	d.t.Helper()

	// The cycle is measured from the silence the detector waited out.
	d.silenceAt = d.now().Add(-20 * time.Millisecond)
	d.push(frames.NewVADUserStoppedSpeakingFrame(0.02, d.now()), "VAD#0")
	d.wait(20 * time.Millisecond)
	d.push(frames.NewTranscriptionFrame("hi", "u", ""), "STT#0")
	if opts.detect > 0 {
		d.wait(opts.detect)
	}
	d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")

	if opts.hold {
		d.wait(10 * time.Millisecond)
		d.push(d.ttfb("LLM#0", 10*time.Millisecond), "LLM#0")
		d.push(frames.NewLLMMarkerFrame("hold"), "LLM#0")
		d.wait(opts.holdFor)
		d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
	}

	d.wait(20 * time.Millisecond)
	d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")

	if opts.marker {
		complete := frames.NewLLMMarkerFrame("done")
		complete.AppendToContextImmediately = false
		d.push(complete, "LLM#0")
		d.wait(20 * time.Millisecond)
	}
	if opts.buffered > 0 {
		d.wait(opts.buffered)
	}

	d.push(frames.NewLLMTextFrame("Hi!"), "LLM#0")
	if opts.aggregation > 0 {
		d.wait(opts.aggregation)
		d.push(frames.NewMetricsFrame(frames.TextAggregationMetricsData{
			Processor: d.name("TTS#0"),
			Value:     opts.aggregation,
		}), "TTS#0")
	}

	d.wait(20 * time.Millisecond)
	d.push(frames.NewTTSAudioRawFrame(nil, 24000, 1), "TTS#0")
	d.push(frames.NewBotStartedSpeakingFrame(), "Transport#0")
	d.spokeAt = d.now()
	return d.last()
}

// labels is the sequence of stages a breakdown names.
func labels(b observers.LatencyBreakdown) []string {
	out := make([]string, 0, len(b.Contributions))
	for _, c := range b.Contributions {
		out = append(out, c.Label)
	}
	return out
}

// labeled is the first contribution with a given label.
func labeled(t *testing.T, b observers.LatencyBreakdown, label string) observers.LatencyContribution {
	t.Helper()
	for _, c := range b.Contributions {
		if c.Label == label {
			return c
		}
	}
	t.Fatalf("no %q contribution, got %v", label, labels(b))
	return observers.LatencyContribution{}
}

// hasLabel reports whether a breakdown names a stage at all.
func hasLabel(b observers.LatencyBreakdown, label string) bool {
	for _, c := range b.Contributions {
		if c.Label == label {
			return true
		}
	}
	return false
}

// near reports whether two durations are within tolerance of each other.
func near(got, want, tolerance time.Duration) bool {
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	return diff <= tolerance
}

// TestContributionsSumToTheMeasuredLatency covers the invariant the whole
// timeline rests on: nothing in the interval goes unnamed.
func TestContributionsSumToTheMeasuredLatency(t *testing.T) {
	d := newCycleDriver(t, 0)
	b := d.completeTurn(aTurn())

	var total time.Duration
	for _, c := range b.Contributions {
		total += c.Duration
	}
	if want := d.spokeAt.Sub(d.silenceAt); total != want {
		t.Errorf("contributions sum to %s, want the measured %s", total, want)
	}
	if b.Total != total {
		t.Errorf("Total = %s, want the %s the contributions sum to", b.Total, total)
	}
	if b.MeasuredFrom != observers.MeasuredFromUserSilence {
		t.Errorf("measured from %q, want user silence", b.MeasuredFrom)
	}
}

// TestCompleteTurnNamesEachStage covers a turn the model completed: it has no
// wait, and lists the rest in the order they happened.
func TestCompleteTurnNamesEachStage(t *testing.T) {
	d := newCycleDriver(t, 0)
	b := d.completeTurn(aTurn())

	want := []string{
		"endpointing wait",
		"transcription",
		"LLM inference",
		"turn completion",
		"speech synthesis",
	}
	if got := labels(b); strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("stages =\n%v\nwant\n%v", got, want)
	}

	endpointing := b.Contributions[0]
	if endpointing.Owner != "config: VAD StopSecs" {
		t.Errorf("endpointing owner = %q, want the VAD setting", endpointing.Owner)
	}
	if endpointing.OwnerKind != observers.OwnerSetting {
		t.Errorf("endpointing owner kind = %q, want setting", endpointing.OwnerKind)
	}
	if !near(endpointing.Duration, 20*time.Millisecond, 10*time.Millisecond) {
		t.Errorf("endpointing wait = %s, want about 20ms", endpointing.Duration)
	}
}

// TestHeldTurnNamesTheWait covers an incomplete marker showing up as a wait
// rather than as slow inference.
func TestHeldTurnNamesTheWait(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.hold = true
	b := d.completeTurn(opts)

	wait := labeled(t, b, "waiting for user")
	if wait.Owner != "config: filter_incomplete_user_turns" {
		t.Errorf("wait owner = %q, want the turn-completion setting", wait.Owner)
	}
	if wait.Duration <= 50*time.Millisecond {
		t.Errorf("wait = %s, want more than 50ms", wait.Duration)
	}

	// Both inferences are listed, so neither absorbs the wait.
	inferences := 0
	for _, c := range b.Contributions {
		if c.Label == "LLM inference" {
			inferences++
		}
	}
	if inferences != 2 {
		t.Errorf("LLM inference entries = %d, want 2: the wait sits between them", inferences)
	}
}

// TestBotWithoutTurnCompletionHasNoMarkerEntry covers parts that did not happen
// not being listed.
func TestBotWithoutTurnCompletionHasNoMarkerEntry(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.marker = false
	b := d.completeTurn(opts)

	for _, absent := range []string{"turn completion", "waiting for user", "awaiting speakable text"} {
		if hasLabel(b, absent) {
			t.Errorf("%q is listed, want it absent: this bot does not gate its replies", absent)
		}
	}
}

// TestMissingMarkerReportedOnceMarkersAreInUse covers a bot that emits markers,
// on a response that carried none: the wait for a speakable token is then a
// diagnosis rather than the model simply still streaming.
func TestMissingMarkerReportedOnceMarkersAreInUse(t *testing.T) {
	d := newCycleDriver(t, 0)
	d.completeTurn(aTurn()) // establishes that this bot uses markers

	opts := aTurn()
	opts.marker = false
	opts.buffered = 50 * time.Millisecond
	b := d.completeTurn(opts)

	buffered := labeled(t, b, "awaiting speakable text")
	if buffered.Duration <= 40*time.Millisecond {
		t.Errorf("awaiting speakable text = %s, want more than 40ms", buffered.Duration)
	}
}

// TestTurnContributionLinesOrderAndTotal covers the rendering: chronological by
// default, largest first by cost, always with a total.
func TestTurnContributionLinesOrderAndTotal(t *testing.T) {
	b := observers.LatencyBreakdown{
		Contributions: []observers.LatencyContribution{
			{
				Key: "endpointing_wait", Label: "endpointing wait",
				Owner: "config: VAD StopSecs", OwnerKind: observers.OwnerSetting,
				Duration: 200 * time.Millisecond,
			},
			{
				Key: "speech_synthesis", Label: "speech synthesis",
				Owner: "TTS#0", OwnerKind: observers.OwnerService,
				Duration: 400 * time.Millisecond,
			},
		},
	}

	chronological := b.TurnContributionLines(false)
	if !strings.Contains(chronological[0], "endpointing wait") {
		t.Errorf("first line = %q, want the earliest stage", chronological[0])
	}
	byCost := b.TurnContributionLines(true)
	if !strings.Contains(byCost[0], "speech synthesis") {
		t.Errorf("first line by cost = %q, want the most expensive stage", byCost[0])
	}
	if last := chronological[len(chronological)-1]; !strings.Contains(last, " 0.600s  TOTAL") {
		t.Errorf("last line = %q, want the total", last)
	}
}

// TestTurnContributionLinesRenderNothingWhenEmpty covers a cycle that named
// nothing, which is what a pipeline that does not collect metrics reports.
func TestTurnContributionLinesRenderNothingWhenEmpty(t *testing.T) {
	if got := (observers.LatencyBreakdown{}).TurnContributionLines(false); len(got) != 0 {
		t.Errorf("lines = %v, want none", got)
	}
}

// TestTurnCompletionCoversTheGateAndTheMarkerToken covers the span running from
// the model's first chunk to the first speakable token, which is the gate
// reading the verdict plus the token the marker occupies.
func TestTurnCompletionCoversTheGateAndTheMarkerToken(t *testing.T) {
	d := newCycleDriver(t, 0)
	b := d.completeTurn(aTurn())

	completion := labeled(t, b, "turn completion")
	if completion.Owner != "config: filter_incomplete_user_turns" {
		t.Errorf("owner = %q, want the turn-completion setting", completion.Owner)
	}
	if completion.Duration <= 15*time.Millisecond {
		t.Errorf("turn completion = %s, want more than 15ms", completion.Duration)
	}
}

// TestEveryContributionListedWhenTheThresholdIsOff covers nothing being folded
// away when the threshold is lifted.
func TestEveryContributionListedWhenTheThresholdIsOff(t *testing.T) {
	d := newRealClockDriver(t, -1)
	b := d.completeTurn(aTurn())

	if !hasLabel(b, "output transport") {
		t.Errorf("stages = %v, want the brief ones listed too", labels(b))
	}
	for _, c := range b.Contributions {
		if c.Label == "pipeline" && c.Duration >= 10*time.Millisecond {
			t.Errorf("pipeline = %s, want nothing left over once every span is named", c.Duration)
		}
	}
}

// TestCoreStagesAreListedWhenBriefButMeasurable covers the rule that keeps a
// timeline's shape: a core stage of a few milliseconds is listed, and one that
// rounds to zero is not.
func TestCoreStagesAreListedWhenBriefButMeasurable(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.detect = 2 * time.Millisecond
	b := d.completeTurn(opts)

	detection := labeled(t, b, "turn detection")
	if detection.Duration >= 5*time.Millisecond {
		t.Errorf("turn detection = %s, want under the threshold", detection.Duration)
	}
	if detection.Owner != "config: user turn strategies" {
		t.Errorf("owner = %q, want the turn strategies setting", detection.Owner)
	}

	if b := d.completeTurn(aTurn()); hasLabel(b, "turn detection") {
		t.Error("turn detection is listed, want it absent: it took no measurable time")
	}
}

// TestSentenceAggregationListedOnlyWhenItWaits covers the wait before speech can
// start: grouping into sentences reports it, streaming tokens has none.
func TestSentenceAggregationListedOnlyWhenItWaits(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.aggregation = 50 * time.Millisecond
	b := d.completeTurn(opts)

	aggregation := labeled(t, b, "sentence aggregation")
	if !near(aggregation.Duration, 50*time.Millisecond, 10*time.Millisecond) {
		t.Errorf("sentence aggregation = %s, want about 50ms", aggregation.Duration)
	}
	if aggregation.Owner != "config: text aggregator" {
		t.Errorf("owner = %q, want the text aggregator setting", aggregation.Owner)
	}
	synthesis := labeled(t, b, "speech synthesis")
	if !synthesis.StartTime.After(aggregation.StartTime) {
		t.Error("synthesis starts before aggregation, want it after: the wait comes first")
	}

	if b := d.completeTurn(aTurn()); hasLabel(b, "sentence aggregation") {
		t.Error("sentence aggregation is listed, want it absent: nothing waited")
	}
}

// toolCall pushes the frames of one tool call.
func (d *cycleDriver) toolCall(name, id string) (*frames.FunctionCallInProgressFrame, *frames.FunctionCallResultFrame) {
	args := json.RawMessage(`{}`)
	return frames.NewFunctionCallInProgressFrame(id, name, args, false, "group"),
		frames.NewFunctionCallResultFrame(id, name, args, "{}")
}

// TestFunctionHandlersListedWithTheWaitThatPrecedesThem covers a tool turn
// naming both writing the call and running the handler.
func TestFunctionHandlersListedWithTheWaitThatPrecedesThem(t *testing.T) {
	d := newCycleDriver(t, 0)

	d.push(frames.NewVADUserStoppedSpeakingFrame(0.02, d.now()), "VAD#0")
	d.push(frames.NewTranscriptionFrame("weather?", "u", ""), "STT#0")
	d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")
	d.wait(20 * time.Millisecond)

	inProgress, result := d.toolCall("get_weather", "1")
	d.push(inProgress, "LLM#0")
	d.wait(30 * time.Millisecond)
	d.push(result, "LLM#0")

	d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")
	d.push(frames.NewLLMTextFrame("Nice out."), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(frames.NewTTSAudioRawFrame(nil, 24000, 1), "TTS#0")
	d.push(frames.NewBotStartedSpeakingFrame(), "Transport#0")

	b := d.last()
	call := labeled(t, b, "function handler")
	if call.Owner != "get_weather" {
		t.Errorf("handler owner = %q, want the function's name", call.Owner)
	}
	if !near(call.Duration, 30*time.Millisecond, 20*time.Millisecond) {
		t.Errorf("handler = %s, want about 30ms", call.Duration)
	}

	writing := labeled(t, b, "LLM tool call")
	if writing.Owner != d.name("LLM#0") {
		t.Errorf("tool call owner = %q, want %q", writing.Owner, d.name("LLM#0"))
	}
	if !writing.StartTime.Before(call.StartTime) {
		t.Error("writing the call starts after running it, want before")
	}
	if hasLabel(b, "pipeline") {
		t.Errorf("stages = %v, want nothing left over once the tool spans are named", labels(b))
	}
}

// TestConcurrentHandlersAreOneSpan covers handlers dispatched together counting
// their wall clock once: listing each would count the same stretch twice and
// push the sum past the latency it describes.
func TestConcurrentHandlersAreOneSpan(t *testing.T) {
	d := newCycleDriver(t, 0)

	d.push(frames.NewVADUserStoppedSpeakingFrame(0.02, d.now()), "VAD#0")
	d.push(frames.NewTranscriptionFrame("weather and food?", "u", ""), "STT#0")
	d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")

	weatherStart, weatherResult := d.toolCall("get_weather", "1")
	foodStart, foodResult := d.toolCall("get_restaurant", "2")
	d.push(weatherStart, "LLM#0")
	d.push(foodStart, "LLM#0")
	d.wait(30 * time.Millisecond)
	d.push(foodResult, "LLM#0")
	d.wait(50 * time.Millisecond)
	d.push(weatherResult, "LLM#0")

	d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")
	d.push(frames.NewLLMTextFrame("Here you go."), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(frames.NewTTSAudioRawFrame(nil, 24000, 1), "TTS#0")
	d.push(frames.NewBotStartedSpeakingFrame(), "Transport#0")

	b := d.last()
	var handlers []observers.LatencyContribution
	for _, c := range b.Contributions {
		if c.Label == "function handler" {
			handlers = append(handlers, c)
		}
	}
	if len(handlers) != 1 {
		t.Fatalf("function handler entries = %d, want 1: they ran together", len(handlers))
	}
	// Named for the handler the wait actually depended on.
	if handlers[0].Owner != "get_weather +1" {
		t.Errorf("owner = %q, want the slowest handler and a count of the rest", handlers[0].Owner)
	}
	if !near(handlers[0].Duration, 80*time.Millisecond, 20*time.Millisecond) {
		t.Errorf("handlers = %s, want about 80ms", handlers[0].Duration)
	}
}

// TestNoTwoContributionsOverlap covers spans tiling the interval rather than
// covering the same time twice.
func TestNoTwoContributionsOverlap(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.hold = true
	opts.aggregation = 30 * time.Millisecond
	b := d.completeTurn(opts)

	ordered := append([]observers.LatencyContribution(nil), b.Contributions...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].StartTime.Before(ordered[j].StartTime)
	})
	for i := 0; i+1 < len(ordered); i++ {
		prev, next := ordered[i], ordered[i+1]
		// The residual is spread across the cycle rather than sitting anywhere
		// in it, so it is not part of the tiling.
		if prev.Label == "pipeline" || next.Label == "pipeline" {
			continue
		}
		if prev.StartTime.Add(prev.Duration).After(next.StartTime) {
			t.Errorf("%q runs past the start of %q", prev.Label, next.Label)
		}
	}
}

// TestLongHoldReportedWithoutWaitingItOut covers the clock being the test's: a
// ten second hold costs the suite nothing.
func TestLongHoldReportedWithoutWaitingItOut(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.hold = true
	opts.holdFor = 10 * time.Second
	b := d.completeTurn(opts)

	if got := labeled(t, b, "waiting for user").Duration; got != 10*time.Second {
		t.Errorf("wait = %s, want exactly 10s", got)
	}
}

// TestDurationsAreExact covers every span being the interval the cycle
// described, rather than whatever a wall clock left around it.
func TestDurationsAreExact(t *testing.T) {
	d := newCycleDriver(t, 0)
	opts := aTurn()
	opts.detect = 400 * time.Millisecond
	opts.aggregation = 250 * time.Millisecond
	b := d.completeTurn(opts)

	for _, want := range []struct {
		label string
		dur   time.Duration
	}{
		{"endpointing wait", 20 * time.Millisecond},
		{"turn detection", 400 * time.Millisecond},
		{"sentence aggregation", 250 * time.Millisecond},
	} {
		if got := labeled(t, b, want.label).Duration; got != want.dur {
			t.Errorf("%s = %s, want exactly %s", want.label, got, want.dur)
		}
	}
	if hasLabel(b, "pipeline") {
		t.Errorf("stages = %v, want nothing unaccounted for", labels(b))
	}
}

// TestTurnHeldTwiceReportsBothWaits covers a turn that can be told to wait more
// than once before it completes.
func TestTurnHeldTwiceReportsBothWaits(t *testing.T) {
	d := newCycleDriver(t, 0)

	d.push(frames.NewVADUserStoppedSpeakingFrame(0.02, d.now()), "VAD#0")
	d.wait(20 * time.Millisecond)
	d.push(frames.NewTranscriptionFrame("hi", "u", ""), "STT#0")

	for range 2 {
		d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
		d.wait(20 * time.Millisecond)
		d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")
		d.push(frames.NewLLMMarkerFrame("hold"), "LLM#0")
		d.wait(5 * time.Second)
	}

	d.push(frames.NewLLMFullResponseStartFrame(), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(d.ttfb("LLM#0", 20*time.Millisecond), "LLM#0")
	d.push(frames.NewLLMTextFrame("Hi!"), "LLM#0")
	d.wait(20 * time.Millisecond)
	d.push(frames.NewTTSAudioRawFrame(nil, 24000, 1), "TTS#0")
	d.push(frames.NewBotStartedSpeakingFrame(), "Transport#0")

	b := d.last()
	var waits []observers.LatencyContribution
	for _, c := range b.Contributions {
		if c.Label == "waiting for user" {
			waits = append(waits, c)
		}
	}
	if len(waits) != 2 {
		t.Fatalf("waits = %d, want 2: the turn was held twice", len(waits))
	}
	for _, w := range waits {
		if w.Duration != 5*time.Second {
			t.Errorf("wait = %s, want exactly 5s", w.Duration)
		}
	}
}
