package observers_test

import (
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// metricsRecorder collects everything a ServiceMetrics observer reports.
type metricsRecorder struct {
	mu        sync.Mutex
	clock     time.Time
	latencies []observers.ServiceLatencyRecord
	usages    []observers.ServiceUsageRecord
}

func newMetricsRecorder() *metricsRecorder {
	return &metricsRecorder{clock: time.Date(2026, time.September, 13, 12, 0, 0, 0, time.UTC)}
}

func (r *metricsRecorder) now() time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.clock
}

func (r *metricsRecorder) recordLatency(rec observers.ServiceLatencyRecord) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.latencies = append(r.latencies, rec)
}

func (r *metricsRecorder) recordUsage(rec observers.ServiceUsageRecord) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.usages = append(r.usages, rec)
}

func (r *metricsRecorder) allLatencies() []observers.ServiceLatencyRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]observers.ServiceLatencyRecord(nil), r.latencies...)
}

func (r *metricsRecorder) allUsages() []observers.ServiceUsageRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]observers.ServiceUsageRecord(nil), r.usages...)
}

// onlyLatency returns the single latency record, failing when there is not one.
func (r *metricsRecorder) onlyLatency(t *testing.T) observers.ServiceLatencyRecord {
	t.Helper()
	all := r.allLatencies()
	if len(all) != 1 {
		t.Fatalf("latency records = %d, want 1", len(all))
	}
	return all[0]
}

// onlyUsage returns the single usage record, failing when there is not one.
func (r *metricsRecorder) onlyUsage(t *testing.T) observers.ServiceUsageRecord {
	t.Helper()
	all := r.allUsages()
	if len(all) != 1 {
		t.Fatalf("usage records = %d, want 1", len(all))
	}
	return all[0]
}

// newServiceMetrics builds an observer reporting into the recorder.
func newServiceMetrics(r *metricsRecorder) *observers.ServiceMetrics {
	return observers.NewServiceMetrics(observers.ServiceMetricsConfig{
		Now:              r.now,
		OnServiceLatency: r.recordLatency,
		OnServiceUsage:   r.recordUsage,
	})
}

// base names the processor and model every metric here is attributed to.
func base() frames.BaseMetricsData {
	return frames.BaseMetricsData{Processor: "cartesia_tts", Model: "sonic"}
}

// tokens is a count a model reported, as the optional counts are carried.
func tokens(n int64) *int64 { return new(n) }

// TestTimeToFirstByteIsRecorded covers the simplest measurement.
func TestTimeToFirstByteIsRecorded(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(frames.TTFBMetricsData{
		BaseMetricsData: base(),
		Value:           250 * time.Millisecond,
	}), processor.Downstream)

	rec := r.onlyLatency(t)
	if rec.Kind != observers.LatencyTTFB || rec.Duration != 250*time.Millisecond {
		t.Errorf("record = %+v, want a 250ms ttfb", rec)
	}
	if rec.Processor != "cartesia_tts" || rec.Model != "sonic" {
		t.Errorf("record names %s/%s, want cartesia_tts/sonic", rec.Processor, rec.Model)
	}
	if !rec.Timestamp.Equal(r.now()) {
		t.Errorf("timestamp = %v, want the observer's clock", rec.Timestamp)
	}
}

// TestTimeToFirstAudioKeepsWhatItBuildsOn covers a measurement that decomposes
// reporting its parts, so a consumer can see how much of the delay the listener
// hears is padding rather than the service answering.
func TestTimeToFirstAudioKeepsWhatItBuildsOn(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(frames.TTFAMetricsData{
		BaseMetricsData: base(),
		TTFA:            400 * time.Millisecond,
		TTFB:            250 * time.Millisecond,
		LeadingSilence:  150 * time.Millisecond,
	}), processor.Downstream)

	rec := r.onlyLatency(t)
	if rec.Kind != observers.LatencyTTFA {
		t.Fatalf("kind = %s, want ttfa", rec.Kind)
	}
	if rec.Duration != 400*time.Millisecond || rec.TTFB != 250*time.Millisecond ||
		rec.LeadingSilence != 150*time.Millisecond {
		t.Errorf("record = %+v, want the measurement and both of its parts", rec)
	}
}

// TestLLMTokensIncludingTheOptionalOnes covers every count a model reports
// surviving into the record, and a count nobody reported staying unreported
// rather than becoming a measured zero.
func TestLLMTokensIncludingTheOptionalOnes(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(frames.LLMUsageMetricsData{
		Processor: "llm", Model: "a-model",
		Value: frames.LLMTokenUsage{
			PromptTokens:        100,
			CompletionTokens:    20,
			TotalTokens:         120,
			CacheReadTokens:     tokens(64),
			CacheCreationTokens: tokens(32),
			ReasoningTokens:     tokens(8),
		},
	}), processor.Downstream)

	rec := r.onlyUsage(t)
	if rec.Kind != observers.UsageLLM {
		t.Fatalf("kind = %s, want llm", rec.Kind)
	}
	if rec.PromptTokens != 100 || rec.CompletionTokens != 20 || rec.TotalTokens != 120 {
		t.Errorf("record = %+v, want the reported counts", rec)
	}
	for name, got := range map[string]*int64{
		"cache read":     rec.CacheReadTokens,
		"cache creation": rec.CacheCreationTokens,
		"reasoning":      rec.ReasoningTokens,
	} {
		if got == nil {
			t.Errorf("%s tokens went missing", name)
		}
	}
	if rec.InputAudioTokens != nil {
		t.Error("audio tokens were reported, want them left unreported: the model said nothing about them")
	}
}

// TestSpeechToTextAndTextToSpeechUsage covers audio in and characters out.
func TestSpeechToTextAndTextToSpeechUsage(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(
		frames.STTUsageMetricsData{
			Processor: "stt",
			Value:     frames.STTUsage{AudioSeconds: 4.5},
		},
		frames.TTSUsageMetricsData{BaseMetricsData: base(), Value: 128},
	), processor.Downstream)

	all := r.allUsages()
	if len(all) != 2 {
		t.Fatalf("usage records = %d, want 2", len(all))
	}
	if all[0].Kind != observers.UsageSTT || all[0].AudioSeconds != 4.5 {
		t.Errorf("record = %+v, want 4.5s of audio", all[0])
	}
	if all[1].Kind != observers.UsageTTS || all[1].Characters != 128 {
		t.Errorf("record = %+v, want 128 characters", all[1])
	}
}

// TestNothingIsSummed covers the grain the observer keeps: two inferences report
// twice, and the records stay apart rather than being added up here.
func TestNothingIsSummed(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	for _, n := range []int64{100, 250} {
		push(o, frames.NewMetricsFrame(frames.LLMUsageMetricsData{
			Processor: "llm",
			Value:     frames.LLMTokenUsage{PromptTokens: n, TotalTokens: n},
		}), processor.Downstream)
	}

	all := r.allUsages()
	if len(all) != 2 || all[0].PromptTokens != 100 || all[1].PromptTokens != 250 {
		t.Errorf("records = %+v, want the two reports kept apart", all)
	}
}

// TestRelayedMetricReportedOnce covers a frame passed along the pipeline being
// one metric rather than one per hop.
func TestRelayedMetricReportedOnce(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	f := frames.NewMetricsFrame(frames.TTFBMetricsData{BaseMetricsData: base(), Value: time.Second})
	for range 4 {
		push(o, f, processor.Downstream)
	}

	if got := r.allLatencies(); len(got) != 1 {
		t.Errorf("records = %d, want 1", len(got))
	}
}

// TestMetricsMeasuringSomethingElseAreLeftAlone covers the deliberate absences:
// only what a service made someone wait for is a record here. Processing time,
// text aggregation and end-of-turn predictions describe how the work was done.
func TestMetricsMeasuringSomethingElseAreLeftAlone(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(
		frames.ProcessingMetricsData{BaseMetricsData: base(), Value: time.Second},
		frames.TextAggregationMetricsData{BaseMetricsData: base(), Value: time.Second},
		frames.TurnMetricsData{BaseMetricsData: base(), Complete: true, Probability: 0.9},
	), processor.Downstream)

	if got := r.allLatencies(); len(got) != 0 {
		t.Errorf("latency records = %+v, want none", got)
	}
	if got := r.allUsages(); len(got) != 0 {
		t.Errorf("usage records = %+v, want none", got)
	}
}

// TestOtherFramesAreIgnored covers only metrics frames carrying metrics.
func TestOtherFramesAreIgnored(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewTextFrame("hello"), processor.Downstream)

	if got := r.allLatencies(); len(got) != 0 {
		t.Errorf("latency records = %+v, want none", got)
	}
}

// TestOneFrameCanCarrySeveralMetrics covers each becoming its own record, across
// both kinds of report.
func TestOneFrameCanCarrySeveralMetrics(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(
		frames.TTFBMetricsData{BaseMetricsData: base(), Value: 250 * time.Millisecond},
		frames.TTSUsageMetricsData{BaseMetricsData: base(), Value: 42},
	), processor.Downstream)

	if got := r.allLatencies(); len(got) != 1 {
		t.Errorf("latency records = %d, want 1", len(got))
	}
	if got := r.allUsages(); len(got) != 1 {
		t.Errorf("usage records = %d, want 1", len(got))
	}
}

// TestTimeToFirstAnswerTokenKeepsTheThinking covers the measurement that says
// how much of the delay was the model thinking rather than responding.
func TestTimeToFirstAnswerTokenKeepsTheThinking(t *testing.T) {
	r := newMetricsRecorder()
	o := newServiceMetrics(r)

	push(o, frames.NewMetricsFrame(frames.TTFATMetricsData{
		Processor:    "llm",
		TTFAT:        900 * time.Millisecond,
		TTFB:         200 * time.Millisecond,
		ThinkingTime: 700 * time.Millisecond,
	}), processor.Downstream)

	rec := r.onlyLatency(t)
	if rec.Kind != observers.LatencyTTFAT {
		t.Fatalf("kind = %s, want ttfat", rec.Kind)
	}
	if rec.Duration != 900*time.Millisecond || rec.TTFB != 200*time.Millisecond ||
		rec.ThinkingTime != 700*time.Millisecond {
		t.Errorf("record = %+v, want the measurement and both of its parts", rec)
	}
}
