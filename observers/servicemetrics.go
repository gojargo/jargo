package observers

import (
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// ServiceLatencyKind is which measurement of a service's own time a record
// carries.
type ServiceLatencyKind string

// The waits a service reports.
const (
	LatencyTTFB ServiceLatencyKind = "ttfb"
	LatencyTTFA ServiceLatencyKind = "ttfa"
)

// ServiceUsageKind is which kind of service consumed something.
type ServiceUsageKind string

// The kinds of service that report usage.
const (
	UsageSTT ServiceUsageKind = "stt"
	UsageLLM ServiceUsageKind = "llm"
	UsageTTS ServiceUsageKind = "tts"
)

// ServiceLatencyRecord is one measurement of how long a service took.
type ServiceLatencyRecord struct {
	// Kind is which wait was measured.
	Kind ServiceLatencyKind
	// Processor is the name of the processor that reported it.
	Processor string
	// Model is the model the processor was using, "" where it names none.
	Model string
	// Timestamp is the wall-clock time the measurement was observed at.
	Timestamp time.Time
	// Duration is the measurement itself.
	Duration time.Duration
	// TTFB is the time to first byte the measurement builds on, on the kinds
	// that report one. It is that same measurement rather than another one, so
	// do not add it to Duration.
	TTFB time.Duration
	// LeadingSilence is the silence at the head of the first audio, on the time
	// to first audible sample.
	LeadingSilence time.Duration
}

// ServiceUsageRecord is what one service consumed doing a piece of work.
//
// A field is set only where the kind of service reports it, so an LLM record
// carries token counts and a text-to-speech record carries characters. The
// counts a service may leave unreported are pointers, because a count nobody
// measured must not read as a measured zero to whoever is adding up a bill.
type ServiceUsageRecord struct {
	// Kind is which kind of service reported.
	Kind ServiceUsageKind
	// Processor is the name of the processor that reported it.
	Processor string
	// Model is the model the processor was using, "" where it names none.
	Model string
	// Timestamp is the wall-clock time the usage was observed at.
	Timestamp time.Time

	// AudioSeconds is the audio transcribed, for speech-to-text.
	AudioSeconds float64
	// Characters is the characters synthesized, for text-to-speech.
	Characters int

	// PromptTokens is the tokens in the prompt, for an LLM.
	PromptTokens int64
	// CompletionTokens is the tokens generated.
	CompletionTokens int64
	// TotalTokens is the tokens in the prompt and the completion together.
	TotalTokens int64
	// CacheReadTokens is the prompt tokens served from cache.
	CacheReadTokens *int64
	// CacheCreationTokens is the prompt tokens written to cache.
	CacheCreationTokens *int64
	// ReasoningTokens is the tokens spent reasoning before answering.
	ReasoningTokens *int64
	// InputAudioTokens is the audio tokens in the prompt.
	InputAudioTokens *int64
	// OutputAudioTokens is the audio tokens generated.
	OutputAudioTokens *int64
	// CacheReadAudioTokens is the audio prompt tokens served from cache.
	CacheReadAudioTokens *int64
	// InputTextTokens is the text tokens in the prompt, where the model reports
	// a per-modality breakdown.
	InputTextTokens *int64
	// OutputTextTokens is the text tokens generated, where the model reports a
	// per-modality breakdown.
	OutputTextTokens *int64
}

// ServiceMetricsConfig configures a ServiceMetrics observer.
type ServiceMetricsConfig struct {
	// MaxFrames is how many recent frame ids the observer remembers to
	// recognize one it has already reported; 0 uses 100.
	MaxFrames int
	// Now reads the current time. Nil uses time.Now. Supplying one lets a test
	// place records without waiting.
	Now func() time.Time
	// OnServiceLatency is called for each measurement of a service's own time.
	OnServiceLatency func(r ServiceLatencyRecord)
	// OnServiceUsage is called for each report of what a service consumed.
	OnServiceUsage func(r ServiceUsageRecord)
}

// ServiceMetrics reports each metric a service publishes as its own record.
//
// Services report metrics as they finish a piece of work, and this observer
// turns each one into a record: what was measured, which processor and model
// reported it, and when. Nothing is summed, so a consumer groups the records by
// turn, session or model as it needs, and a session that ends abruptly still
// leaves behind everything that happened before it did. A turn that runs two
// inferences reports two records, and a consumer that wants a total groups them
// itself: summing here would lose the grain, and a total held in memory is lost
// with the process holding it.
//
// What a service made someone wait for is here; what it did with its own time is
// not. Processing time, text aggregation and end-of-turn predictions are all
// deliberately absent: aggregation already appears as a span in
// [LatencyBreakdown], and the other two describe how work was done rather than
// what it cost the person waiting.
type ServiceMetrics struct {
	cfg ServiceMetricsConfig

	mu sync.Mutex
	dd deduper
}

// NewServiceMetrics builds a ServiceMetrics observer.
func NewServiceMetrics(cfg ServiceMetricsConfig) *ServiceMetrics {
	return &ServiceMetrics{cfg: cfg, dd: newDeduper(cfg.MaxFrames)}
}

// now reads the clock the observer was configured with.
func (o *ServiceMetrics) now() time.Time {
	if o.cfg.Now != nil {
		return o.cfg.Now()
	}
	return time.Now()
}

// OnPushFrame implements processor.Observer. It reports the metrics a frame
// carries, the first time that frame is seen: every processor that passes it
// along reports it.
//
// Metrics travel in one direction, so a frame is identified by its id alone.
func (o *ServiceMetrics) OnPushFrame(data processor.FramePushed) {
	f, ok := data.Frame.(*frames.MetricsFrame)
	if !ok {
		return
	}

	o.mu.Lock()
	if o.dd.seenBefore(f.ID()) {
		o.mu.Unlock()
		return
	}
	at := o.now()
	o.mu.Unlock()

	for _, m := range f.Data {
		if r, ok := latencyRecord(m, at); ok {
			if o.cfg.OnServiceLatency != nil {
				o.cfg.OnServiceLatency(r)
			}
			continue
		}
		if r, ok := usageRecord(m, at); ok {
			if o.cfg.OnServiceUsage != nil {
				o.cfg.OnServiceUsage(r)
			}
		}
	}
}

// latencyRecord builds a record for a metric that measures time spent waiting,
// reporting false for one that measures something else.
func latencyRecord(m frames.MetricsData, at time.Time) (ServiceLatencyRecord, bool) {
	r := ServiceLatencyRecord{
		Processor: m.MetricsProcessor(),
		Model:     m.MetricsModel(),
		Timestamp: at,
	}
	switch m := m.(type) {
	case frames.TTFAMetricsData:
		r.Kind = LatencyTTFA
		r.Duration = m.TTFA
		r.TTFB = m.TTFB
		r.LeadingSilence = m.LeadingSilence
	case frames.TTFBMetricsData:
		r.Kind = LatencyTTFB
		r.Duration = m.Value
	default:
		return ServiceLatencyRecord{}, false
	}
	return r, true
}

// usageRecord builds a record for a metric that measures what was consumed,
// reporting false for one that measures something else.
func usageRecord(m frames.MetricsData, at time.Time) (ServiceUsageRecord, bool) {
	r := ServiceUsageRecord{
		Processor: m.MetricsProcessor(),
		Model:     m.MetricsModel(),
		Timestamp: at,
	}
	switch m := m.(type) {
	case frames.LLMUsageMetricsData:
		t := m.Value
		r.Kind = UsageLLM
		r.PromptTokens = t.PromptTokens
		r.CompletionTokens = t.CompletionTokens
		r.TotalTokens = t.TotalTokens
		r.CacheReadTokens = t.CacheReadTokens
		r.CacheCreationTokens = t.CacheCreationTokens
		r.ReasoningTokens = t.ReasoningTokens
		r.InputAudioTokens = t.InputAudioTokens
		r.OutputAudioTokens = t.OutputAudioTokens
		r.CacheReadAudioTokens = t.CacheReadAudioTokens
		r.InputTextTokens = t.InputTextTokens
		r.OutputTextTokens = t.OutputTextTokens
	case frames.STTUsageMetricsData:
		r.Kind = UsageSTT
		r.AudioSeconds = m.Value.AudioSeconds
	case frames.TTSUsageMetricsData:
		r.Kind = UsageTTS
		r.Characters = m.Value
	default:
		return ServiceUsageRecord{}, false
	}
	return r, true
}

// Compile-time interface check.
var _ processor.Observer = (*ServiceMetrics)(nil)
