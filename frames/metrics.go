package frames

import (
	"fmt"
	"time"
)

// LLMTokenUsage reports the token counts billed for one LLM generation.
// CacheReadTokens were served from a prompt cache and CacheCreationTokens were
// written to one.
//
// Services differ over whether they report the input count net or gross of the
// cache. TotalTokens is the gross figure either way, so it stays comparable
// between services, and it is therefore not always PromptTokens plus
// CompletionTokens. Read the cache counts for the breakdown rather than
// subtracting.
//
// The per-modality audio and text counts are subsets, of the prompt tokens
// (input) and completion tokens (output). Realtime (speech-to-speech) models
// bill audio and text at different rates and report this breakdown.
//
// The counts a service may or may not account for are pointers, so that a
// service which reports a figure of zero is distinguishable from one that does
// not report the figure at all. A model with no prompt cache leaves
// CacheReadTokens nil; a model with one that served nothing from it this
// generation reports zero. Only the second is worth showing on a cost
// dashboard, and only the pointer tells them apart.
type LLMTokenUsage struct {
	// PromptTokens is the number of input tokens. It is net of the cache on a
	// service that reports its cache reads separately.
	PromptTokens int64
	// CompletionTokens is the number of output tokens.
	CompletionTokens int64
	// TotalTokens is every token the generation used, cached input included.
	TotalTokens int64
	// CacheReadTokens is the number of input tokens read from the prompt cache.
	CacheReadTokens *int64
	// CacheCreationTokens is the number of input tokens written to the prompt cache.
	CacheCreationTokens *int64
	// ReasoningTokens is the number of completion tokens the model spent
	// reasoning before answering, on a model that reports them. It is a subset
	// of CompletionTokens.
	ReasoningTokens *int64
	// InputAudioTokens is the number of input (prompt) tokens that were audio,
	// as reported by realtime models. It is a subset of PromptTokens.
	InputAudioTokens *int64
	// OutputAudioTokens is the number of output (completion) tokens that were
	// audio. It is a subset of CompletionTokens.
	OutputAudioTokens *int64
	// CacheReadAudioTokens is the number of cache-read input tokens that were
	// audio. It is a subset of CacheReadTokens, and realtime models price it
	// apart from cached text.
	CacheReadAudioTokens *int64
	// InputTextTokens is the number of input (prompt) tokens that were text,
	// when the model reports a per-modality breakdown. Subset of PromptTokens.
	InputTextTokens *int64
	// OutputTextTokens is the number of output (completion) tokens that were
	// text, when the model reports a per-modality breakdown. Subset of
	// CompletionTokens.
	OutputTextTokens *int64
}

// AddTokens accumulates n into a reported count, starting the count at n when
// the service had not reported one yet. It is for a service whose accounting
// arrives split across several entries, such as a per-modality breakdown.
func AddTokens(count *int64, n int64) *int64 {
	if count == nil {
		return new(n)
	}
	*count += n
	return count
}

// TokenCount reads one of the optional counts: the number of tokens, and
// whether the service accounted for them at all. A count that was not reported
// reads as zero and false, which a caller adding up a bill must not treat as a
// measured zero.
func TokenCount(count *int64) (int64, bool) {
	if count == nil {
		return 0, false
	}
	return *count, true
}

// STTUsage is how much audio a speech-to-text service was given. It is raw
// usage, not cost, and each report is the amount since the one before it, so a
// consumer sums them across a session.
//
// A streaming service is sent all the audio, silence included, so its total
// approximates the length of the stream, which is what most stream-priced
// providers bill. A segmented service is sent only the detected speech, so its
// total covers just those segments.
type STTUsage struct {
	// AudioSeconds is the seconds of audio submitted since the last report.
	AudioSeconds float64
}

// MetricsData is one measurement carried by a MetricsFrame. The concrete types
// are the kinds a processor can report: TTFBMetricsData, TTFAMetricsData,
// ProcessingMetricsData, LLMUsageMetricsData, STTUsageMetricsData,
// TTSUsageMetricsData, TextAggregationMetricsData and TurnMetricsData. A
// consumer switches on the type to read the value.
type MetricsData interface {
	// MetricsProcessor is the name of the processor that measured it.
	MetricsProcessor() string
	// MetricsModel is the model it is attributed to, "" when unknown.
	MetricsModel() string

	isMetricsData()
}

// BaseMetricsData is embedded by every measurement and carries what they all
// have: which processor measured it, and against which model.
type BaseMetricsData struct {
	// Processor is the name of the processor that measured it.
	Processor string
	// Model is the model the measurement is attributed to, "" when unknown.
	Model string
}

// MetricsProcessor implements MetricsData.
func (d BaseMetricsData) MetricsProcessor() string { return d.Processor }

// MetricsModel implements MetricsData.
func (d BaseMetricsData) MetricsModel() string { return d.Model }

func (d BaseMetricsData) isMetricsData() {}

// TTFBMetricsData is time to first byte: how long a service took to produce
// anything at all, its first token or its first audio.
type TTFBMetricsData struct {
	BaseMetricsData
	// Value is the measured time to first byte.
	Value time.Duration
}

// TTFAMetricsData is time to first audible sample: time to first byte plus any
// silence a TTS service padded onto the start of its response.
//
// It is reported with its breakdown so a consumer can see how much of the delay
// the listener hears is padding rather than the service answering, without
// having to match it up with the TTFBMetricsData reported separately. TTFB here
// is that same measurement, not another one, so do not add the two together.
type TTFAMetricsData struct {
	BaseMetricsData
	// TTFA is the time to the first audible sample: TTFB plus LeadingSilence.
	TTFA time.Duration
	// TTFB is the time to first byte that TTFA builds on.
	TTFB time.Duration
	// LeadingSilence is the silence before the first audible sample: TTFA minus
	// TTFB.
	LeadingSilence time.Duration
}

// ProcessingMetricsData is the wall-clock time an operation took.
type ProcessingMetricsData struct {
	BaseMetricsData
	// Value is the measured processing time.
	Value time.Duration
}

// LLMUsageMetricsData is the token usage billed for one LLM generation.
type LLMUsageMetricsData struct {
	BaseMetricsData
	// Value is the token usage.
	Value LLMTokenUsage
}

// STTUsageMetricsData is the audio a speech-to-text service was given.
type STTUsageMetricsData struct {
	BaseMetricsData
	// Value is the usage.
	Value STTUsage
}

// TTSUsageMetricsData is the number of characters a TTS service synthesized,
// which is what providers bill against.
type TTSUsageMetricsData struct {
	BaseMetricsData
	// Value is the number of characters.
	Value int
}

// TextAggregationMetricsData is the time from a model's first token to the
// first complete sentence: what grouping text into sentences costs before
// synthesis can start.
type TextAggregationMetricsData struct {
	BaseMetricsData
	// Value is the measured aggregation time.
	Value time.Duration
}

// TurnMetricsData is what an end-of-turn analyzer decided about a user turn, and
// what it cost to decide. Without it a turn that ends on the safety-net timeout
// is indistinguishable from one the analyzer judged unfinished.
type TurnMetricsData struct {
	BaseMetricsData
	// Complete is whether the turn was predicted to be finished.
	Complete bool
	// Probability is the analyzer's confidence that it was finished.
	Probability float64
	// E2EProcessing is how long deciding took end to end, measured from the
	// point speech turned to silence.
	E2EProcessing time.Duration
}

// MetricsFrame reports measurements made by one or more processors. It is a
// system frame, so it is delivered with priority and is not dropped by an
// interruption: usage is billed even when a turn is cut short.
type MetricsFrame struct {
	BaseSystemFrame
	// Data is the measurements this frame reports. One frame can carry several
	// kinds, and measurements from more than one processor.
	Data []MetricsData
}

// NewMetricsFrame builds a MetricsFrame reporting data.
func NewMetricsFrame(data ...MetricsData) *MetricsFrame {
	return &MetricsFrame{
		BaseSystemFrame: NewBaseSystemFrame("MetricsFrame"),
		Data:            data,
	}
}

// String implements fmt.Stringer.
func (f *MetricsFrame) String() string {
	if len(f.Data) == 1 {
		return fmt.Sprintf("%s(processor: %s, %T)", f.Name(), f.Data[0].MetricsProcessor(), f.Data[0])
	}
	return fmt.Sprintf("%s(%d measurements)", f.Name(), len(f.Data))
}

// Compile-time interface checks.
var (
	_ SystemFrame = (*MetricsFrame)(nil)
	_ MetricsData = TTFBMetricsData{}
	_ MetricsData = TTFAMetricsData{}
	_ MetricsData = ProcessingMetricsData{}
	_ MetricsData = LLMUsageMetricsData{}
	_ MetricsData = STTUsageMetricsData{}
	_ MetricsData = TTSUsageMetricsData{}
	_ MetricsData = TextAggregationMetricsData{}
	_ MetricsData = TurnMetricsData{}
)
