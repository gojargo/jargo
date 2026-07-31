package frames

import (
	"fmt"
	"time"
)

// LLMTokenUsage reports the token counts billed for one LLM generation. The
// cache counts are a subset of the input tokens: CacheReadTokens were served
// from a prompt cache and CacheCreationTokens were written to one. The
// per-modality audio and text counts are likewise subsets — of the prompt
// tokens (input) and completion tokens (output). Realtime (speech-to-speech)
// models bill audio and text at different rates and report this breakdown; a
// text-only generation leaves the audio fields zero.
type LLMTokenUsage struct {
	// PromptTokens is the number of input tokens.
	PromptTokens int64
	// CompletionTokens is the number of output tokens.
	CompletionTokens int64
	// CacheReadTokens is the number of input tokens read from the prompt cache.
	CacheReadTokens int64
	// CacheCreationTokens is the number of input tokens written to the prompt cache.
	CacheCreationTokens int64
	// TotalTokens is the sum of the prompt and completion tokens.
	TotalTokens int64
	// InputAudioTokens is the number of input (prompt) tokens that were audio,
	// as reported by realtime models. It is a subset of PromptTokens.
	InputAudioTokens int64
	// OutputAudioTokens is the number of output (completion) tokens that were
	// audio. It is a subset of CompletionTokens.
	OutputAudioTokens int64
	// InputTextTokens is the number of input (prompt) tokens that were text,
	// when the model reports a per-modality breakdown. Subset of PromptTokens.
	InputTextTokens int64
	// OutputTextTokens is the number of output (completion) tokens that were
	// text, when the model reports a per-modality breakdown. Subset of
	// CompletionTokens.
	OutputTextTokens int64
}

// MetricsFrame reports metrics measured by a processor. It is a system frame, so
// it is delivered with priority and is not dropped by an interruption — usage is
// billed even when a turn is cut short. Each field is optional: a processor sets
// the kinds it measured (e.g. an LLM reports TTFB, processing time and tokens).
type MetricsFrame struct {
	BaseSystemFrame
	// Processor is the name of the processor that produced the metrics.
	Processor string
	// Model is the model that produced the metrics, when known.
	Model string
	// TTFB is the time to first byte (first token or audio), or nil when not measured.
	TTFB *time.Duration
	// TTFA is the time to the first audible sample (TTFB plus any leading
	// silence a TTS service padded onto its response), or nil when not measured.
	TTFA *time.Duration
	// LeadingSilence is the silence before the first audible sample (TTFA minus
	// TTFB), or nil when not measured.
	LeadingSilence *time.Duration
	// Processing is the wall-clock time the operation took, or nil when not measured.
	Processing *time.Duration
	// Tokens reports LLM token usage, or nil when not applicable.
	Tokens *LLMTokenUsage
	// Characters reports the number of characters synthesized by TTS, or nil.
	Characters *int
	// Turn reports an end-of-turn prediction, or nil when not applicable.
	Turn *TurnPrediction
}

// TurnPrediction is what an end-of-turn analyzer decided about a user turn, and
// what it cost to decide. Without it a turn that ends on the safety-net timeout
// is indistinguishable from one the analyzer judged unfinished.
type TurnPrediction struct {
	// Complete is whether the turn was predicted to be finished.
	Complete bool
	// Probability is the analyzer's confidence that it was finished.
	Probability float64
	// Processing is how long the analysis took, measured from the point the
	// analyzer was asked.
	Processing time.Duration
}

// NewMetricsFrame builds a MetricsFrame attributed to the named processor.
func NewMetricsFrame(processor string) *MetricsFrame {
	return &MetricsFrame{
		BaseSystemFrame: NewBaseSystemFrame("MetricsFrame"),
		Processor:       processor,
	}
}

// String implements fmt.Stringer.
func (f *MetricsFrame) String() string {
	if f.Tokens != nil {
		return fmt.Sprintf("%s(processor: %s, tokens: %d in / %d out)",
			f.Name(), f.Processor, f.Tokens.PromptTokens, f.Tokens.CompletionTokens)
	}
	return fmt.Sprintf("%s(processor: %s)", f.Name(), f.Processor)
}

// Compile-time interface check.
var _ SystemFrame = (*MetricsFrame)(nil)
