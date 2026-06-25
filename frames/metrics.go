package frames

import "fmt"

// LLMTokenUsage reports the token counts billed for one LLM generation. The
// cache counts are a subset of the input tokens: CacheReadTokens were served
// from a prompt cache and CacheCreationTokens were written to one.
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
}

// MetricsFrame reports metrics measured by a processor. It is a system frame, so
// it is delivered with priority and is not dropped by an interruption — usage is
// billed even when a turn is cut short. It currently carries LLM token usage;
// further metric kinds can be added as fields.
type MetricsFrame struct {
	BaseSystemFrame
	// Processor is the name of the processor that produced the metrics.
	Processor string
	// Tokens reports LLM token usage, or nil when not applicable.
	Tokens *LLMTokenUsage
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
