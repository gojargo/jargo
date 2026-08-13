package frames

import (
	"errors"
	"fmt"
	"time"
)

// This file holds the configuration a context summarization runs under and the
// three frames that carry one through the pipeline. The selection logic those
// configurations drive lives in utils/context, beside the rest of the
// conversation machinery; the types are here because they travel on frames and
// because the conversation they describe, LLMContext, is here too.

// DefaultSummarizationTimeout bounds a summary generation that names no timeout
// of its own.
const DefaultSummarizationTimeout = 120 * time.Second

// DefaultSummarizationPrompt instructs the model how to compress a conversation.
// It is used whenever a summary configuration names no prompt of its own.
const DefaultSummarizationPrompt = `You are summarizing a conversation between a user and an AI assistant.

Your task:
1. Create a concise summary that preserves:
   - Key facts, decisions, and agreements
   - Important context needed to continue the conversation
   - User preferences and requirements mentioned
   - Any unresolved questions or action items

2. Format:
   - Use clear, factual statements
   - Group related information
   - Prioritize information likely to be referenced later
   - Keep the summary concise to fit within the specified token budget

3. Omit:
   - Greetings and small talk
   - Redundant information
   - Tangential discussions that were resolved

The conversation transcript follows. Generate only the summary, no other text.`

// SummaryConfig controls how a summary is generated. It is shared by automatic
// and on-demand summarization alike.
type SummaryConfig struct {
	// TargetContextTokens caps the length of the generated summary, passed to
	// the model as its token limit. Size it so the summary plus the messages
	// kept after it still fit the model's context comfortably. Zero uses 6000.
	TargetContextTokens int
	// MinMessagesAfterSummary is how many of the most recent messages are kept
	// out of the summary, so the immediate conversational context survives
	// compression verbatim. Zero uses 4; a negative value is rejected.
	MinMessagesAfterSummary int
	// SummarizationPrompt tells the model how to summarize. Empty uses
	// DefaultSummarizationPrompt.
	SummarizationPrompt string
	// SummaryMessageTemplate formats the summary as it is written back into the
	// conversation. It must contain "{summary}", which is replaced by the
	// generated text, so an application can wrap the summary in delimiters of
	// its own and let the system prompt tell a summary from live conversation.
	// Empty uses "Conversation summary: {summary}".
	SummaryMessageTemplate string
	// LLM generates the summary on its own, instead of the conversation's LLM.
	// Set it to route summarization to a cheaper or faster model while an
	// expensive one carries the conversation. Nil sends the request to the
	// pipeline's LLM instead, as an LLMContextSummaryRequestFrame.
	//
	// It is typed as any because the LLM services are built on this package, so
	// their interface cannot be named here. Set it to an llm.Inferencer;
	// anything else is reported and the request goes to the pipeline's LLM.
	LLM any
	// SummarizationTimeout bounds one summary generation. Exceeding it abandons
	// that summary and unblocks the next. Zero uses DefaultSummarizationTimeout.
	SummarizationTimeout time.Duration
}

const (
	defaultTargetContextTokens     = 6000
	defaultMinMessagesAfterSummary = 4
	// DefaultSummaryMessageTemplate wraps the summary as it is written back into
	// the conversation when a configuration names no template of its own.
	DefaultSummaryMessageTemplate  = "Conversation summary: {summary}"
	defaultMaxContextTokens        = 8000
	defaultMaxUnsummarizedMessages = 20
	// summaryTokenBudgetRatio is the share of the context limit a summary is
	// allowed to take when the configured target would not fit under it.
	summaryTokenBudgetRatio = 0.8
)

// WithDefaults fills the unset fields, leaving the caller's values alone.
func (c SummaryConfig) WithDefaults() SummaryConfig {
	if c.TargetContextTokens == 0 {
		c.TargetContextTokens = defaultTargetContextTokens
	}
	if c.MinMessagesAfterSummary == 0 {
		c.MinMessagesAfterSummary = defaultMinMessagesAfterSummary
	}
	if c.SummaryMessageTemplate == "" {
		c.SummaryMessageTemplate = DefaultSummaryMessageTemplate
	}
	if c.SummarizationTimeout == 0 {
		c.SummarizationTimeout = DefaultSummarizationTimeout
	}
	return c
}

// The ways a summarization configuration can be unusable.
var (
	// ErrTargetContextTokens marks a negative summary token budget.
	ErrTargetContextTokens = errors.New("TargetContextTokens must be positive")
	// ErrMinMessagesAfterSummary marks a negative count of messages to keep.
	ErrMinMessagesAfterSummary = errors.New("MinMessagesAfterSummary must be non-negative")
	// ErrNoSummarizationThreshold marks both automatic triggers being disabled at
	// once, which would leave nothing to start a summarization.
	ErrNoSummarizationThreshold = errors.New("at least one of MaxContextTokens and MaxUnsummarizedMessages must be set")
	// ErrMaxContextTokens marks a negative context-size threshold.
	ErrMaxContextTokens = errors.New("MaxContextTokens must be positive")
	// ErrMaxUnsummarizedMessages marks a negative message-count threshold.
	ErrMaxUnsummarizedMessages = errors.New("MaxUnsummarizedMessages must be at least 1")
)

// Validate reports whether the configuration is usable.
func (c SummaryConfig) Validate() error {
	if c.TargetContextTokens < 0 {
		return ErrTargetContextTokens
	}
	if c.MinMessagesAfterSummary < 0 {
		return ErrMinMessagesAfterSummary
	}
	return nil
}

// SummaryPrompt is the prompt to summarize with: the configured one, or the
// default when none was given.
func (c SummaryConfig) SummaryPrompt() string {
	if c.SummarizationPrompt != "" {
		return c.SummarizationPrompt
	}
	return DefaultSummarizationPrompt
}

// AutoSummarizationConfig controls when a conversation is compressed
// automatically, and how the summary it produces is generated. Summarization
// runs when either threshold is passed.
//
// Each threshold is a pointer so that turning it off is distinct from leaving it
// at its default: nil disables that threshold, and the two may not both be nil.
// Build one with NewAutoSummarizationConfig to start from the defaults.
type AutoSummarizationConfig struct {
	// MaxContextTokens is the estimated context size above which the
	// conversation is compressed. Nil turns the token threshold off, leaving
	// MaxUnsummarizedMessages the only trigger.
	MaxContextTokens *int
	// MaxUnsummarizedMessages is how many messages may accumulate since the last
	// summary before the conversation is compressed again, so it is compressed
	// regularly even when the token threshold is never reached. Nil turns the
	// message threshold off.
	MaxUnsummarizedMessages *int
	// SummaryConfig controls how the summary itself is generated.
	SummaryConfig SummaryConfig
}

// NewAutoSummarizationConfig is an automatic summarization configuration with
// both thresholds at their defaults: 8000 estimated tokens, or 20 messages since
// the last summary. Set a field to nil afterwards to turn that threshold off.
func NewAutoSummarizationConfig() AutoSummarizationConfig {
	tokens, messages := defaultMaxContextTokens, defaultMaxUnsummarizedMessages
	return AutoSummarizationConfig{
		MaxContextTokens:        &tokens,
		MaxUnsummarizedMessages: &messages,
	}
}

// WithDefaults fills the unset fields and reconciles the summary's token budget
// with the context limit.
func (c AutoSummarizationConfig) WithDefaults() AutoSummarizationConfig {
	c.SummaryConfig = c.SummaryConfig.WithDefaults()
	// A summary allowed to be longer than the whole context it is compressing
	// into would never bring the conversation back under the threshold, so it is
	// pulled down to a share of the limit.
	if c.MaxContextTokens != nil && c.SummaryConfig.TargetContextTokens > *c.MaxContextTokens {
		c.SummaryConfig.TargetContextTokens = int(float64(*c.MaxContextTokens) * summaryTokenBudgetRatio)
	}
	return c
}

// Validate reports whether the configuration is usable.
func (c AutoSummarizationConfig) Validate() error {
	if c.MaxContextTokens == nil && c.MaxUnsummarizedMessages == nil {
		return ErrNoSummarizationThreshold
	}
	if c.MaxContextTokens != nil && *c.MaxContextTokens <= 0 {
		return ErrMaxContextTokens
	}
	if c.MaxUnsummarizedMessages != nil && *c.MaxUnsummarizedMessages < 1 {
		return ErrMaxUnsummarizedMessages
	}
	return c.SummaryConfig.Validate()
}

// LLMSummarizeContextFrame asks for the conversation to be compressed now,
// whatever the automatic thresholds say. Push it into the pipeline to compress
// on demand; it works whether or not automatic summarization is enabled.
//
// It is a control frame, so it is handled in order with the conversation around
// it rather than jumping ahead of a turn in flight.
type LLMSummarizeContextFrame struct {
	BaseControlFrame

	// Config overrides the summary generation settings for this request alone.
	// Nil uses the configured defaults.
	Config *SummaryConfig
}

// NewLLMSummarizeContextFrame builds an on-demand summarization request.
func NewLLMSummarizeContextFrame() *LLMSummarizeContextFrame {
	return &LLMSummarizeContextFrame{
		BaseControlFrame: NewBaseControlFrame("LLMSummarizeContextFrame"),
	}
}

// LLMContextSummaryRequestFrame asks the LLM service to generate a summary of
// the conversation. The summarizer pushes it upstream; the service answers with
// an LLMContextSummaryResultFrame carrying the same RequestID.
type LLMContextSummaryRequestFrame struct {
	BaseControlFrame

	// RequestID matches this request to the result answering it, so a result
	// arriving for a request that has since been abandoned is recognized and
	// dropped.
	RequestID string
	// Context is the conversation to summarize.
	Context *LLMContext
	// MinMessagesToKeep is how many of the most recent messages stay out of the
	// summary.
	MinMessagesToKeep int
	// TargetContextTokens caps the length of the generated summary.
	TargetContextTokens int
	// SummarizationPrompt tells the model how to summarize.
	SummarizationPrompt string
	// SummarizationTimeout bounds this generation. Zero uses
	// DefaultSummarizationTimeout.
	SummarizationTimeout time.Duration
}

// NewLLMContextSummaryRequestFrame builds a summarization request.
func NewLLMContextSummaryRequestFrame(requestID string, convo *LLMContext) *LLMContextSummaryRequestFrame {
	return &LLMContextSummaryRequestFrame{
		BaseControlFrame: NewBaseControlFrame("LLMContextSummaryRequestFrame"),
		RequestID:        requestID,
		Context:          convo,
	}
}

// String implements fmt.Stringer.
func (f *LLMContextSummaryRequestFrame) String() string {
	return fmt.Sprintf("%s(request_id: %s, keep: %d)", f.Name(), f.RequestID, f.MinMessagesToKeep)
}

// LLMContextSummaryResultFrame carries a generated summary back to the
// summarizer that asked for it.
//
// It is uninterruptible: the request that produced it may well have been
// canceled by a barge-in, and the summarizer still has to see the result so it
// can clear the request it is holding open. Dropping it would leave
// summarization blocked for the rest of the call.
type LLMContextSummaryResultFrame struct {
	BaseControlFrame
	UninterruptibleMixin

	// RequestID identifies the request this answers.
	RequestID string
	// Summary is the generated summary, empty when Error is set.
	Summary string
	// LastSummarizedIndex is the index of the last message the summary covers,
	// and -1 when nothing was summarized.
	LastSummarizedIndex int
	// Error describes a summarization that failed, and is empty on success.
	Error string
}

// NewLLMContextSummaryResultFrame builds a summarization result.
func NewLLMContextSummaryResultFrame(requestID, summary string, lastSummarizedIndex int) *LLMContextSummaryResultFrame {
	return &LLMContextSummaryResultFrame{
		BaseControlFrame:    NewBaseControlFrame("LLMContextSummaryResultFrame"),
		RequestID:           requestID,
		Summary:             summary,
		LastSummarizedIndex: lastSummarizedIndex,
	}
}

// NewLLMContextSummaryErrorFrame builds the result of a summarization that
// failed. It carries no summary, so the summarizer clears the request it was
// holding open and leaves the conversation as it stands.
func NewLLMContextSummaryErrorFrame(requestID, message string) *LLMContextSummaryResultFrame {
	return &LLMContextSummaryResultFrame{
		BaseControlFrame:    NewBaseControlFrame("LLMContextSummaryResultFrame"),
		RequestID:           requestID,
		LastSummarizedIndex: -1,
		Error:               message,
	}
}

// String implements fmt.Stringer.
func (f *LLMContextSummaryResultFrame) String() string {
	if f.Error != "" {
		return fmt.Sprintf("%s(request_id: %s, error: %s)", f.Name(), f.RequestID, f.Error)
	}
	return fmt.Sprintf("%s(request_id: %s, chars: %d, last: %d)",
		f.Name(), f.RequestID, len(f.Summary), f.LastSummarizedIndex)
}

// Compile-time interface checks.
var (
	_ ControlFrame    = (*LLMSummarizeContextFrame)(nil)
	_ ControlFrame    = (*LLMContextSummaryRequestFrame)(nil)
	_ ControlFrame    = (*LLMContextSummaryResultFrame)(nil)
	_ Uninterruptible = (*LLMContextSummaryResultFrame)(nil)
)
