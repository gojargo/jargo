package context

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/gojargo/jargo/frames"
)

// DefaultSummarizationTimeout bounds a summary generation that names no timeout
// of its own.
const DefaultSummarizationTimeout = 120 * time.Second

const (
	// charsPerToken is the heuristic behind every estimate here: one token is
	// roughly four characters. It holds well enough across prose, code and
	// languages for a threshold check, which is all these estimates are for. Ask
	// the model's own tokenizer when an exact count matters.
	charsPerToken = 4
	// tokenOverheadPerMessage is the structural cost of a message beyond its
	// content: the role, the delimiters the provider wraps it in.
	tokenOverheadPerMessage = 10
)

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
	// pipeline's LLM instead, as a request frame.
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
	defaultSummaryMessageTemplate  = "Conversation summary: {summary}"
	defaultMaxContextTokens        = 8000
	defaultMaxUnsummarizedMessages = 20
	// summaryTokenBudgetRatio is the share of the context limit a summary is
	// allowed to take when the configured target would not fit under it.
	summaryTokenBudgetRatio = 0.8
)

// withDefaults fills the unset fields, leaving the caller's values alone.
func (c SummaryConfig) withDefaults() SummaryConfig {
	if c.TargetContextTokens == 0 {
		c.TargetContextTokens = defaultTargetContextTokens
	}
	if c.MinMessagesAfterSummary == 0 {
		c.MinMessagesAfterSummary = defaultMinMessagesAfterSummary
	}
	if c.SummaryMessageTemplate == "" {
		c.SummaryMessageTemplate = defaultSummaryMessageTemplate
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
	// ErrNoThreshold marks both automatic triggers being disabled at once,
	// which would leave nothing to start a summarization.
	ErrNoThreshold = errors.New("at least one of MaxContextTokens and MaxUnsummarizedMessages must be set")
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
// The two thresholds are independently disableable, but not both: set one to
// Disabled to leave only the other.
type AutoSummarizationConfig struct {
	// MaxContextTokens is the estimated context size above which the
	// conversation is compressed. Zero uses 8000; Disabled turns the
	// token threshold off, leaving MaxUnsummarizedMessages the only trigger.
	MaxContextTokens int
	// MaxUnsummarizedMessages is how many messages may accumulate since the last
	// summary before the conversation is compressed again, so it is compressed
	// regularly even when the token threshold is never reached. Zero uses 20;
	// Disabled turns the message threshold off.
	MaxUnsummarizedMessages int
	// SummaryConfig controls how the summary itself is generated.
	SummaryConfig SummaryConfig
}

// Disabled turns off one of AutoSummarizationConfig's two thresholds. It is
// distinct from the zero value, which asks for that threshold's default.
const Disabled = -1

// withDefaults fills the unset fields and reconciles the summary's token budget
// with the context limit, which is what upstream's construction does.
func (c AutoSummarizationConfig) withDefaults() AutoSummarizationConfig {
	if c.MaxContextTokens == 0 {
		c.MaxContextTokens = defaultMaxContextTokens
	}
	if c.MaxUnsummarizedMessages == 0 {
		c.MaxUnsummarizedMessages = defaultMaxUnsummarizedMessages
	}
	c.SummaryConfig = c.SummaryConfig.withDefaults()
	// A summary allowed to be longer than the whole context it is compressing
	// into would never bring the conversation back under the threshold, so it is
	// pulled down to a share of the limit.
	if c.MaxContextTokens != Disabled && c.SummaryConfig.TargetContextTokens > c.MaxContextTokens {
		c.SummaryConfig.TargetContextTokens = int(float64(c.MaxContextTokens) * summaryTokenBudgetRatio)
	}
	return c
}

// Validate reports whether the configuration is usable.
func (c AutoSummarizationConfig) Validate() error {
	if c.MaxContextTokens == Disabled && c.MaxUnsummarizedMessages == Disabled {
		return ErrNoThreshold
	}
	if c.MaxContextTokens < 0 && c.MaxContextTokens != Disabled {
		return ErrMaxContextTokens
	}
	if c.MaxUnsummarizedMessages != Disabled && c.MaxUnsummarizedMessages < 0 {
		return ErrMaxUnsummarizedMessages
	}
	return c.SummaryConfig.Validate()
}

// MessagesToSummarize is what GetMessagesToSummarize selected: the messages to
// fold into the summary, and the index of the last of them. LastSummarizedIndex
// is -1 when there is nothing to summarize.
type MessagesToSummarize struct {
	Messages            []frames.Message
	LastSummarizedIndex int
}

// EstimateTokens estimates how many tokens text is, by the four-characters-per-
// token heuristic.
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return len(text) / charsPerToken
}

// EstimateContextTokens estimates the size of a conversation: every message's
// content, the tool calls it requested and the results it carries, plus the
// structural overhead each one costs.
//
// A message written in one provider's own format is skipped: only that
// provider's adapter can read it, so nothing here can measure it.
func EstimateContextTokens(convo *frames.LLMContext) int {
	return estimateMessagesTokens(convo.Messages())
}

// estimateMessagesTokens is EstimateContextTokens over a message list already in
// hand, so a caller holding one does not read the conversation twice.
func estimateMessagesTokens(messages []frames.Message) int {
	total := 0
	for _, m := range messages {
		if m.IsLLMSpecific() {
			continue
		}
		total += tokenOverheadPerMessage
		total += EstimateTokens(m.Text)
		for _, tc := range m.ToolCalls {
			total += EstimateTokens(tc.Name + string(tc.Args))
		}
		// A tool result is addressed to the call it answers, and that
		// addressing costs the same structural overhead as a message's role.
		for range m.ToolResults {
			total += tokenOverheadPerMessage
		}
	}
	return total
}

// isToolResultPending reports whether a tool result is a placeholder standing in
// for an answer that has not arrived: the sentinel written the moment a
// synchronous call starts, or an async tool's started marker. Either way the
// call it belongs to is still running.
func isToolResultPending(content string) bool {
	if content == frames.ToolResultInProgress {
		return true
	}
	var payload struct {
		Type   string `json:"type"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal([]byte(content), &payload); err != nil {
		return false
	}
	return payload.Type == frames.AsyncToolPayloadType && payload.Status == frames.AsyncToolStatusRunning
}

// earliestUnresolvedToolCall finds the first message in [start, end) that
// requested a tool call nothing in that range answers, and returns its index, or
// -1 when every call made in the range is answered inside it.
//
// A call counts as answered only by a result within the range: one that falls in
// the messages being kept would be orphaned from its request if the request were
// summarized away, which providers reject outright. A result still carrying a
// placeholder does not count either, since the call is still running.
func earliestUnresolvedToolCall(messages []frames.Message, start, end int) int {
	// Tool call id to the index of the message that requested it.
	pending := make(map[string]int)

	for i := start; i < end; i++ {
		m := messages[i]
		// A message in a provider's own format carries neither tool calls nor
		// results, so it cannot affect what is outstanding.
		if m.IsLLMSpecific() {
			continue
		}

		if m.Role == frames.RoleAssistant {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" {
					pending[tc.ID] = i
				}
			}
		}

		// A result answers its call unless it is still a placeholder.
		for _, tr := range m.ToolResults {
			if tr.ID == "" {
				continue
			}
			if _, ok := pending[tr.ID]; !ok {
				continue
			}
			if !isToolResultPending(tr.Content) {
				delete(pending, tr.ID)
			}
		}

		// An async tool reports its final result as a developer message, which
		// is what resolves the call its started marker left outstanding.
		if am, ok := frames.ParseAsyncToolMessage(m); ok && am.Kind == frames.AsyncToolFinal {
			delete(pending, am.ToolCallID)
		}
	}

	earliest := -1
	for _, idx := range pending {
		if earliest == -1 || idx < earliest {
			earliest = idx
		}
	}
	return earliest
}

// GetMessagesToSummarize selects the messages to fold into a summary, keeping
// out of it:
//
//   - a system message at the head of the list, which frames the assistant's
//     behavior and has to survive compression (jargo normally holds the system
//     prompt outside the message list, where it is preserved anyway; this covers
//     a conversation that carries one as a message);
//   - the minMessagesToKeep most recent messages, which hold the immediate
//     conversational context;
//   - and everything from the first unanswered tool call onwards, so a request
//     is never summarized away from the result answering it.
//
// It reports no messages, and an index of -1, when there is nothing to
// summarize.
func GetMessagesToSummarize(convo *frames.LLMContext, minMessagesToKeep int) MessagesToSummarize {
	return selectMessagesToSummarize(convo.Messages(), minMessagesToKeep)
}

// selectMessagesToSummarize is GetMessagesToSummarize over a message list
// already in hand.
func selectMessagesToSummarize(messages []frames.Message, minMessagesToKeep int) MessagesToSummarize {
	none := MessagesToSummarize{LastSummarizedIndex: -1}

	if len(messages) <= minMessagesToKeep {
		return none
	}

	// Only the head of the list is treated as the system preamble. A system
	// message anywhere else is a mid-conversation injection and belongs in the
	// summary like any other message.
	start := 0
	if !messages[0].IsLLMSpecific() && messages[0].Role == frames.RoleSystem {
		start = 1
	}

	end := len(messages) - minMessagesToKeep
	if start >= end {
		return none
	}

	if unresolved := earliestUnresolvedToolCall(messages, start, end); unresolved >= 0 && unresolved < end {
		slog.Debug("stopping the summary before a tool call still awaiting its result",
			"index", unresolved, "was_summarizing_to", end, "skipped", end-unresolved)
		end = unresolved
	}

	if start >= end {
		return none
	}

	return MessagesToSummarize{
		Messages:            append([]frames.Message(nil), messages[start:end]...),
		LastSummarizedIndex: end - 1,
	}
}

// FormatMessagesForSummary renders messages as the transcript handed to the
// model to summarize.
//
// A message written in one provider's own format is left out: it holds internal
// data (a reasoning block, a thought signature) that is not meaningful as plain
// text, and the conversational content of that turn is carried by the ordinary
// assistant message beside it.
func FormatMessagesForSummary(messages []frames.Message) string {
	parts := make([]string, 0, len(messages))

	for _, m := range messages {
		if m.IsLLMSpecific() {
			continue
		}

		if m.Text != "" {
			parts = append(parts, strings.ToUpper(string(m.Role))+": "+m.Text)
		}

		for _, tc := range m.ToolCalls {
			name := tc.Name
			if name == "" {
				name = "unknown"
			}
			parts = append(parts, fmt.Sprintf("TOOL_CALL: %s(%s)", name, string(tc.Args)))
		}

		for _, tr := range m.ToolResults {
			id := tr.ID
			if id == "" {
				id = "unknown"
			}
			parts = append(parts, fmt.Sprintf("TOOL_RESULT[%s]: %s", id, tr.Content))
		}
	}

	return strings.Join(parts, "\n\n")
}
