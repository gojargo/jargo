package context

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// The cases here are ported from upstream's summarization suite. Two of its
// tests have no counterpart and are left out on purpose: one drives a message
// whose content is a list of typed parts (an image beside text), which jargo has
// no representation for, and one covers a configuration type upstream has
// already deprecated for removal.
//
// Upstream writes a tool result as its own message with role "tool" and a
// tool_call_id. jargo writes the same thing as a message carrying ToolResults,
// so the messages below are that shape; the sequences and the indices they
// assert are upstream's.

// contextOf builds a conversation whose message list is exactly msgs. The system
// prompt is left empty: these cases put the system message in the list, as
// upstream does, so the selection logic that preserves a leading system message
// is the thing under test.
func contextOf(msgs ...frames.Message) *frames.LLMContext {
	c := frames.NewLLMContext("")
	c.SetMessages(msgs)
	return c
}

func userMsg(text string) frames.Message {
	return frames.Message{Role: frames.RoleUser, Text: text}
}

func assistantMsg(text string) frames.Message {
	return frames.Message{Role: frames.RoleAssistant, Text: text}
}

func systemMsg(text string) frames.Message {
	return frames.Message{Role: frames.RoleSystem, Text: text}
}

func toolCallMsg(calls ...frames.ToolCall) frames.Message {
	return frames.Message{Role: frames.RoleAssistant, ToolCalls: calls}
}

func toolResultMsg(id, content string) frames.Message {
	return frames.Message{
		Role:        frames.RoleUser,
		ToolResults: []frames.ToolResult{{ID: id, Content: content}},
	}
}

func TestEstimateTokensSimpleText(t *testing.T) {
	if got := EstimateTokens("Hello world"); got != 2 {
		t.Errorf("EstimateTokens(%q) = %d, want 2", "Hello world", got)
	}
	if got := EstimateTokens("This is a test message"); got != 5 {
		t.Errorf("EstimateTokens(%q) = %d, want 5", "This is a test message", got)
	}
}

func TestEstimateTokensEmpty(t *testing.T) {
	if got := EstimateTokens(""); got != 0 {
		t.Errorf("EstimateTokens(\"\") = %d, want 0", got)
	}
}

func TestEstimateContextTokens(t *testing.T) {
	empty := contextOf()
	if got := EstimateContextTokens(empty); got != 0 {
		t.Errorf("EstimateContextTokens(empty) = %d, want 0", got)
	}

	convo := contextOf(
		systemMsg("You are helpful"),
		userMsg("Hello"),
		assistantMsg("Hi there"),
	)
	total := EstimateContextTokens(convo)
	if total <= 30 || total >= 50 {
		t.Errorf("EstimateContextTokens = %d, want between 30 and 50 exclusive", total)
	}
}

func TestGetMessagesToSummarizeBasic(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Message 1"),
		assistantMsg("Response 1"),
		userMsg("Message 2"),
		assistantMsg("Response 2"),
		userMsg("Message 3"),
		assistantMsg("Response 3"),
	)

	got := GetMessagesToSummarize(convo, 2)

	if len(got.Messages) != 4 {
		t.Fatalf("selected %d messages, want 4", len(got.Messages))
	}
	if got.Messages[0].Text != "Message 1" {
		t.Errorf("first selected = %q, want %q", got.Messages[0].Text, "Message 1")
	}
	if last := got.Messages[len(got.Messages)-1].Text; last != "Response 2" {
		t.Errorf("last selected = %q, want %q", last, "Response 2")
	}
	if got.LastSummarizedIndex != 4 {
		t.Errorf("LastSummarizedIndex = %d, want 4", got.LastSummarizedIndex)
	}
}

func TestGetMessagesToSummarizeNoSystem(t *testing.T) {
	convo := contextOf(
		userMsg("Message 1"),
		assistantMsg("Response 1"),
		userMsg("Message 2"),
		assistantMsg("Response 2"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 3 {
		t.Fatalf("selected %d messages, want 3", len(got.Messages))
	}
	if got.LastSummarizedIndex != 2 {
		t.Errorf("LastSummarizedIndex = %d, want 2", got.LastSummarizedIndex)
	}
}

func TestGetMessagesToSummarizeInsufficient(t *testing.T) {
	convo := contextOf(userMsg("Message 1"), assistantMsg("Response 1"))

	got := GetMessagesToSummarize(convo, 2)

	if len(got.Messages) != 0 {
		t.Errorf("selected %d messages, want 0", len(got.Messages))
	}
	if got.LastSummarizedIndex != -1 {
		t.Errorf("LastSummarizedIndex = %d, want -1", got.LastSummarizedIndex)
	}
}

// A system message that is not at the head of the list is a mid-conversation
// injection, so it is summarized like any other message.
func TestGetMessagesToSummarizeMidConversationSystemMessage(t *testing.T) {
	convo := contextOf(
		userMsg("Message 1"),
		systemMsg("Mid-conversation instruction"),
		assistantMsg("Response 1"),
		userMsg("Message 2"),
		assistantMsg("Response 2"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 4 {
		t.Fatalf("selected %d messages, want 4", len(got.Messages))
	}
	if got.Messages[0].Text != "Message 1" {
		t.Errorf("first selected = %q, want the head of the list", got.Messages[0].Text)
	}
	if got.Messages[1].Role != frames.RoleSystem {
		t.Errorf("mid-conversation system message was not selected for the summary")
	}
	if got.LastSummarizedIndex != 3 {
		t.Errorf("LastSummarizedIndex = %d, want 3", got.LastSummarizedIndex)
	}
}

// Only the leading system message is preserved; a later one is summarized.
func TestGetMessagesToSummarizeSystemAtHeadWithMidSystem(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Message 1"),
		systemMsg("Mid-conversation instruction"),
		assistantMsg("Response 1"),
		userMsg("Message 2"),
		assistantMsg("Response 2"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 4 {
		t.Fatalf("selected %d messages, want 4", len(got.Messages))
	}
	if got.Messages[0].Text != "Message 1" {
		t.Errorf("first selected = %q, want the message after the system prompt", got.Messages[0].Text)
	}
	if got.LastSummarizedIndex != 4 {
		t.Errorf("LastSummarizedIndex = %d, want 4", got.LastSummarizedIndex)
	}
}

func TestFormatMessagesForSummary(t *testing.T) {
	got := FormatMessagesForSummary([]frames.Message{
		userMsg("Hello"),
		assistantMsg("Hi there"),
	})

	want := "USER: Hello\n\nASSISTANT: Hi there"
	if got != want {
		t.Errorf("FormatMessagesForSummary =\n%q\nwant\n%q", got, want)
	}
}

// A tool call whose result has not arrived stops the summary before the call, so
// the request is never separated from the result that answers it.
func TestFunctionCallInProgressNotSummarized(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("What time is it?"),
		toolCallMsg(frames.ToolCall{ID: "call_123", Name: "get_time", Args: []byte("{}")}),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 1 {
		t.Fatalf("selected %d messages, want 1", len(got.Messages))
	}
	if got.Messages[0].Text != "What time is it?" {
		t.Errorf("first selected = %q", got.Messages[0].Text)
	}
	if got.LastSummarizedIndex != 1 {
		t.Errorf("LastSummarizedIndex = %d, want 1", got.LastSummarizedIndex)
	}
}

func TestCompletedFunctionCallCanBeSummarized(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("What time is it?"),
		toolCallMsg(frames.ToolCall{ID: "call_123", Name: "get_time", Args: []byte("{}")}),
		toolResultMsg("call_123", `{"time": "10:30 AM"}`),
		assistantMsg("It's 10:30 AM"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 4 {
		t.Fatalf("selected %d messages, want 4", len(got.Messages))
	}
	if got.Messages[0].Text != "What time is it?" {
		t.Errorf("first selected = %q", got.Messages[0].Text)
	}
	if last := got.Messages[len(got.Messages)-1].Text; last != "It's 10:30 AM" {
		t.Errorf("last selected = %q", last)
	}
	if got.LastSummarizedIndex != 4 {
		t.Errorf("LastSummarizedIndex = %d, want 4", got.LastSummarizedIndex)
	}
}

// One assistant message requesting two calls, only one of them answered: the
// summary has to stop before that message.
func TestMultipleFunctionCallsInProgress(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Message 1"),
		assistantMsg("Response 1"),
		userMsg("What's the time and date?"),
		toolCallMsg(
			frames.ToolCall{ID: "call_time", Name: "get_time", Args: []byte("{}")},
			frames.ToolCall{ID: "call_date", Name: "get_date", Args: []byte("{}")},
		),
		toolResultMsg("call_time", `{"time": "10:30 AM"}`),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 3 {
		t.Fatalf("selected %d messages, want 3", len(got.Messages))
	}
	if got.LastSummarizedIndex != 3 {
		t.Errorf("LastSummarizedIndex = %d, want 3", got.LastSummarizedIndex)
	}
}

func TestMultipleCompletedFunctionCalls(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("What's the time and date?"),
		toolCallMsg(
			frames.ToolCall{ID: "call_time", Name: "get_time", Args: []byte("{}")},
			frames.ToolCall{ID: "call_date", Name: "get_date", Args: []byte("{}")},
		),
		toolResultMsg("call_time", `{"time": "10:30 AM"}`),
		toolResultMsg("call_date", `{"date": "January 1, 2024"}`),
		assistantMsg("It's 10:30 AM on January 1, 2024"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 5 {
		t.Fatalf("selected %d messages, want 5", len(got.Messages))
	}
	if got.LastSummarizedIndex != 5 {
		t.Errorf("LastSummarizedIndex = %d, want 5", got.LastSummarizedIndex)
	}
}

func TestNoFunctionCalls(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Message 1"),
		assistantMsg("Response 1"),
		userMsg("Message 2"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 2 {
		t.Fatalf("selected %d messages, want 2", len(got.Messages))
	}
	if got.LastSummarizedIndex != 2 {
		t.Errorf("LastSummarizedIndex = %d, want 2", got.LastSummarizedIndex)
	}
}

func TestFunctionCallFormattingInTranscript(t *testing.T) {
	got := FormatMessagesForSummary([]frames.Message{
		userMsg("What time is it?"),
		toolCallMsg(frames.ToolCall{ID: "call_123", Name: "get_time", Args: []byte(`{"tz":"UTC"}`)}),
		toolResultMsg("call_123", `{"time": "10:30 AM"}`),
	})

	want := "USER: What time is it?" +
		"\n\nTOOL_CALL: get_time({\"tz\":\"UTC\"})" +
		"\n\nTOOL_RESULT[call_123]: {\"time\": \"10:30 AM\"}"
	if got != want {
		t.Errorf("FormatMessagesForSummary =\n%q\nwant\n%q", got, want)
	}
}

// A result that lands among the messages being kept leaves its call unanswered
// as far as the summary is concerned, so the summary stops before the call.
func TestToolResponseInKeptRangeIsTreatedAsOrphan(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Message 1"),
		assistantMsg("Response 1"),
		toolCallMsg(frames.ToolCall{ID: "call_1", Name: "get_time", Args: []byte("{}")}),
		toolResultMsg("call_1", `{"time": "10:30 AM"}`),
		userMsg("Latest message"),
	)

	// Keeping the last 2 puts the result at index 4 in the kept range, while the
	// call at index 3 would have been summarized.
	got := GetMessagesToSummarize(convo, 2)

	if len(got.Messages) != 2 {
		t.Fatalf("selected %d messages, want 2", len(got.Messages))
	}
	if got.LastSummarizedIndex != 2 {
		t.Errorf("LastSummarizedIndex = %d, want 2", got.LastSummarizedIndex)
	}
}

func TestToolResponseInSummarizedRangeIsNotOrphan(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Message 1"),
		toolCallMsg(frames.ToolCall{ID: "call_1", Name: "get_time", Args: []byte("{}")}),
		toolResultMsg("call_1", `{"time": "10:30 AM"}`),
		assistantMsg("Response 1"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 4 {
		t.Fatalf("selected %d messages, want 4", len(got.Messages))
	}
	if got.LastSummarizedIndex != 4 {
		t.Errorf("LastSummarizedIndex = %d, want 4", got.LastSummarizedIndex)
	}
}

// The placeholder written the moment a synchronous call starts does not answer
// it: the call is still running.
func TestInProgressPlaceholderLeavesCallUnresolved(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("What time is it?"),
		toolCallMsg(frames.ToolCall{ID: "call_1", Name: "get_time", Args: []byte("{}")}),
		toolResultMsg("call_1", frames.ToolResultInProgress),
		assistantMsg("One moment"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 1 {
		t.Fatalf("selected %d messages, want 1", len(got.Messages))
	}
	if got.LastSummarizedIndex != 1 {
		t.Errorf("LastSummarizedIndex = %d, want 1", got.LastSummarizedIndex)
	}
}

// An async tool's started marker leaves its call unresolved until the final
// developer message reports the result.
func TestAsyncToolStartedLeavesCallUnresolved(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Kick it off"),
		toolCallMsg(frames.ToolCall{ID: "call_1", Name: "long_job", Args: []byte("{}")}),
		frames.NewAsyncToolStartedMessage("call_1"),
		assistantMsg("Started"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 1 {
		t.Fatalf("selected %d messages, want 1", len(got.Messages))
	}
	if got.LastSummarizedIndex != 1 {
		t.Errorf("LastSummarizedIndex = %d, want 1", got.LastSummarizedIndex)
	}
}

func TestAsyncToolFinalResolvesCall(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("Kick it off"),
		toolCallMsg(frames.ToolCall{ID: "call_1", Name: "long_job", Args: []byte("{}")}),
		frames.NewAsyncToolStartedMessage("call_1"),
		frames.NewAsyncToolFinalMessage("call_1", "done"),
		assistantMsg("Finished"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 5 {
		t.Fatalf("selected %d messages, want 5", len(got.Messages))
	}
	if got.LastSummarizedIndex != 5 {
		t.Errorf("LastSummarizedIndex = %d, want 5", got.LastSummarizedIndex)
	}
}

// A message written in one provider's own format cannot be read here, so it is
// skipped by every part of the selection.
func TestEstimateContextTokensSkipsLLMSpecificMessages(t *testing.T) {
	plain := contextOf(userMsg("Hello"))
	withSpecific := contextOf(
		userMsg("Hello"),
		frames.NewLLMSpecificMessage("anthropic", map[string]any{"thinking": "a long private deliberation"}),
	)

	if EstimateContextTokens(plain) != EstimateContextTokens(withSpecific) {
		t.Errorf("a provider-specific message changed the estimate: %d vs %d",
			EstimateContextTokens(plain), EstimateContextTokens(withSpecific))
	}
}

func TestFormatMessagesSkipsLLMSpecificMessages(t *testing.T) {
	got := FormatMessagesForSummary([]frames.Message{
		userMsg("Hello"),
		frames.NewLLMSpecificMessage("anthropic", map[string]any{"thinking": "private"}),
		assistantMsg("Hi there"),
	})

	want := "USER: Hello\n\nASSISTANT: Hi there"
	if got != want {
		t.Errorf("FormatMessagesForSummary =\n%q\nwant\n%q", got, want)
	}
}

func TestFunctionCallTrackingSkipsLLMSpecificMessages(t *testing.T) {
	convo := contextOf(
		systemMsg("System prompt"),
		userMsg("What time is it?"),
		frames.NewLLMSpecificMessage("anthropic", map[string]any{"thinking": "private"}),
		toolCallMsg(frames.ToolCall{ID: "call_1", Name: "get_time", Args: []byte("{}")}),
		toolResultMsg("call_1", `{"time":"10:30"}`),
		assistantMsg("It's 10:30"),
		userMsg("Latest message"),
	)

	got := GetMessagesToSummarize(convo, 1)

	if len(got.Messages) != 5 {
		t.Fatalf("selected %d messages, want 5", len(got.Messages))
	}
	if got.LastSummarizedIndex != 5 {
		t.Errorf("LastSummarizedIndex = %d, want 5", got.LastSummarizedIndex)
	}
}

func TestSummaryConfigDefaults(t *testing.T) {
	c := SummaryConfig{}.withDefaults()

	if c.TargetContextTokens != 6000 {
		t.Errorf("TargetContextTokens = %d, want 6000", c.TargetContextTokens)
	}
	if c.MinMessagesAfterSummary != 4 {
		t.Errorf("MinMessagesAfterSummary = %d, want 4", c.MinMessagesAfterSummary)
	}
	if c.SummaryMessageTemplate != "Conversation summary: {summary}" {
		t.Errorf("SummaryMessageTemplate = %q", c.SummaryMessageTemplate)
	}
	if c.SummarizationTimeout != DefaultSummarizationTimeout {
		t.Errorf("SummarizationTimeout = %v, want %v", c.SummarizationTimeout, DefaultSummarizationTimeout)
	}
}

func TestSummaryConfigPrompt(t *testing.T) {
	if got := (SummaryConfig{}).SummaryPrompt(); got != DefaultSummarizationPrompt {
		t.Error("an unset prompt should be the default summarization prompt")
	}
	if got := (SummaryConfig{SummarizationPrompt: "custom"}).SummaryPrompt(); got != "custom" {
		t.Errorf("SummaryPrompt() = %q, want %q", got, "custom")
	}
}

func TestAutoSummarizationConfigDefaults(t *testing.T) {
	c := AutoSummarizationConfig{}.withDefaults()

	if c.MaxContextTokens != 8000 {
		t.Errorf("MaxContextTokens = %d, want 8000", c.MaxContextTokens)
	}
	if c.MaxUnsummarizedMessages != 20 {
		t.Errorf("MaxUnsummarizedMessages = %d, want 20", c.MaxUnsummarizedMessages)
	}
	if c.SummaryConfig.TargetContextTokens != 6000 {
		t.Errorf("TargetContextTokens = %d, want 6000", c.SummaryConfig.TargetContextTokens)
	}
}

// A summary allowed to be longer than the context limit it compresses into is
// pulled down to a share of that limit.
func TestAutoSummarizationConfigTargetTokensAdjusted(t *testing.T) {
	c := AutoSummarizationConfig{
		MaxContextTokens: 4000,
		SummaryConfig:    SummaryConfig{TargetContextTokens: 6000},
	}.withDefaults()

	if c.SummaryConfig.TargetContextTokens != 3200 {
		t.Errorf("TargetContextTokens = %d, want 3200 (80%% of 4000)", c.SummaryConfig.TargetContextTokens)
	}
}

func TestAutoSummarizationConfigTargetTokensNotAdjustedWhenTokenThresholdOff(t *testing.T) {
	c := AutoSummarizationConfig{
		MaxContextTokens: Disabled,
		SummaryConfig:    SummaryConfig{TargetContextTokens: 6000},
	}.withDefaults()

	if c.SummaryConfig.TargetContextTokens != 6000 {
		t.Errorf("TargetContextTokens = %d, want 6000 left alone", c.SummaryConfig.TargetContextTokens)
	}
}

func TestAutoSummarizationConfigOneThresholdMayBeDisabled(t *testing.T) {
	if err := (AutoSummarizationConfig{MaxContextTokens: Disabled}).Validate(); err != nil {
		t.Errorf("disabling only the token threshold should be allowed: %v", err)
	}
	if err := (AutoSummarizationConfig{MaxUnsummarizedMessages: Disabled}).Validate(); err != nil {
		t.Errorf("disabling only the message threshold should be allowed: %v", err)
	}
}

func TestAutoSummarizationConfigBothThresholdsDisabledIsRejected(t *testing.T) {
	c := AutoSummarizationConfig{MaxContextTokens: Disabled, MaxUnsummarizedMessages: Disabled}
	if err := c.Validate(); err == nil {
		t.Error("disabling both thresholds should be rejected")
	}
}
