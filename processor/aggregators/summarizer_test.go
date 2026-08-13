package aggregators_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
)

// Ported from upstream's summarizer suite. Upstream keeps the system prompt as
// the first message of the conversation, so these do too: the selection and the
// rebuild both have a branch for it, and that branch is under test.

// waitFor polls cond until it holds or timeout elapses.
func waitFor(timeout time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return cond()
}

// summarizerFixture is upstream's asyncSetUp: a conversation holding only the
// system message.
func summarizerFixture() *frames.LLMContext {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "You are a helpful assistant."}})
	return c
}

func addUserMessages(c *frames.LLMContext, n int, text string) {
	for range n {
		c.AddMessage(frames.Message{Role: frames.RoleUser, Text: text})
	}
}

// recordRequests attaches a handler collecting every summarization request.
func recordRequests(s *aggregators.Summarizer) *[]*frames.LLMContextSummaryRequestFrame {
	var got []*frames.LLMContextSummaryRequestFrame
	s.Add(aggregators.EventRequestSummarization, func(_ context.Context, _ any, args ...any) {
		if f, ok := args[0].(*frames.LLMContextSummaryRequestFrame); ok {
			got = append(got, f)
		}
	})
	return &got
}

func TestSummarizationTriggeredByTokenLimit(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(100),
		MaxUnsummarizedMessages: new(100),
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "This is a test message that adds tokens to the context.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if len(*got) != 1 {
		t.Fatalf("requested summarization %d times, want 1", len(*got))
	}
	if (*got)[0].Context != convo {
		t.Error("the request does not carry the conversation it is for")
	}
}

func TestSummarizationTriggeredByMessageCount(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(100000),
		MaxUnsummarizedMessages: new(5),
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	for i := range 6 {
		convo.AddMessage(frames.Message{Role: frames.RoleUser, Text: "Message " + string(rune('0'+i))})
	}
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if len(*got) != 1 {
		t.Fatalf("requested summarization %d times, want 1", len(*got))
	}
}

func TestSummarizationNotTriggeredBelowThresholds(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(10000),
		MaxUnsummarizedMessages: new(20),
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 3, "Short message")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if len(*got) != 0 {
		t.Errorf("requested summarization %d times below both thresholds, want 0", len(*got))
	}
}

func TestSummarizationInProgressPreventsDuplicate(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(50),
		MaxUnsummarizedMessages: new(100),
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message to add tokens.")

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 1 {
		t.Fatalf("first trigger requested %d times, want 1", len(*got))
	}
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 1 {
		t.Errorf("a second trigger requested again while one was in flight: %d", len(*got))
	}
}

func TestSummaryResultIsApplied(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{MinMessagesAfterSummary: 2},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")
	originalCount := len(convo.Messages())

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 1 {
		t.Fatalf("no summarization was requested")
	}

	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryResultFrame(
		(*got)[0].RequestID, "This is a test summary.", 5))

	msgs := convo.Messages()
	if len(msgs) >= originalCount {
		t.Errorf("the conversation holds %d messages, want fewer than %d", len(msgs), originalCount)
	}

	summaries := 0
	for _, m := range msgs {
		if strings.Contains(m.Text, "Conversation summary:") {
			summaries++
		}
	}
	if summaries != 1 {
		t.Errorf("the conversation holds %d summary messages, want 1", summaries)
	}
}

// An interruption releases the in-flight marker, so the next turn may summarize
// again.
func TestInterruptionReleasesSummarization(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{MaxContextTokens: new(50), MaxUnsummarizedMessages: new(100)}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 1 {
		t.Fatalf("first trigger requested %d times, want 1", len(*got))
	}

	s.ProcessFrame(t.Context(), frames.NewInterruptionFrame())

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 2 {
		t.Errorf("after an interruption the next turn requested %d times, want 2", len(*got))
	}
}

func TestStaleSummaryResultIsIgnored(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{MinMessagesAfterSummary: 2},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")
	originalCount := len(convo.Messages())

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 1 {
		t.Fatalf("no summarization was requested")
	}

	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryResultFrame(
		"stale-id-123", "Stale summary", 3))

	if len(convo.Messages()) != originalCount {
		t.Errorf("a stale result changed the conversation: %d messages, want %d",
			len(convo.Messages()), originalCount)
	}
}

// With the automatic thresholds off, only a pushed frame summarizes.
func TestManualSummarizationViaFrame(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(100000),
		MaxUnsummarizedMessages: new(100),
	}
	s := aggregators.NewSummarizer(convo, cfg, false)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 5, "Message")

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 0 {
		t.Fatalf("a turn summarized with auto-triggering off")
	}

	s.ProcessFrame(t.Context(), frames.NewLLMSummarizeContextFrame())
	if len(*got) != 1 {
		t.Fatalf("the pushed frame requested %d summarizations, want 1", len(*got))
	}
	if (*got)[0].RequestID == "" {
		t.Error("the request carries no id")
	}
	if (*got)[0].Context != convo {
		t.Error("the request does not carry the conversation it is for")
	}
}

func TestManualSummarizationWithConfigOverride(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(100000),
		SummaryConfig: frames.SummaryConfig{
			TargetContextTokens:     6000,
			MinMessagesAfterSummary: 4,
		},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 5, "Message")

	f := frames.NewLLMSummarizeContextFrame()
	f.Config = &frames.SummaryConfig{TargetContextTokens: 500, MinMessagesAfterSummary: 1}
	s.ProcessFrame(t.Context(), f)

	if len(*got) != 1 {
		t.Fatalf("requested %d summarizations, want 1", len(*got))
	}
	if (*got)[0].TargetContextTokens != 500 {
		t.Errorf("TargetContextTokens = %d, want the override's 500", (*got)[0].TargetContextTokens)
	}
	if (*got)[0].MinMessagesToKeep != 1 {
		t.Errorf("MinMessagesToKeep = %d, want the override's 1", (*got)[0].MinMessagesToKeep)
	}
}

func TestManualSummarizationBlockedWhenInProgress(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{MaxContextTokens: new(100000), MaxUnsummarizedMessages: new(100)}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 5, "Message")

	s.ProcessFrame(t.Context(), frames.NewLLMSummarizeContextFrame())
	if len(*got) != 1 {
		t.Fatalf("requested %d summarizations, want 1", len(*got))
	}
	s.ProcessFrame(t.Context(), frames.NewLLMSummarizeContextFrame())
	if len(*got) != 1 {
		t.Errorf("a second pushed frame requested again while one was in flight: %d", len(*got))
	}
}

// The summary is context supplied to the assistant, not something it said.
func TestSummaryMessageRoleIsUser(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{MinMessagesAfterSummary: 2},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryResultFrame(
		(*got)[0].RequestID, "A summary.", 5))

	for _, m := range convo.Messages() {
		if strings.Contains(m.Text, "Conversation summary:") {
			if m.Role != frames.RoleUser {
				t.Errorf("the summary message has role %q, want %q", m.Role, frames.RoleUser)
			}
			return
		}
	}
	t.Fatal("no summary message was written")
}

func TestSummaryMessageDefaultTemplate(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{MinMessagesAfterSummary: 2},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryResultFrame(
		(*got)[0].RequestID, "A summary.", 5))

	want := "Conversation summary: A summary."
	for _, m := range convo.Messages() {
		if m.Text == want {
			return
		}
	}
	t.Errorf("no message reads %q; conversation is %v", want, convo.Messages())
}

func TestSummaryMessageCustomTemplate(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig: frames.SummaryConfig{
			MinMessagesAfterSummary: 2,
			SummaryMessageTemplate:  "<summary>{summary}</summary>",
		},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryResultFrame(
		(*got)[0].RequestID, "A summary.", 5))

	want := "<summary>A summary.</summary>"
	for _, m := range convo.Messages() {
		if m.Text == want {
			return
		}
	}
	t.Errorf("no message reads %q; conversation is %v", want, convo.Messages())
}

func TestOnSummaryAppliedEvent(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{MinMessagesAfterSummary: 2},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	applied := make(chan aggregators.SummaryApplied, 1)
	s.Add(aggregators.EventSummaryApplied, func(_ context.Context, _ any, args ...any) {
		if e, ok := args[0].(aggregators.SummaryApplied); ok {
			applied <- e
		}
	})

	addUserMessages(convo, 10, "Test message.")
	originalCount := len(convo.Messages())

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryResultFrame(
		(*got)[0].RequestID, "A summary.", 5))

	select {
	case e := <-applied:
		if e.OriginalMessageCount != originalCount {
			t.Errorf("OriginalMessageCount = %d, want %d", e.OriginalMessageCount, originalCount)
		}
		if e.NewMessageCount != len(convo.Messages()) {
			t.Errorf("NewMessageCount = %d, want %d", e.NewMessageCount, len(convo.Messages()))
		}
		if e.NewMessageCount >= e.OriginalMessageCount {
			t.Error("the event reports no compression")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the summary-applied event never fired")
	}
}

func TestOnSummaryAppliedNotFiredOnError(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{MinMessagesAfterSummary: 2},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	fired := make(chan struct{}, 1)
	s.Add(aggregators.EventSummaryApplied, func(context.Context, any, ...any) {
		fired <- struct{}{}
	})

	addUserMessages(convo, 10, "Test message.")
	originalCount := len(convo.Messages())

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	s.ProcessFrame(t.Context(), frames.NewLLMContextSummaryErrorFrame(
		(*got)[0].RequestID, "the model failed"))

	if waitFor(200*time.Millisecond, func() bool { return len(fired) > 0 }) {
		t.Error("the summary-applied event fired for a failed summarization")
	}
	if len(convo.Messages()) != originalCount {
		t.Errorf("a failed summarization changed the conversation: %d messages, want %d",
			len(convo.Messages()), originalCount)
	}
}

func TestRequestFrameCarriesTheTimeout(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig:    frames.SummaryConfig{SummarizationTimeout: 60 * time.Second},
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "Test message to add tokens.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if len(*got) != 1 {
		t.Fatalf("requested %d summarizations, want 1", len(*got))
	}
	if (*got)[0].SummarizationTimeout != 60*time.Second {
		t.Errorf("SummarizationTimeout = %v, want 60s", (*got)[0].SummarizationTimeout)
	}
}

// With the token threshold off, a very large conversation does not summarize
// until the message threshold is crossed.
func TestTokenThresholdOffLeavesOnlyTheMessageThreshold(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        nil,
		MaxUnsummarizedMessages: new(5),
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 3, strings.Repeat("x", 10000))
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 0 {
		t.Fatalf("a huge conversation summarized with the token threshold off")
	}

	addUserMessages(convo, 3, "Message")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if len(*got) != 1 {
		t.Errorf("crossing the message threshold requested %d summarizations, want 1", len(*got))
	}
}

// With the message threshold off, only size summarizes.
func TestMessageThresholdOffLeavesOnlyTheTokenThreshold(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(100),
		MaxUnsummarizedMessages: nil,
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 10, "This is a test message that adds tokens to the context.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if len(*got) != 1 {
		t.Errorf("crossing the token threshold requested %d summarizations, want 1", len(*got))
	}
}

func TestMessageThresholdOffDoesNotTriggerBelowTokens(t *testing.T) {
	convo := summarizerFixture()
	cfg := frames.AutoSummarizationConfig{
		MaxContextTokens:        new(100000),
		MaxUnsummarizedMessages: nil,
	}
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())
	got := recordRequests(s)

	addUserMessages(convo, 30, "Message")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if len(*got) != 0 {
		t.Errorf("summarized %d times below the token threshold with the message one off", len(*got))
	}
}
