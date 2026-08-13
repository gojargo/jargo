package aggregators

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
	"github.com/google/uuid"
)

// The events a Summarizer raises.
const (
	// EventRequestSummarization fires when a summary is needed and no dedicated
	// LLM was configured to produce it. Its argument is the
	// *frames.LLMContextSummaryRequestFrame to put to the pipeline's LLM. It is
	// synchronous: the summarizer has marked the request in flight and is waiting
	// for it to be sent.
	EventRequestSummarization = "on_request_summarization"
	// EventSummaryApplied fires once a summary has been written into the
	// conversation. Its argument is a SummaryApplied describing the compression.
	EventSummaryApplied = "on_summary_applied"
)

// SummaryApplied describes a compression that has been applied to the
// conversation.
type SummaryApplied struct {
	// OriginalMessageCount is how many messages the conversation held before.
	OriginalMessageCount int
	// NewMessageCount is how many it holds after.
	NewMessageCount int
	// SummarizedMessageCount is how many were folded into the summary.
	SummarizedMessageCount int
	// PreservedMessageCount is how many survived uncompressed, counting a
	// preserved leading system message.
	PreservedMessageCount int
}

// Summarizer compresses a conversation that has grown long, either
// automatically once a threshold is passed or on demand when an
// LLMSummarizeContextFrame arrives. It watches the conversation, asks for a
// summary, and applies the result.
//
// It is not a processor. The assistant aggregator owns one and hands it every
// frame, which is what lets the compression happen beside the conversation
// rather than in the path of it.
//
// A summary is produced one of two ways. With a dedicated LLM configured it is
// generated directly, off the pipeline entirely. Without one, the summarizer
// raises EventRequestSummarization and the owner puts the request to the
// pipeline's own LLM, whose answer comes back as an
// LLMContextSummaryResultFrame.
type Summarizer struct {
	events.Registry

	context    *frames.LLMContext
	autoConfig frames.AutoSummarizationConfig
	// autoTrigger gates the threshold checks only. The summarizer exists whether
	// or not it is set, so an on-demand LLMSummarizeContextFrame is always
	// honored.
	autoTrigger bool

	mu sync.Mutex
	// inProgress is set from the moment a summary is asked for until its result
	// arrives, so only one summarization is ever in flight.
	inProgress bool
	// pendingRequestID identifies the summary being waited on, so a result for a
	// request that has since been abandoned is recognized and dropped.
	pendingRequestID string

	// Goroutines for the dedicated-LLM path, waited on by Cleanup.
	taskCtx    context.Context
	taskCancel context.CancelFunc
	taskWG     sync.WaitGroup
}

// NewSummarizer builds a Summarizer over convo. autoTrigger enables the
// threshold checks; leave it false to compress only when asked with an
// LLMSummarizeContextFrame.
func NewSummarizer(convo *frames.LLMContext, cfg frames.AutoSummarizationConfig, autoTrigger bool) *Summarizer {
	s := &Summarizer{
		context:     convo,
		autoConfig:  cfg.WithDefaults(),
		autoTrigger: autoTrigger,
	}
	s.taskCtx, s.taskCancel = context.WithCancel(context.Background())
	s.Register(EventRequestSummarization, true)
	s.Register(EventSummaryApplied, false)
	return s
}

// ProcessFrame lets the summarizer watch the conversation. The owner calls it
// for every frame, after the frame has been forwarded.
func (s *Summarizer) ProcessFrame(ctx context.Context, f frames.Frame) {
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		s.handleResponseStart(ctx)
	case *frames.LLMSummarizeContextFrame:
		s.handleManualRequest(ctx, fr)
	case *frames.LLMContextSummaryResultFrame:
		s.handleSummaryResult(ctx, fr)
	case *frames.InterruptionFrame:
		s.handleInterruption()
	}
}

// Cleanup abandons any summarization in flight and waits for the goroutines of
// the dedicated-LLM path and of the asynchronous events.
func (s *Summarizer) Cleanup(ctx context.Context) {
	s.taskCancel()
	s.taskWG.Wait()
	s.Registry.Cleanup(ctx)
	s.clearState()
}

// handleResponseStart checks the thresholds as each turn begins.
func (s *Summarizer) handleResponseStart(ctx context.Context) {
	if s.shouldSummarize() {
		s.requestSummarization(ctx, nil)
	}
}

// handleManualRequest compresses on demand, whatever the thresholds say. It
// takes the same path as an automatic request, so the bookkeeping is identical.
func (s *Summarizer) handleManualRequest(ctx context.Context, f *frames.LLMSummarizeContextFrame) {
	s.mu.Lock()
	busy := s.inProgress
	s.mu.Unlock()
	if busy {
		slog.Debug("summarization already in progress, ignoring the manual request")
		return
	}
	s.requestSummarization(ctx, f.Config)
}

// handleInterruption releases the in-flight marker so a new summarization can
// start. The pending request id is deliberately kept: the request frame may have
// been canceled by the interruption, but the result frame is uninterruptible and
// still arrives, and it has to be recognized when it does.
func (s *Summarizer) handleInterruption() {
	s.mu.Lock()
	s.inProgress = false
	s.mu.Unlock()
}

// clearState abandons the summarization in flight, if any.
func (s *Summarizer) clearState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inProgress {
		slog.Debug("clearing the pending summarization")
		s.inProgress = false
		s.pendingRequestID = ""
	}
}

// shouldSummarize reports whether the conversation has grown past either
// threshold. Both can be read: a conversation crosses whichever comes first.
func (s *Summarizer) shouldSummarize() bool {
	if !s.autoTrigger {
		return false
	}

	s.mu.Lock()
	busy := s.inProgress
	s.mu.Unlock()
	if busy {
		slog.Debug("summarization already in progress")
		return false
	}

	totalTokens := uctx.EstimateContextTokens(s.context)
	messages := len(s.context.Messages())

	tokenLimit := s.autoConfig.MaxContextTokens
	tokenLimitExceeded := tokenLimit != nil && totalTokens >= *tokenLimit

	// One message is discounted: the summary written by the previous compression
	// is itself a message, and counting it would shorten every window after the
	// first.
	sinceSummary := messages - 1
	messageThreshold := s.autoConfig.MaxUnsummarizedMessages
	messageThresholdExceeded := messageThreshold != nil && sinceSummary >= *messageThreshold

	if !tokenLimitExceeded && !messageThresholdExceeded {
		return false
	}

	slog.Debug("summarizing the conversation",
		"tokens", totalTokens, "token_limit", tokenLimit,
		"messages_since_summary", sinceSummary, "message_threshold", messageThreshold)
	return true
}

// requestSummarization asks for a summary, by the dedicated LLM when one is
// configured and by the pipeline's LLM otherwise. override, when non-nil,
// replaces the configured summary settings for this request alone.
func (s *Summarizer) requestSummarization(ctx context.Context, override *frames.SummaryConfig) {
	cfg := s.autoConfig.SummaryConfig
	if override != nil {
		cfg = override.WithDefaults()
	}

	requestID := uuid.NewString()

	s.mu.Lock()
	s.inProgress = true
	s.pendingRequestID = requestID
	s.mu.Unlock()

	slog.Debug("requesting a conversation summary", "request_id", requestID)

	request := frames.NewLLMContextSummaryRequestFrame(requestID, s.context)
	request.MinMessagesToKeep = cfg.MinMessagesAfterSummary
	request.TargetContextTokens = cfg.TargetContextTokens
	request.SummarizationPrompt = cfg.SummaryPrompt()
	request.SummarizationTimeout = cfg.SummarizationTimeout

	if inf, ok := cfg.LLM.(llm.Inferencer); ok {
		// A dedicated LLM answers directly, so the pipeline is never involved.
		s.taskWG.Go(func() { s.generateWithDedicatedLLM(inf, request) })
		return
	}
	if cfg.LLM != nil {
		slog.Error("the configured summarization LLM does not run inference; "+
			"putting the request to the pipeline's LLM instead",
			"type", fmt.Sprintf("%T", cfg.LLM))
	}

	s.Call(ctx, EventRequestSummarization, s, request)
}

// generateWithDedicatedLLM produces the summary on the configured LLM and feeds
// the outcome back through the same path a pipeline-generated one takes, so a
// summary is applied identically however it was produced.
func (s *Summarizer) generateWithDedicatedLLM(inf llm.Inferencer, request *frames.LLMContextSummaryRequestFrame) {
	timeout := request.SummarizationTimeout
	if timeout == 0 {
		timeout = frames.DefaultSummarizationTimeout
	}
	ctx, cancel := context.WithTimeout(s.taskCtx, timeout)
	defer cancel()

	summary, lastIndex, err := llm.GenerateSummary(ctx, inf, request)
	if err != nil {
		slog.Error("generating the conversation summary failed", "error", err)
		s.handleSummaryResult(s.taskCtx, frames.NewLLMContextSummaryErrorFrame(request.RequestID, err.Error()))
		return
	}
	s.handleSummaryResult(s.taskCtx, frames.NewLLMContextSummaryResultFrame(request.RequestID, summary, lastIndex))
}

// handleSummaryResult applies a generated summary, having checked that it is
// the one being waited on and that the conversation has not moved on beneath it.
func (s *Summarizer) handleSummaryResult(ctx context.Context, f *frames.LLMContextSummaryResultFrame) {
	slog.Debug("received a summary result", "request_id", f.RequestID)

	s.mu.Lock()
	pending := s.pendingRequestID
	s.mu.Unlock()
	if f.RequestID != pending {
		slog.Debug("ignoring a stale summary result", "request_id", f.RequestID)
		return
	}

	s.clearState()

	if f.Error != "" {
		slog.Error("conversation summarization failed", "error", f.Error)
		return
	}

	if !s.validateSummaryContext(f.LastSummarizedIndex) {
		slog.Warn("the conversation changed while the summary was being generated, not applying it")
		return
	}

	s.applySummary(ctx, f.Summary, f.LastSummarizedIndex)
}

// validateSummaryContext reports whether the conversation still matches the
// summary that was generated for it. A turn landing while the model was
// summarizing can leave the index pointing somewhere else, and applying the
// summary then would drop messages it does not cover.
func (s *Summarizer) validateSummaryContext(lastSummarizedIndex int) bool {
	if lastSummarizedIndex < 0 {
		return false
	}
	messages := s.context.Messages()
	if lastSummarizedIndex >= len(messages) {
		return false
	}
	minKeep := s.autoConfig.SummaryConfig.MinMessagesAfterSummary
	remaining := len(messages) - 1 - lastSummarizedIndex
	return remaining >= minKeep
}

// applySummary rewrites the conversation as the preserved system message, the
// summary, and the messages after the summarized range.
//
// The summary is written as a user message: it is context supplied to the
// assistant, not something the assistant said.
func (s *Summarizer) applySummary(ctx context.Context, summary string, lastSummarizedIndex int) {
	cfg := s.autoConfig.SummaryConfig
	messages := s.context.Messages()

	// Only a system message at the head of the list is the preamble. One
	// anywhere else is a mid-conversation injection, and belongs to whichever
	// side of the cut it fell on.
	systemPreserved := 0
	rebuilt := make([]frames.Message, 0, len(messages))
	if len(messages) > 0 && !messages[0].IsLLMSpecific() && messages[0].Role == frames.RoleSystem {
		rebuilt = append(rebuilt, messages[0])
		systemPreserved = 1
	}

	recent := messages[lastSummarizedIndex+1:]

	rebuilt = append(rebuilt, frames.Message{
		Role: frames.RoleUser,
		Text: strings.ReplaceAll(cfg.SummaryMessageTemplate, "{summary}", summary),
	})
	rebuilt = append(rebuilt, recent...)

	originalCount := len(messages)
	s.context.SetMessages(rebuilt)

	summarized := lastSummarizedIndex + 1 - systemPreserved

	slog.Info("applied a conversation summary",
		"summarized", summarized, "messages_before", originalCount, "messages_after", len(rebuilt))

	s.Call(ctx, EventSummaryApplied, s, SummaryApplied{
		OriginalMessageCount:   originalCount,
		NewMessageCount:        len(rebuilt),
		SummarizedMessageCount: summarized,
		PreservedMessageCount:  len(recent) + systemPreserved,
	})
}
