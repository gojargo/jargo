package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// Generating a conversation summary: which messages it covers, the ways it
// declines to produce one, and what the service broadcasts for each outcome.

// errProviderRefused is what a test provider returns when it means to fail the
// inference.
var errProviderRefused = errors.New("provider refused")

// summarizingService is shaped like a real provider: it embeds the base and
// hands itself over, which is what lets the base reach the inference it
// defines here.
type summarizingService struct {
	*llm.Base

	summary string
	err     error
	delay   time.Duration

	mu   sync.Mutex
	got  *frames.LLMContext
	opts llm.InferenceOptions
}

func newSummarizingService(summary string) *summarizingService {
	s := &summarizingService{summary: summary}
	s.Base = llm.New("FakeSummarizer", s)
	return s
}

func (s *summarizingService) Generate(_ context.Context, _ *frames.LLMContext, _ llm.Emit) error {
	return nil
}

func (s *summarizingService) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	s.mu.Lock()
	s.got, s.opts = convo, opts
	s.mu.Unlock()

	if s.delay > 0 {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	if s.err != nil {
		return "", s.err
	}
	return s.summary, nil
}

func (s *summarizingService) inference() (*frames.LLMContext, llm.InferenceOptions) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.got, s.opts
}

// plainService runs no inference of its own, which is the service the base
// cannot generate a summary on.
type plainService struct{ *llm.Base }

func newPlainService() *plainService {
	s := &plainService{}
	s.Base = llm.New("PlainLLM", s)
	return s
}

func (s *plainService) Generate(_ context.Context, _ *frames.LLMContext, _ llm.Emit) error {
	return nil
}

// summaryRequest builds a request over a conversation of n messages.
func summaryRequest(id string, messages int) *frames.LLMContextSummaryRequestFrame {
	convo := frames.NewLLMContext("")
	for i := range messages {
		if i%2 == 0 {
			convo.AddUserMessage("message " + string(rune('1'+i)))
		} else {
			convo.AddAssistantMessage("response " + string(rune('1'+i)))
		}
	}
	req := frames.NewLLMContextSummaryRequestFrame(id, convo)
	req.MinMessagesToKeep = 1
	req.TargetContextTokens = 1000
	req.SummarizationPrompt = "Summarize this"
	return req
}

// runSummary puts a summarization request to a service running in a pipeline
// and reports the result it broadcasts.
func runSummary(
	t *testing.T, svc processor.Processor, req *frames.LLMContextSummaryRequestFrame,
) *frames.LLMContextSummaryResultFrame {
	t.Helper()

	var mu sync.Mutex
	var result *frames.LLMContextSummaryResultFrame
	done := make(chan struct{}, 1)

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		fr, ok := f.(*frames.LLMContextSummaryResultFrame)
		if !ok {
			return
		}
		mu.Lock()
		result = fr
		mu.Unlock()
		select {
		case done <- struct{}{}:
		default:
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(req)

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the service broadcast no summary result")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return result
}

// A conversation with nothing left to cover once the messages kept back are
// taken out has no summary to generate.
func TestGenerateSummaryReportsNothingToSummarize(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "System prompt"}})
	req := frames.NewLLMContextSummaryRequestFrame("test", convo)
	req.MinMessagesToKeep = 1

	_, _, err := llm.GenerateSummary(t.Context(), newSummarizingService("unused"), req)

	if !errors.Is(err, llm.ErrNothingToSummarize) {
		t.Errorf("GenerateSummary = %v, want ErrNothingToSummarize", err)
	}
}

// A model that answers the summarization with nothing produced no summary.
func TestGenerateSummaryReportsEmptySummary(t *testing.T) {
	svc := newSummarizingService("")

	_, _, err := llm.GenerateSummary(t.Context(), svc, summaryRequest("test", 3))

	if !errors.Is(err, llm.ErrEmptySummary) {
		t.Errorf("GenerateSummary = %v, want ErrEmptySummary", err)
	}
}

// A provider that fails the inference fails the summary, carrying its reason.
func TestGenerateSummaryReportsTheInferenceFailure(t *testing.T) {
	svc := newSummarizingService("")
	svc.err = errProviderRefused

	_, _, err := llm.GenerateSummary(t.Context(), svc, summaryRequest("test", 3))

	if !errors.Is(err, errProviderRefused) {
		t.Errorf("GenerateSummary = %v, want it to carry the provider's failure", err)
	}
}

func TestGenerateSummarySucceeds(t *testing.T) {
	svc := newSummarizingService("This is a summary of the conversation")

	summary, lastIndex, err := llm.GenerateSummary(t.Context(), svc, summaryRequest("test", 3))
	if err != nil {
		t.Fatalf("GenerateSummary = %v, want a summary", err)
	}

	if summary != "This is a summary of the conversation" {
		t.Errorf("summary = %q, want the model's answer", summary)
	}
	// Three messages keeping one back covers the first two, so the summary runs
	// to the second of them.
	if lastIndex != 1 {
		t.Errorf("last summarized index = %d, want 1", lastIndex)
	}
}

// The summary is generated from a transcript put to the model as a conversation
// of its own, so nothing about the summarization enters the conversation being
// summarized.
func TestGenerateSummaryPutsATranscriptToTheModel(t *testing.T) {
	svc := newSummarizingService("a summary")
	req := summaryRequest("test", 3)

	if _, _, err := llm.GenerateSummary(t.Context(), svc, req); err != nil {
		t.Fatalf("GenerateSummary = %v, want a summary", err)
	}

	convo, opts := svc.inference()
	messages := convo.Messages()
	if len(messages) != 1 || messages[0].Role != frames.RoleUser {
		t.Fatalf("the model was put %d messages %v, want one from the user", len(messages), messages)
	}
	if !strings.HasPrefix(messages[0].Text, "Conversation history:\n") {
		t.Errorf("the model was put %q, want it headed by the transcript", messages[0].Text)
	}
	if !strings.Contains(messages[0].Text, "USER: message 1") {
		t.Errorf("the transcript %q does not carry the conversation", messages[0].Text)
	}
	if opts.SystemInstruction != req.SummarizationPrompt {
		t.Errorf("system instruction = %q, want the request's prompt", opts.SystemInstruction)
	}
	if opts.MaxTokens != req.TargetContextTokens {
		t.Errorf("max tokens = %d, want the request's target of %d", opts.MaxTokens, req.TargetContextTokens)
	}
	// The conversation being summarized is left alone.
	if len(req.Context.Messages()) != 3 {
		t.Errorf("the summarized conversation now holds %d messages, want the 3 it started with",
			len(req.Context.Messages()))
	}
}

// A service that runs no inference of its own cannot generate a summary, and
// says which service it was.
func TestSummaryRequestReportsAServiceThatRunsNoInference(t *testing.T) {
	result := runSummary(t, newPlainService(), summaryRequest("no-inference", 3))

	if result.Error == "" {
		t.Fatal("the service broadcast no error for a summary it cannot generate")
	}
	if !strings.Contains(result.Error, llm.ErrNoInference.Error()) {
		t.Errorf("error = %q, want it to report that no inference runs", result.Error)
	}
	if !strings.Contains(result.Error, "PlainLLM") {
		t.Errorf("error = %q, want it to name the service", result.Error)
	}
}

// The outcome is broadcast against the request that asked for it, and a failed
// generation covers nothing.
func TestSummaryRequestBroadcastsTheFailure(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "System prompt"}})
	req := frames.NewLLMContextSummaryRequestFrame("test_123", convo)
	req.MinMessagesToKeep = 1

	result := runSummary(t, newSummarizingService("unused"), req)

	if result.RequestID != "test_123" {
		t.Errorf("request id = %q, want the one that was asked for", result.RequestID)
	}
	if result.Summary != "" {
		t.Errorf("summary = %q, want none for a failed generation", result.Summary)
	}
	if result.LastSummarizedIndex != -1 {
		t.Errorf("last summarized index = %d, want -1", result.LastSummarizedIndex)
	}
	if !strings.Contains(result.Error, llm.ErrNothingToSummarize.Error()) {
		t.Errorf("error = %q, want it to report there was nothing to summarize", result.Error)
	}
}

// A summarization request reaches the inference the concrete service defines,
// and its summary is broadcast against the request.
func TestSummaryRequestReachesTheServiceInference(t *testing.T) {
	svc := newSummarizingService("a summary of the conversation")

	result := runSummary(t, svc, summaryRequest("req-1", 3))

	if result.Error != "" {
		t.Fatalf("summarization failed: %s", result.Error)
	}
	if result.Summary != "a summary of the conversation" {
		t.Errorf("summary = %q, want the one the service generated", result.Summary)
	}
	if result.LastSummarizedIndex != 1 {
		t.Errorf("last summarized index = %d, want 1", result.LastSummarizedIndex)
	}
	if result.RequestID != "req-1" {
		t.Errorf("request id = %q, want req-1", result.RequestID)
	}
	if convo, _ := svc.inference(); convo == nil {
		t.Error("the service's inference was never reached")
	}
}

// A generation that runs past the request's timeout is given up on, and covers
// nothing. Running out of time is reported to the pipeline on its own, so the
// result it answers with carries no error of its own.
func TestSummaryRequestTimesOut(t *testing.T) {
	svc := newSummarizingService("too late")
	svc.delay = 10 * time.Second

	req := summaryRequest("timeout_test", 3)
	req.SummarizationTimeout = 50 * time.Millisecond

	result := runSummary(t, svc, req)

	if result.RequestID != "timeout_test" {
		t.Errorf("request id = %q, want the one that was asked for", result.RequestID)
	}
	if result.Summary != "" {
		t.Errorf("summary = %q, want none for a generation that timed out", result.Summary)
	}
	if result.LastSummarizedIndex != -1 {
		t.Errorf("last summarized index = %d, want -1", result.LastSummarizedIndex)
	}
	if result.Error != "" {
		t.Errorf("error = %q, want none: the timeout is reported on its own", result.Error)
	}
}
