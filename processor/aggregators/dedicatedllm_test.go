package aggregators_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/llm"
)

// fakeInferencer stands in for a model asked to summarize off to the side of the
// pipeline. It records what it was asked and answers with whatever it was told
// to, or fails.
type fakeInferencer struct {
	answer string
	err    error

	mu sync.Mutex
	// entries counts calls that were entered, whether or not they answered. A
	// call abandoned partway through is an entry and not an answer, which is
	// what tells an abandoned attempt from one that never started.
	entries int
	asked   int
	prompts []string
	// gate, when non-nil, holds the answer back until it is closed, so a test
	// can look at the summarizer while a summary is in flight.
	gate chan struct{}
	// exited, when non-nil, is closed as the call returns, after a pause. A
	// caller waiting for the work to stop sees it closed; one that only asked it
	// to stop does not.
	exited chan struct{}
}

func (f *fakeInferencer) RunInference(
	ctx context.Context, convo *frames.LLMContext, _ llm.InferenceOptions,
) (string, error) {
	f.mu.Lock()
	f.entries++
	f.mu.Unlock()
	if f.gate != nil {
		select {
		case <-f.gate:
		case <-ctx.Done():
			if f.exited != nil {
				// Long enough that a Cleanup which only canceled, without
				// waiting, would have returned by now.
				time.Sleep(100 * time.Millisecond)
				close(f.exited)
			}
			return "", ctx.Err()
		}
	}
	f.mu.Lock()
	f.asked++
	if msgs := convo.Messages(); len(msgs) > 0 {
		f.prompts = append(f.prompts, msgs[len(msgs)-1].Text)
	}
	f.mu.Unlock()
	return f.answer, f.err
}

// timesEntered is how many calls were begun, answered or not.
func (f *fakeInferencer) timesEntered() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.entries
}

func (f *fakeInferencer) timesAsked() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.asked
}

func (f *fakeInferencer) lastPrompt() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.prompts) == 0 {
		return ""
	}
	return f.prompts[len(f.prompts)-1]
}

// errModelUnavailable is what a summarization model that cannot answer reports.
//
//nolint:gochecknoglobals // sentinel error for the tests below
var errModelUnavailable = errors.New("the model is unavailable")

// dedicatedConfig is the auto-summarization configuration that routes the work
// to inf rather than to the pipeline's own model.
func dedicatedConfig(inf llm.Inferencer) frames.AutoSummarizationConfig {
	return frames.AutoSummarizationConfig{
		MaxContextTokens: new(50),
		SummaryConfig: frames.SummaryConfig{
			MinMessagesAfterSummary: 2,
			LLM:                     inf,
		},
	}
}

// TestDedicatedLLMSummarizesWithoutThePipeline checks the point of configuring a
// summarization model of its own: the summary is produced on that model and
// applied, and the pipeline is never asked. It is how the work is routed to a
// cheaper model than the one carrying the conversation.
func TestDedicatedLLMSummarizesWithoutThePipeline(t *testing.T) {
	convo := summarizerFixture()
	inf := &fakeInferencer{answer: "They discussed the weather."}
	s := aggregators.NewSummarizer(convo, dedicatedConfig(inf), true)
	defer s.Cleanup(t.Context())
	requests := recordRequests(s)

	addUserMessages(convo, 10, "Test message.")
	originalCount := len(convo.Messages())
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if !waitFor(3*time.Second, func() bool { return len(convo.Messages()) < originalCount }) {
		t.Fatalf("the conversation still holds %d messages, want the summary applied",
			len(convo.Messages()))
	}
	if inf.timesAsked() != 1 {
		t.Errorf("the dedicated model was asked %d times, want 1", inf.timesAsked())
	}
	if len(*requests) != 0 {
		t.Errorf("the pipeline was asked to summarize %d times, want none: the "+
			"dedicated model answers directly", len(*requests))
	}

	var summaries int
	for _, m := range convo.Messages() {
		if strings.Contains(m.Text, "They discussed the weather.") {
			summaries++
		}
	}
	if summaries != 1 {
		t.Errorf("the conversation holds %d copies of the summary, want 1", summaries)
	}
}

// TestDedicatedLLMIsGivenTheTranscript checks the model is asked to summarize
// the conversation rather than handed an empty request.
func TestDedicatedLLMIsGivenTheTranscript(t *testing.T) {
	convo := summarizerFixture()
	inf := &fakeInferencer{answer: "A summary."}
	s := aggregators.NewSummarizer(convo, dedicatedConfig(inf), true)
	defer s.Cleanup(t.Context())

	addUserMessages(convo, 10, "The train leaves at nine.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if !waitFor(3*time.Second, func() bool { return inf.timesAsked() == 1 }) {
		t.Fatal("the dedicated model was never asked")
	}
	if got := inf.lastPrompt(); !strings.Contains(got, "The train leaves at nine.") {
		t.Errorf("the model was asked %q, want the conversation to summarize", got)
	}
}

// TestDedicatedLLMFailureLeavesTheConversationAlone checks a model that cannot
// answer costs nothing: the conversation is left as it stands rather than being
// truncated against a summary that does not exist.
func TestDedicatedLLMFailureLeavesTheConversationAlone(t *testing.T) {
	convo := summarizerFixture()
	inf := &fakeInferencer{err: errModelUnavailable}
	s := aggregators.NewSummarizer(convo, dedicatedConfig(inf), true)
	defer s.Cleanup(t.Context())

	addUserMessages(convo, 10, "Test message.")
	before := len(convo.Messages())
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if !waitFor(3*time.Second, func() bool { return inf.timesAsked() == 1 }) {
		t.Fatal("the dedicated model was never asked")
	}
	// Give the failure every chance to do damage before concluding it did none.
	time.Sleep(200 * time.Millisecond)
	if got := len(convo.Messages()); got != before {
		t.Errorf("the conversation holds %d messages, want the original %d", got, before)
	}
}

// TestDedicatedLLMFailureReleasesTheNextAttempt checks a failed summary does not
// leave the summarizer thinking one is still in flight, which would stop it ever
// trying again.
func TestDedicatedLLMFailureReleasesTheNextAttempt(t *testing.T) {
	convo := summarizerFixture()
	inf := &fakeInferencer{err: errModelUnavailable}
	s := aggregators.NewSummarizer(convo, dedicatedConfig(inf), true)
	defer s.Cleanup(t.Context())

	addUserMessages(convo, 10, "Test message.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if !waitFor(3*time.Second, func() bool { return inf.timesAsked() == 1 }) {
		t.Fatal("the dedicated model was never asked")
	}

	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if !waitFor(3*time.Second, func() bool { return inf.timesAsked() == 2 }) {
		t.Errorf("the model was asked %d times, want the failed attempt to have "+
			"released the next one", inf.timesAsked())
	}
}

// TestDedicatedLLMCleanupStopsTheSummaryAndWaitsForIt checks teardown does both
// halves: the in-flight summary is canceled rather than left to run to
// completion against a pipeline that is going away, and Cleanup returns only
// once the goroutine doing it has actually stopped, so none outlives the
// pipeline that started it.
func TestDedicatedLLMCleanupStopsTheSummaryAndWaitsForIt(t *testing.T) {
	convo := summarizerFixture()
	inf := &fakeInferencer{
		answer: "A summary.",
		// Never closed: the summary only ends by being canceled.
		gate:   make(chan struct{}),
		exited: make(chan struct{}),
	}
	s := aggregators.NewSummarizer(convo, dedicatedConfig(inf), true)

	addUserMessages(convo, 10, "Test message.")
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if !waitFor(3*time.Second, func() bool { return inf.timesEntered() == 1 }) {
		t.Fatal("the dedicated model was never asked")
	}

	done := make(chan struct{})
	go func() {
		s.Cleanup(context.Background())
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Cleanup never returned: the summary was not canceled")
	}

	select {
	case <-inf.exited:
	default:
		t.Error("Cleanup returned while the summary goroutine was still running")
	}
}

// TestDedicatedLLMTimeoutLeavesTheConversationAlone is upstream's
// test_dedicated_llm_timeout: a model that takes too long is abandoned, the
// conversation is left as it stands, and the state is cleared so the next turn
// may try again.
func TestDedicatedLLMTimeoutLeavesTheConversationAlone(t *testing.T) {
	convo := summarizerFixture()
	// Never answers on its own: only the timeout ends it.
	inf := &fakeInferencer{answer: "A summary.", gate: make(chan struct{})}
	cfg := dedicatedConfig(inf)
	cfg.SummaryConfig.SummarizationTimeout = 100 * time.Millisecond
	s := aggregators.NewSummarizer(convo, cfg, true)
	defer s.Cleanup(t.Context())

	addUserMessages(convo, 10, "Test message.")
	before := len(convo.Messages())
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())

	if !waitFor(3*time.Second, func() bool { return inf.timesEntered() == 1 }) {
		t.Fatal("the dedicated model was never asked")
	}
	// Past the timeout, with room for the abandonment to do damage if it were
	// going to.
	time.Sleep(400 * time.Millisecond)

	if got := len(convo.Messages()); got != before {
		t.Errorf("the conversation holds %d messages, want the original %d: a "+
			"summary that timed out must not be applied", got, before)
	}

	// The abandoned attempt was entered and never answered, so nothing has been
	// asked yet.
	if got := inf.timesAsked(); got != 0 {
		t.Errorf("the model answered %d times, want 0: the attempt was abandoned", got)
	}

	// The state is cleared, so the next turn summarizes again. A second entry is
	// what shows it: a summarizer that still thought one was in flight would
	// refuse, and a timeout that never fired would leave the first attempt
	// holding the gate rather than starting another.
	close(inf.gate)
	s.ProcessFrame(t.Context(), frames.NewLLMFullResponseStartFrame())
	if !waitFor(3*time.Second, func() bool { return inf.timesEntered() == 2 }) {
		t.Errorf("the model was entered %d times, want 2: the timed-out attempt "+
			"left the summarizer thinking one was still in flight", inf.timesEntered())
	}
}
