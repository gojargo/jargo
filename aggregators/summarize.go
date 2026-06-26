package aggregators

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
)

// Summarizer condenses the oldest conversation messages into a compact running
// summary. It receives the prior summary (empty on the first compaction) and the
// messages being dropped, and returns the new summary that stands in for them.
// service/llm.Summarizer adapts any LLM Generator to this contract.
type Summarizer interface {
	Summarize(ctx context.Context, prior string, dropped []frames.Message) (string, error)
}

// SummarizeConfig enables automatic context summarization on the assistant
// aggregator. When the shared context grows past TriggerTokens (estimated),
// older turns are folded into a summary appended to the system prompt while the
// most recent KeepRecentMessages are kept verbatim. Summarization runs in the
// background, so it never adds latency to a turn.
type SummarizeConfig struct {
	// Summarizer produces the summary. Required; a zero Summarizer disables
	// summarization.
	Summarizer Summarizer
	// TriggerTokens is the estimated context size, in tokens, above which older
	// turns are summarized. Zero uses a default tuned for voice sessions.
	TriggerTokens int
	// KeepRecentMessages is how many of the most recent messages stay out of the
	// summary. Zero uses a default; values below the minimum are raised to it.
	KeepRecentMessages int
	// Timeout bounds a single summarization call. Zero uses a default.
	Timeout time.Duration
}

const (
	defaultTriggerTokens    = 4000
	defaultKeepRecent       = 8
	minKeepRecent           = 2
	defaultSummarizeTimeout = 30 * time.Second
)

// WithSummarization enables automatic context summarization on the assistant
// aggregator (see SummarizeConfig):
//
//	sum := llm.NewSummarizer(anthropic.NewLLM(anthropic.Config{}))
//	pair := aggregators.New(ctx, aggregators.WithSummarization(
//	    aggregators.SummarizeConfig{Summarizer: sum}))
func WithSummarization(cfg SummarizeConfig) Option {
	return func(o *options) { o.summarize = &cfg }
}

// summarizer holds the live summarization state for an assistant aggregator. The
// running flag keeps at most one compaction in flight at a time.
type summarizer struct {
	cfg SummarizeConfig

	mu      sync.Mutex
	running bool
}

// newSummarizer applies defaults to cfg. The caller guarantees cfg.Summarizer is
// non-nil.
func newSummarizer(cfg SummarizeConfig) *summarizer {
	if cfg.TriggerTokens <= 0 {
		cfg.TriggerTokens = defaultTriggerTokens
	}
	if cfg.KeepRecentMessages <= 0 {
		cfg.KeepRecentMessages = defaultKeepRecent
	}
	if cfg.KeepRecentMessages < minKeepRecent {
		cfg.KeepRecentMessages = minKeepRecent
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultSummarizeTimeout
	}
	return &summarizer{cfg: cfg}
}

// maybeSummarize compacts the context in the background when it has grown past
// the trigger and no summarization is already in flight. It is called after a
// turn is committed and returns immediately.
func (a *AssistantAggregator) maybeSummarize() {
	s := a.summarize
	if s == nil || a.context.EstimatedTokens() < s.cfg.TriggerTokens {
		return
	}
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.mu.Unlock()
	go a.runSummarize(s)
}

// runSummarize performs one compaction. It runs under a fresh background context
// (not the per-frame context, which is canceled on interruption) bounded by the
// configured timeout, so a barge-in does not abort an in-flight summary.
func (a *AssistantAggregator) runSummarize(s *summarizer) {
	defer func() {
		s.mu.Lock()
		s.running = false
		s.mu.Unlock()
	}()
	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Timeout)
	defer cancel()
	if _, err := a.context.Compact(ctx, s.cfg.KeepRecentMessages, s.cfg.Summarizer.Summarize); err != nil {
		slog.Warn("context summarization failed", "processor", a.Name(), "error", err)
	}
}
