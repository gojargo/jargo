package turns

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Turn-completion markers the LLM is instructed to prefix its reply with.
const (
	// MarkerComplete means the user's turn was complete; respond normally.
	MarkerComplete = "✓"
	// markerIncompleteShort means the turn was likely incomplete (a short wait).
	markerIncompleteShort = "○"
	// markerIncompleteLong means the turn was clearly incomplete (a longer wait).
	markerIncompleteLong = "◐"
)

const (
	defaultIncompleteShortTimeout = 5 * time.Second
	defaultIncompleteLongTimeout  = 10 * time.Second
)

// defaultCompletionInstructions is the system-prompt block that teaches the LLM
// the marker protocol. Prepend it (or your own via Config.Instructions) to the
// system prompt when using turn-completion gating.
const defaultCompletionInstructions = "Before every reply, judge whether the user's most recent turn is a complete " +
	"thought you should answer now. Begin your reply with exactly one marker and a space:\n" +
	"  ✓  the turn is complete — answer normally after the marker.\n" +
	"  ○  the turn seems incomplete — the user likely has a little more to say; output only this marker.\n" +
	"  ◐  the turn is clearly cut off mid-thought; output only this marker.\n" +
	"When you output ○ or ◐, write nothing else."

const (
	defaultIncompleteShortPrompt = "The user did not continue. Respond to what they have said so far."
	defaultIncompleteLongPrompt  = "The user did not continue. Respond to what they have said so far."
)

// UserTurnCompletionConfig configures LLM turn-completion gating.
type UserTurnCompletionConfig struct {
	// Instructions overrides the marker-protocol system-prompt block.
	Instructions string
	// IncompleteShortTimeout is how long to wait after a "○" before re-prompting;
	// 0 uses 5s.
	IncompleteShortTimeout time.Duration
	// IncompleteLongTimeout is how long to wait after a "◐"; 0 uses 10s.
	IncompleteLongTimeout time.Duration
	// IncompleteShortPrompt / IncompleteLongPrompt are the re-prompt messages.
	IncompleteShortPrompt string
	IncompleteLongPrompt  string
}

func (c UserTurnCompletionConfig) withDefaults() UserTurnCompletionConfig {
	if c.IncompleteShortTimeout == 0 {
		c.IncompleteShortTimeout = defaultIncompleteShortTimeout
	}
	if c.IncompleteLongTimeout == 0 {
		c.IncompleteLongTimeout = defaultIncompleteLongTimeout
	}
	if c.IncompleteShortPrompt == "" {
		c.IncompleteShortPrompt = defaultIncompleteShortPrompt
	}
	if c.IncompleteLongPrompt == "" {
		c.IncompleteLongPrompt = defaultIncompleteLongPrompt
	}
	return c
}

// CompletionInstructions returns the marker-protocol instructions to prepend to
// the LLM system prompt for turn-completion gating.
func CompletionInstructions(cfg UserTurnCompletionConfig) string {
	if cfg.Instructions != "" {
		return cfg.Instructions
	}
	return defaultCompletionInstructions
}

// CompletionFilter parses the LLM's turn-completion markers in its streamed
// output. On "✓" it broadcasts a UserTurnInferenceCompletedFrame (which a
// FilterIncompleteUserTurnStrategies finalizer turns into the user's turn end)
// and forwards the reply; on "○"/"◐" it suppresses the reply and, after a
// timeout, re-prompts the LLM. Place it immediately after the LLM service.
type CompletionFilter struct {
	*processor.Base
	cfg UserTurnCompletionConfig

	mu            sync.Mutex
	ctx           context.Context
	buffer        string
	completeFound bool
	suppressed    bool
	broadcasted   bool
	cancelTimer   func()
}

// NewCompletionFilter builds a turn-completion filter.
func NewCompletionFilter(cfg UserTurnCompletionConfig) *CompletionFilter {
	f := &CompletionFilter{cfg: cfg.withDefaults()}
	f.Base = processor.New("UserTurnCompletion", f)
	return f
}

// Setup records the session context for re-prompts.
func (f *CompletionFilter) Setup(ctx context.Context, s processor.Setup) error {
	if err := f.Base.Setup(ctx, s); err != nil {
		return err
	}
	f.mu.Lock()
	f.ctx = ctx
	f.mu.Unlock()
	return nil
}

// Cleanup stops any pending re-prompt timer.
func (f *CompletionFilter) Cleanup(ctx context.Context) error {
	f.mu.Lock()
	f.cancel()
	f.mu.Unlock()
	return f.Base.Cleanup(ctx)
}

// ProcessFrame parses markers in the LLM output stream.
func (f *CompletionFilter) ProcessFrame(ctx context.Context, fr frames.Frame, dir processor.Direction) error {
	if err := f.Base.ProcessFrame(ctx, fr, dir); err != nil {
		return err
	}
	switch v := fr.(type) {
	case *frames.LLMFullResponseStartFrame:
		f.resetTurn()
		return f.PushFrame(ctx, fr, dir)
	case *frames.LLMTextFrame:
		return f.handleText(ctx, v, dir)
	case *frames.LLMFullResponseEndFrame:
		f.flush(ctx)
		return f.PushFrame(ctx, fr, dir)
	case *frames.InterruptionFrame:
		f.mu.Lock()
		f.cancel()
		f.mu.Unlock()
		f.resetTurn()
		return f.PushFrame(ctx, fr, dir)
	case *frames.FunctionCallsStartedFrame:
		// A tool call commits the turn before any marker.
		f.broadcastCompletion(ctx)
		return f.PushFrame(ctx, fr, dir)
	default:
		return f.PushFrame(ctx, fr, dir)
	}
}

// handleText buffers streamed text until a marker is found, then acts on it.
func (f *CompletionFilter) handleText(ctx context.Context, frame *frames.LLMTextFrame, dir processor.Direction) error {
	f.mu.Lock()
	if f.suppressed {
		f.mu.Unlock()
		return nil // dropped: the model kept talking after an incomplete marker
	}
	if f.completeFound {
		f.mu.Unlock()
		return f.PushFrame(ctx, frame, dir) // past the marker: pass through
	}
	f.buffer += frame.Text

	// Short before long, matching Pipecat's precedence when both appear.
	if strings.Contains(f.buffer, markerIncompleteShort) {
		return f.onIncomplete(ctx, markerIncompleteShort, f.cfg.IncompleteShortTimeout, f.cfg.IncompleteShortPrompt)
	}
	if strings.Contains(f.buffer, markerIncompleteLong) {
		return f.onIncomplete(ctx, markerIncompleteLong, f.cfg.IncompleteLongTimeout, f.cfg.IncompleteLongPrompt)
	}
	if i := strings.Index(f.buffer, MarkerComplete); i >= 0 {
		rest := strings.TrimPrefix(f.buffer[i+len(MarkerComplete):], " ")
		f.completeFound = true
		f.buffer = ""
		f.mu.Unlock()
		f.broadcastCompletion(ctx)
		_ = f.PushFrame(ctx, frames.NewLLMMarkerFrame(MarkerComplete), processor.Downstream)
		if rest != "" {
			return f.PushFrame(ctx, frames.NewLLMTextFrame(rest), dir)
		}
		return nil
	}
	f.mu.Unlock()
	return nil // still buffering, no marker yet
}

// onIncomplete suppresses the reply and arms the re-prompt. It is called with the
// mutex held and unlocks it.
func (f *CompletionFilter) onIncomplete(
	ctx context.Context, marker string, timeout time.Duration, prompt string,
) error {
	f.suppressed = true
	f.buffer = ""
	f.startReprompt(timeout, prompt)
	f.mu.Unlock()
	return f.PushFrame(ctx, frames.NewLLMMarkerFrame(marker), processor.Downstream)
}

// startReprompt arms the re-prompt timer. It runs with the mutex held.
func (f *CompletionFilter) startReprompt(timeout time.Duration, prompt string) {
	f.cancel()
	ctx := f.ctx
	stopped := false
	timer := time.AfterFunc(timeout, func() {
		f.mu.Lock()
		if stopped {
			f.mu.Unlock()
			return
		}
		f.cancelTimer = nil
		f.suppressed = false
		f.mu.Unlock()
		msg := []frames.Message{{Role: frames.RoleUser, Text: prompt}}
		_ = f.PushFrame(ctx, frames.NewLLMMessagesAppendFrame(msg), processor.Upstream)
		_ = f.PushFrame(ctx, frames.NewLLMRunFrame(), processor.Upstream)
	})
	f.cancelTimer = func() {
		stopped = true
		timer.Stop()
	}
}

// flush forwards any buffered text when the response ends with no marker found —
// graceful degradation for a model that ignored the protocol.
func (f *CompletionFilter) flush(ctx context.Context) {
	f.mu.Lock()
	buf := f.buffer
	complete, supp := f.completeFound, f.suppressed
	f.buffer = ""
	f.mu.Unlock()
	if !complete && !supp && strings.TrimSpace(buf) != "" {
		_ = f.PushFrame(ctx, frames.NewLLMTextFrame(buf), processor.Downstream)
	}
}

// broadcastCompletion emits the completion signal once per turn.
func (f *CompletionFilter) broadcastCompletion(ctx context.Context) {
	f.mu.Lock()
	if f.broadcasted {
		f.mu.Unlock()
		return
	}
	f.broadcasted = true
	f.mu.Unlock()
	_ = f.PushFrame(ctx, frames.NewUserTurnInferenceCompletedFrame(), processor.Downstream)
	_ = f.PushFrame(ctx, frames.NewUserTurnInferenceCompletedFrame(), processor.Upstream)
}

func (f *CompletionFilter) resetTurn() {
	f.mu.Lock()
	f.cancel()
	f.buffer = ""
	f.completeFound = false
	f.suppressed = false
	f.broadcasted = false
	f.mu.Unlock()
}

// cancel stops the re-prompt timer. It runs with the mutex held.
func (f *CompletionFilter) cancel() {
	if f.cancelTimer != nil {
		f.cancelTimer()
		f.cancelTimer = nil
	}
}

// NewLLMTurnCompletionStop builds the stop strategy that finalizes a turn when
// the CompletionFilter broadcasts a completion. Pair it with deferred detectors
// via FilterIncompleteUserTurnStrategies.
func NewLLMTurnCompletionStop() StopStrategy { return NewExternalCompletionStop() }

// FilterIncompleteUserTurnStrategies builds a stop chain where the detectors only
// trigger LLM inference (they are deferred) and final turn-completion is decided
// by the LLM marker protocol. Pass your detector stop strategies (e.g.
// Smart-Turn); empty uses the defaults. Use with a CompletionFilter after the LLM
// and CompletionInstructions in the system prompt.
func FilterIncompleteUserTurnStrategies(detectors []StopStrategy) UserTurnStrategies {
	if len(detectors) == 0 {
		detectors = DefaultStopStrategies()
	}
	stop := make([]StopStrategy, 0, len(detectors)+1)
	for _, d := range detectors {
		stop = append(stop, Deferred(d))
	}
	stop = append(stop, NewLLMTurnCompletionStop())
	return UserTurnStrategies{Stop: stop}
}
