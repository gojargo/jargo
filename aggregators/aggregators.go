// Package aggregators assembles the conversation around an LLM. The user
// aggregator collects transcriptions into a user message and triggers the LLM;
// the assistant aggregator collects the streamed response into an assistant
// message. Both share one LLMContext, so the conversation accrues across turns.
//
// Place the user aggregator before the LLM and the assistant aggregator at the
// end of the pipeline:
//
//	pipeline.New(input, stt, agg.User(), llm, tts, output, agg.Assistant())
//
// By default the user turn ends when the STT service finalizes a transcription.
// With WithTurnTaking, the turn instead ends when a turntaking.Detector reports
// end-of-turn (a UserStoppedSpeakingFrame), gated on having a finalized
// transcript — so a Smart Turn model, not STT endpointing, decides when the bot
// responds. Add the turntaking.Detector right after the input transport:
//
//	pipeline.New(input, detector, stt, agg.User(), llm, tts, output, agg.Assistant())
package aggregators

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Pair is a user and assistant aggregator sharing one conversation context.
type Pair struct {
	context   *frames.LLMContext
	user      *UserAggregator
	assistant *AssistantAggregator
}

// Option configures an aggregator Pair.
type Option func(*options)

type options struct {
	turnTaking bool
}

// WithTurnTaking gates the user turn on end-of-turn detection: the LLM runs when
// a turntaking.Detector reports the turn complete (UserStoppedSpeakingFrame) and
// a finalized transcript is in hand, rather than on STT finalization alone.
func WithTurnTaking() Option {
	return func(o *options) { o.turnTaking = true }
}

// New builds a user/assistant aggregator pair around ctx.
func New(ctx *frames.LLMContext, opts ...Option) *Pair {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	return &Pair{
		context:   ctx,
		user:      newUser(ctx, o.turnTaking),
		assistant: newAssistant(ctx),
	}
}

// User returns the user-side aggregator.
func (p *Pair) User() processor.Processor { return p.user }

// Assistant returns the assistant-side aggregator.
func (p *Pair) Assistant() processor.Processor { return p.assistant }

// Context returns the shared conversation context.
func (p *Pair) Context() *frames.LLMContext { return p.context }

// UserAggregator collects transcriptions into a user message and, when the
// user's turn ends, appends it to the context and triggers the LLM with an
// LLMContextFrame. With turn taking enabled it also tracks speaking and
// end-of-turn frames; those are system frames handled on a different goroutine
// than transcriptions, so the aggregation state is mutex-guarded.
type UserAggregator struct {
	*processor.Base
	context    *frames.LLMContext
	turnTaking bool

	mu           sync.Mutex
	aggregation  string
	turnComplete bool // turn taking: end-of-turn reported
	haveFinal    bool // turn taking: a finalized transcript has arrived this turn
}

func newUser(ctx *frames.LLMContext, turnTaking bool) *UserAggregator {
	u := &UserAggregator{context: ctx, turnTaking: turnTaking}
	u.Base = processor.New("UserContextAggregator", u)
	return u
}

// ProcessFrame collects transcriptions and triggers the LLM.
func (u *UserAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := u.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch fr := f.(type) {
	case *frames.UserStartedSpeakingFrame:
		// A new turn begins; drop any stale aggregation from a prior turn.
		if u.turnTaking {
			u.mu.Lock()
			u.aggregation = ""
			u.turnComplete = false
			u.haveFinal = false
			u.mu.Unlock()
		}
		return u.PushFrame(ctx, f, dir)

	case *frames.UserStoppedSpeakingFrame:
		if err := u.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		if u.turnTaking {
			u.mu.Lock()
			u.turnComplete = true
			u.mu.Unlock()
			return u.maybeRun(ctx)
		}
		return nil

	case *frames.InterimTranscriptionFrame:
		// Fresh partial speech: a finalized transcript for this turn is no
		// longer the last word, so wait for the next one before responding.
		if u.turnTaking {
			u.mu.Lock()
			u.haveFinal = false
			u.mu.Unlock()
		}
		return u.PushFrame(ctx, f, dir)

	case *frames.TranscriptionFrame:
		return u.handleTranscription(ctx, fr, dir)

	default:
		return u.PushFrame(ctx, f, dir)
	}
}

func (u *UserAggregator) handleTranscription(
	ctx context.Context, fr *frames.TranscriptionFrame, dir processor.Direction,
) error {
	u.mu.Lock()
	if fr.Text != "" {
		if u.aggregation != "" {
			u.aggregation += " "
		}
		u.aggregation += fr.Text
	}
	if fr.Finalized {
		u.haveFinal = true
	}
	u.mu.Unlock()

	// Forward the transcription so downstream processors (e.g. RTVI) see it.
	if err := u.PushFrame(ctx, fr, dir); err != nil {
		return err
	}

	if u.turnTaking {
		return u.maybeRun(ctx)
	}
	// Default: STT finalization marks the end of the user's turn.
	if fr.Finalized {
		return u.maybeRun(ctx)
	}
	return nil
}

// maybeRun commits the aggregated user message and triggers the LLM when the
// turn-completion conditions hold. With turn taking, that means an end-of-turn
// was reported and a finalized transcript is in hand; without it, a finalized
// transcript alone suffices.
func (u *UserAggregator) maybeRun(ctx context.Context) error {
	u.mu.Lock()
	ready := u.aggregation != ""
	if u.turnTaking {
		ready = ready && u.turnComplete && u.haveFinal
	}
	if !ready {
		u.mu.Unlock()
		return nil
	}
	text := u.aggregation
	u.aggregation = ""
	u.turnComplete = false
	u.haveFinal = false
	u.mu.Unlock()

	u.context.AddUserMessage(text)
	return u.PushFrame(ctx, frames.NewLLMContextFrame(u.context), processor.Downstream)
}

// AssistantAggregator collects the LLM's streamed text into a single assistant
// message and appends it to the context when the response completes. If the
// response is interrupted (barge-in), the partial text gathered so far is
// committed so the context reflects what the bot actually said. The response
// fields are touched from both the process goroutine (text frames) and the
// input goroutine (the InterruptionFrame system frame), so they are
// mutex-guarded.
type AssistantAggregator struct {
	*processor.Base
	context *frames.LLMContext

	mu          sync.Mutex
	aggregation string
	started     bool
	// Tool-call state for the current assistant turn. pendingIDs holds the
	// calls still awaiting a result; pendingResults collects results until all
	// have arrived and they can be written as one tool-result message.
	pendingResults []frames.ToolResult
	pendingIDs     map[string]bool
}

func newAssistant(ctx *frames.LLMContext) *AssistantAggregator {
	a := &AssistantAggregator{context: ctx, pendingIDs: make(map[string]bool)}
	a.Base = processor.New("AssistantContextAggregator", a)
	return a
}

// ProcessFrame collects LLM text into an assistant message.
func (a *AssistantAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		a.mu.Lock()
		a.started = true
		a.aggregation = ""
		a.mu.Unlock()
	case *frames.LLMTextFrame:
		a.mu.Lock()
		if a.started {
			a.aggregation += fr.Text
		}
		a.mu.Unlock()
	case *frames.FunctionCallsStartedFrame:
		a.handleFunctionCallsStarted(fr)
	case *frames.FunctionCallResultFrame:
		a.handleFunctionCallResult(fr)
	case *frames.LLMFullResponseEndFrame:
		a.commit()
	case *frames.InterruptionFrame:
		// The response was cut off; keep whatever the bot already said and
		// balance any tool calls that never got a result.
		a.commitInterrupted()
	}
	return a.PushFrame(ctx, f, dir)
}

// handleFunctionCallsStarted writes the assistant turn that requested the tool
// calls — any preamble text plus the tool-use blocks — and records the calls as
// awaiting results.
func (a *AssistantAggregator) handleFunctionCallsStarted(fr *frames.FunctionCallsStartedFrame) {
	a.mu.Lock()
	text := a.aggregation
	a.aggregation = ""
	for _, c := range fr.Calls {
		a.pendingIDs[c.ID] = true
	}
	a.mu.Unlock()
	a.context.AddAssistantToolCalls(text, fr.Calls)
}

// handleFunctionCallResult buffers a tool result and, once every call from the
// assistant turn has one, writes them as a single tool-result message.
func (a *AssistantAggregator) handleFunctionCallResult(fr *frames.FunctionCallResultFrame) {
	a.mu.Lock()
	a.pendingResults = append(a.pendingResults, frames.ToolResult{
		ID:      fr.ToolCallID,
		Name:    fr.ToolName,
		Content: fr.Result,
		IsError: fr.IsError,
	})
	delete(a.pendingIDs, fr.ToolCallID)
	var results []frames.ToolResult
	if len(a.pendingIDs) == 0 {
		results = a.pendingResults
		a.pendingResults = nil
	}
	a.mu.Unlock()
	if results != nil {
		a.context.AddToolResults(results)
	}
}

// commit appends the aggregated assistant message to the context, if any, and
// resets the response state.
func (a *AssistantAggregator) commit() {
	a.mu.Lock()
	text := a.aggregation
	a.aggregation = ""
	a.started = false
	a.mu.Unlock()
	if text != "" {
		a.context.AddAssistantMessage(text)
	}
}

// commitInterrupted closes out a turn cut off by an interruption. Any tool calls
// still awaiting a result get a synthetic error result so the assistant turn
// that requested them stays balanced (a tool-use block always has a matching
// tool-result), then any partial assistant text is committed. This keeps the
// context valid for the next turn.
func (a *AssistantAggregator) commitInterrupted() {
	a.mu.Lock()
	results := a.pendingResults
	a.pendingResults = nil
	for id := range a.pendingIDs {
		results = append(results, frames.ToolResult{ID: id, Content: "interrupted", IsError: true})
		delete(a.pendingIDs, id)
	}
	text := a.aggregation
	a.aggregation = ""
	a.started = false
	a.mu.Unlock()
	if len(results) > 0 {
		a.context.AddToolResults(results)
	}
	if text != "" {
		a.context.AddAssistantMessage(text)
	}
}
