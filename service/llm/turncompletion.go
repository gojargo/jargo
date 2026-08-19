package llm

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Turn-completion gating lives on the LLM service rather than beside it,
// because it works on the text the service is about to emit, not on the frames
// that have already left. The service calls pushTurnText instead of pushing an
// LLMTextFrame, so a suppressed response never becomes a frame at all, and it
// intercepts its own PushFrame so the response-lifecycle frames it generates
// reach the state machine before anything downstream sees them.
//
// The protocol: the model is told to begin every response with one of three
// markers. A complete turn is answered normally; an incomplete one is
// suppressed and a re-prompt is armed, so the user is given time to finish and
// nudged if they do not.

// The markers the model is instructed to begin each response with. They are
// defined with the frames that carry them, since the conversation aggregator
// has to recognize them too.
const (
	// MarkerComplete means the user's turn was complete; answer normally.
	MarkerComplete = frames.UserTurnCompleteMarker
	// MarkerIncompleteShort means the user was cut off and will likely continue
	// within seconds.
	MarkerIncompleteShort = frames.UserTurnIncompleteShortMarker
	// MarkerIncompleteLong means the user needs longer to think.
	MarkerIncompleteLong = frames.UserTurnIncompleteLongMarker
)

// TurnMarker is the completion verdict found in the response being streamed.
type TurnMarker int

const (
	// turnMarkerNone means no marker has been seen yet, so text keeps buffering.
	turnMarkerNone TurnMarker = iota
	// TurnMarkerComplete means the response flows through as speech.
	TurnMarkerComplete
	// TurnMarkerIncomplete means the response is suppressed and a re-prompt is
	// armed. It doubles as a latch: the prompt asks for the marker alone, and a
	// model that disobeys and keeps streaming stays suppressed.
	TurnMarkerIncomplete
)

// IncompleteType is how long to wait before re-prompting.
type IncompleteType int

const (
	// IncompleteShort follows the short marker: the user was cut off.
	IncompleteShort IncompleteType = iota
	// IncompleteLong follows the long marker: the user needs time to think.
	IncompleteLong
)

func (t IncompleteType) String() string {
	if t == IncompleteShort {
		return "short"
	}
	return "long"
}

const (
	defaultIncompleteShortTimeout = 5 * time.Second
	defaultIncompleteLongTimeout  = 10 * time.Second
)

// The prompts and the instructions below are the protocol exactly as it is
// worded for the model. Their lines are long because the wording is theirs, not
// prose written here, and reflowing them would change what the model is told.

// DefaultIncompleteShortPrompt asks the model to nudge a user who paused
// briefly.
//
//nolint:lll // the protocol text is reproduced verbatim as the model is given it
const DefaultIncompleteShortPrompt = `The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given time to continue.

Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational
- Be very concise (1 sentence max)
- Gently prompt them to continue

Example format: ✓ Go ahead, I'm listening.

Generate your ✓ response now.`

// DefaultIncompleteLongPrompt asks the model to check in on a user who has been
// quiet for a while.
//
//nolint:lll // the protocol text is reproduced verbatim as the model is given it
const DefaultIncompleteLongPrompt = `The user has been quiet for a while. Generate a friendly check-in message.

IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given plenty of time.

Your response should:
- Acknowledge they might be thinking or busy
- Offer to help or continue when ready
- Be warm and understanding
- Be brief (1 sentence)

Example format: ✓ No rush! Let me know when you're ready to continue.

Generate your ✓ response now.`

// UserTurnCompletionInstructions teaches the model the marker protocol. It is
// composed onto the system instruction while turn-completion gating is on.
//
//nolint:lll // the protocol text is reproduced verbatim
//nolint:lll // the protocol text is reproduced verbatim as the model is given it
const UserTurnCompletionInstructions = `
CRITICAL INSTRUCTION - MANDATORY RESPONSE FORMAT:
Every single response MUST begin with a turn completion indicator. This is not optional.

TURN COMPLETION DECISION FRAMEWORK:
Ask yourself: "Has the user provided enough information for me to give a meaningful, substantive response?"

Mark as COMPLETE (✓) when:
- The user has answered your question with actual content
- The user has made a complete request or statement
- The user has provided all necessary information for you to respond meaningfully
- The conversation can naturally progress to your substantive response

Mark as INCOMPLETE SHORT (○) when the user will likely continue soon:
- The user was clearly cut off mid-sentence or mid-word
- The user is in the middle of a thought that got interrupted
- Brief technical interruption (they'll resume in a few seconds)

Mark as INCOMPLETE LONG (◐) when the user needs more time:
- The user explicitly asks for time: "let me think", "give me a minute", "hold on"
- The user is clearly pondering or deliberating: "hmm", "well...", "that's a good question"
- The user acknowledged but hasn't answered yet: "That's interesting..."
- The response feels like a preamble before the actual answer

RESPOND in one of these three formats:
1. If COMPLETE: ` + "`✓`" + ` followed by a space and your full substantive response
2. If INCOMPLETE SHORT: ONLY the character ` + "`○`" + ` (user will continue in a few seconds)
3. If INCOMPLETE LONG: ONLY the character ` + "`◐`" + ` (user needs more time to think)

KEY INSIGHT: Grammatically complete ≠ conversationally complete
- "That's a really good question." is grammatically complete but conversationally incomplete (use ◐)
- "I'd go to Japan because I love" is mid-sentence (use ○)

EXAMPLES:

You ask: "Where would you travel?"
User: "I'd go to Japan because I love"
→ ` + "`○`" + `
(Cut off mid-sentence - they'll continue in seconds)

You ask: "Where would you travel?"
User: "That's a good question. Let me think..."
→ ` + "`◐`" + `
(User is deliberating - give them time)

You ask: "Where would you travel?"
User: "Hmm, hold on a second."
→ ` + "`◐`" + `
(User explicitly asked for time)

You ask: "Where would you travel?"
User: "I'd go to Japan because I love the culture."
→ ` + "`✓ Japan is a wonderful choice! The blend of ancient traditions and modern innovation is truly unique. Have you been before?`" + `
(Complete answer - give full response)

User: "I need help with"
→ ` + "`○`" + `
(Cut off mid-request - they'll finish soon)

User: "Well, let me think about that for a moment."
→ ` + "`◐`" + `
(User needs time to think)

User: "Can you help me book a flight to New York next week?"
→ ` + "`✓ I'd be happy to help you with that! Let me gather some information...`" + `
(Complete request - provide full response)

User: "Give me a minute to gather my thoughts."
→ ` + "`◐`" + `
(User explicitly asked for time)

FORMAT REQUIREMENTS:
- ALWAYS use single-character indicators: ` + "`✓`" + ` (complete), ` + "`○`" + ` (short wait), or ` + "`◐`" + ` (long wait)
- For COMPLETE: ` + "`✓`" + ` followed by a space and your full response
- For INCOMPLETE: ONLY the single character (` + "`○`" + ` or ` + "`◐`" + `) with absolutely nothing else
- Your turn indicator must be the very first character in your response

Remember: Focus on conversational completeness and how long the user might need. Was it a mid-sentence cutoff (○) or do they need time to think (◐)?`

// UserTurnCompletionConfig configures turn-completion gating.
type UserTurnCompletionConfig struct {
	// Instructions overrides the marker protocol taught to the model. Empty uses
	// UserTurnCompletionInstructions.
	Instructions string
	// IncompleteShortTimeout is how long to wait after the short marker before
	// re-prompting. Zero uses 5s.
	IncompleteShortTimeout time.Duration
	// IncompleteLongTimeout is how long to wait after the long marker. Zero uses
	// 10s.
	IncompleteLongTimeout time.Duration
	// IncompleteShortPrompt overrides the re-prompt sent when the short timeout
	// expires. Empty uses DefaultIncompleteShortPrompt.
	IncompleteShortPrompt string
	// IncompleteLongPrompt overrides the re-prompt sent when the long timeout
	// expires. Empty uses DefaultIncompleteLongPrompt.
	IncompleteLongPrompt string
}

// CompletionInstructions is the marker protocol to teach the model: the
// configured one, or the default when none was given.
func (c UserTurnCompletionConfig) CompletionInstructions() string {
	if c.Instructions != "" {
		return c.Instructions
	}
	return UserTurnCompletionInstructions
}

// ShortPrompt is the re-prompt for a short incomplete turn.
func (c UserTurnCompletionConfig) ShortPrompt() string {
	if c.IncompleteShortPrompt != "" {
		return c.IncompleteShortPrompt
	}
	return DefaultIncompleteShortPrompt
}

// LongPrompt is the re-prompt for a long incomplete turn.
func (c UserTurnCompletionConfig) LongPrompt() string {
	if c.IncompleteLongPrompt != "" {
		return c.IncompleteLongPrompt
	}
	return DefaultIncompleteLongPrompt
}

// timeout is the wait before re-prompting for the given kind of incomplete turn.
func (c UserTurnCompletionConfig) timeout(t IncompleteType) time.Duration {
	if t == IncompleteShort {
		if c.IncompleteShortTimeout != 0 {
			return c.IncompleteShortTimeout
		}
		return defaultIncompleteShortTimeout
	}
	if c.IncompleteLongTimeout != 0 {
		return c.IncompleteLongTimeout
	}
	return defaultIncompleteLongTimeout
}

// prompt is the re-prompt for the given kind of incomplete turn.
func (c UserTurnCompletionConfig) prompt(t IncompleteType) string {
	if t == IncompleteShort {
		return c.ShortPrompt()
	}
	return c.LongPrompt()
}

// turnCompletionState is the gating state of one service. Upstream runs on a
// single event loop and needs no lock; jargo reaches this from the frame
// goroutine and from the re-prompt timer, so it carries its own.
type turnCompletionState struct {
	mu sync.Mutex
	// enabled reports whether gating is on. It is set by a settings update, which
	// is how the stop strategy turns it on once the pipeline is running.
	enabled bool
	config  UserTurnCompletionConfig
	// buffer holds the response text seen so far, until a marker appears in it.
	buffer string
	// marker is the verdict for the response being streamed.
	marker TurnMarker
	// broadcasted reports whether this turn has already been reported complete,
	// so a turn that both calls a tool and produces the marker reports once.
	broadcasted bool
	// voiced reports whether a complete verdict has been spoken since the user
	// last started speaking. A turn detector can trigger several inferences
	// within one user turn, each producing its own marker; this latch voices at
	// most one of them, so the bot does not repeat itself. It is not a per-turn
	// guarantee: it is cleared on a mid-turn resume as well, because a
	// completion the controller dropped as stale would otherwise silence the
	// turn for good.
	voiced bool
	// cancelTimeout stops the armed re-prompt, and is nil when none is armed.
	cancelTimeout func()
}

// FilterIncompleteUserTurns reports whether turn-completion gating is on.
func (b *Base) FilterIncompleteUserTurns() bool {
	b.turnCompletion.mu.Lock()
	defer b.turnCompletion.mu.Unlock()
	return b.turnCompletion.enabled
}

// SetFilterIncompleteUserTurns turns turn-completion gating on or off, and
// rebuilds the system instruction so the marker protocol is taught exactly while
// it is on.
func (b *Base) SetFilterIncompleteUserTurns(on bool) {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.enabled = on
	b.turnCompletion.mu.Unlock()
	slog.Info("incomplete turn filtering", "service", b.Name(), "enabled", on)
	b.composeSystemInstruction()
}

// SetUserTurnCompletionConfig replaces the gating configuration, and rebuilds
// the system instruction in case the protocol taught to the model changed.
func (b *Base) SetUserTurnCompletionConfig(cfg UserTurnCompletionConfig) {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.config = cfg
	b.turnCompletion.mu.Unlock()
	b.composeSystemInstruction()
}

// UserTurnCompletionConfig is the gating configuration in force.
func (b *Base) UserTurnCompletionConfig() UserTurnCompletionConfig {
	b.turnCompletion.mu.Lock()
	defer b.turnCompletion.mu.Unlock()
	return b.turnCompletion.config
}

// handleTurnCompletionProcessFrame reacts to the frames that arrive at the
// service. It runs before the frame is handled, so the state is right by the
// time any text of the response that follows is parsed.
func (b *Base) handleTurnCompletionProcessFrame(ctx context.Context, f frames.Frame) {
	switch fr := f.(type) {
	case *frames.InterruptionFrame:
		b.cancelIncompleteTimeout()
		b.turnReset(ctx)
		b.clearVoiced()
	case *frames.UserStartedSpeakingFrame:
		// A new user turn: allow one fresh spoken completion.
		b.clearVoiced()
	case *frames.LLMMessagesAppendFrame:
		// A message appended from outside that asks for a run is an explicit
		// request for fresh speech, and it arrives precisely while the user is
		// silent. Clear the latch so the guard does not drop its text.
		if fr.RunLLM {
			b.clearVoiced()
		}
	case *frames.VADUserStartedSpeakingFrame:
		// The user resumed inside a turn that is already open, so no interruption
		// fires and two things that normally reset on a fresh turn are handled
		// here instead. An armed re-prompt would talk over a user who is speaking
		// again, so it is canceled; and one fresh spoken completion is allowed,
		// because a completion the controller dropped as stale would otherwise
		// silence the turn for good.
		b.cancelIncompleteTimeout()
		b.clearVoiced()
	}
}

// handleTurnCompletionPushFrame reacts to the frames the service itself
// generates, which is why it lives on the push path: they never arrive as
// input, so nothing on the receiving side would see them in time.
func (b *Base) handleTurnCompletionPushFrame(ctx context.Context, f frames.Frame) {
	switch f.(type) {
	case *frames.FunctionCallsStartedFrame:
		// Report the turn complete before the call dispatches, which gives the
		// user-stopped-speaking frame the most time to propagate before a result
		// travels back to the aggregator.
		b.broadcastTurnCompletion(ctx)
		// A tool call means a fresh inference is coming and that one is expected
		// to speak, so clear the latch. The response that voiced the marker keeps
		// streaming: its verdict is already complete, so its text takes the
		// complete branch rather than the latch guard.
		b.clearVoiced()
	case *frames.LLMFullResponseStartFrame:
		// A response is starting while a re-prompt is armed, so the model is
		// already re-engaging: either the turn completed and this response
		// carries the marker, or the timeout already fired its own re-prompt.
		// Either way the armed one is now redundant. This is the single point
		// that settles the race between the timeout firing and a completion
		// arriving: whichever inference starts first disarms it.
		b.cancelIncompleteTimeout()
	case *frames.LLMFullResponseEndFrame:
		b.turnReset(ctx)
	}
}

// setVoiced records whether this user turn has had its one spoken completion.
func (b *Base) clearVoiced() {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.voiced = false
	b.turnCompletion.mu.Unlock()
}

// broadcastTurnCompletion reports the user's turn complete, at most once per
// turn. It is called from the two places the model has committed to answering:
// the complete marker appearing in the text, and a tool call starting.
func (b *Base) broadcastTurnCompletion(ctx context.Context) {
	b.turnCompletion.mu.Lock()
	if b.turnCompletion.broadcasted {
		b.turnCompletion.mu.Unlock()
		return
	}
	b.turnCompletion.broadcasted = true
	b.turnCompletion.mu.Unlock()

	if err := b.Broadcast(ctx, func() frames.Frame {
		return frames.NewUserTurnInferenceCompletedFrame()
	}); err != nil {
		slog.Error("reporting the user turn complete failed", "error", err)
	}
}

// startIncompleteTimeout arms the re-prompt for an incomplete turn, replacing
// any already armed.
func (b *Base) startIncompleteTimeout(t IncompleteType) {
	b.cancelIncompleteTimeout()

	b.turnCompletion.mu.Lock()
	timeout := b.turnCompletion.config.timeout(t)
	b.turnCompletion.mu.Unlock()

	slog.Debug("arming the incomplete-turn re-prompt", "kind", t.String(), "timeout", timeout)

	ctx, cancel := context.WithCancel(b.turnCtx)
	b.turnCompletion.mu.Lock()
	b.turnCompletion.cancelTimeout = cancel
	b.turnCompletion.mu.Unlock()

	b.turnWG.Go(func() {
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
		b.incompleteTimeoutExpired(ctx, t)
	})
}

// cancelIncompleteTimeout disarms the re-prompt, if one is armed.
func (b *Base) cancelIncompleteTimeout() {
	b.turnCompletion.mu.Lock()
	cancel := b.turnCompletion.cancelTimeout
	b.turnCompletion.cancelTimeout = nil
	b.turnCompletion.mu.Unlock()
	if cancel != nil {
		slog.Debug("disarming the incomplete-turn re-prompt")
		cancel()
	}
}

// incompleteTimeoutExpired re-prompts the model after the user did not
// continue. The state is reset first, so the response the re-prompt draws is
// parsed as a fresh one.
func (b *Base) incompleteTimeoutExpired(ctx context.Context, t IncompleteType) {
	slog.Debug("the incomplete-turn re-prompt expired, prompting the model", "kind", t.String())

	b.turnReset(ctx)

	b.turnCompletion.mu.Lock()
	b.turnCompletion.cancelTimeout = nil
	prompt := b.turnCompletion.config.prompt(t)
	b.turnCompletion.mu.Unlock()

	msg := []frames.Message{{Role: frames.RoleDeveloper, Text: prompt}}
	if err := b.PushFrame(ctx, frames.NewLLMMessagesAppendFrame(msg), processor.Downstream); err != nil {
		slog.Error("sending the incomplete-turn re-prompt failed", "error", err)
		return
	}
	if err := b.PushFrame(ctx, frames.NewLLMRunFrame(), processor.Downstream); err != nil {
		slog.Error("running the incomplete-turn re-prompt failed", "error", err)
	}
}

// turnReset clears the per-response state. A response that produced no marker
// at all has its buffered text pushed rather than dropped, so a model that
// ignored the protocol still says something.
//
// It deliberately leaves an armed re-prompt alone: that is disarmed when the
// user speaks and when a new inference begins, not when a response ends.
func (b *Base) turnReset(ctx context.Context) {
	b.turnCompletion.mu.Lock()
	orphaned := ""
	if b.turnCompletion.marker == turnMarkerNone && b.turnCompletion.buffer != "" {
		orphaned = b.turnCompletion.buffer
	}
	b.turnCompletion.buffer = ""
	b.turnCompletion.marker = turnMarkerNone
	b.turnCompletion.broadcasted = false
	b.turnCompletion.mu.Unlock()

	if orphaned != "" {
		slog.Warn("turn-completion gating is on but the response carried no marker; "+
			"pushing its text anyway, as the system prompt may be missing the protocol",
			"service", b.Name())
		if err := b.PushFrame(ctx, frames.NewLLMTextFrame(orphaned), processor.Downstream); err != nil {
			slog.Error("pushing the unmarked response failed", "error", err)
		}
	}
}

// pushLLMText emits one chunk of generated text, through the turn-completion
// gating when it is on and straight out when it is not.
func (b *Base) pushLLMText(ctx context.Context, text string) error {
	if b.FilterIncompleteUserTurns() {
		return b.pushTurnText(ctx, text)
	}
	return b.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream)
}

// pushTurnText emits one chunk of generated text through the gating. The
// service calls it instead of pushing an LLMTextFrame, so a suppressed response
// never becomes a frame.
func (b *Base) pushTurnText(ctx context.Context, text string) error {
	b.turnCompletion.mu.Lock()
	voiced, marker := b.turnCompletion.voiced, b.turnCompletion.marker

	// One spoken completion per user turn. Once a completion has been voiced,
	// text from any later inference is dropped; a turn detector can trigger
	// several within one turn. The check on the marker scopes this to fresh
	// responses, since the one that voiced the completion has its verdict set.
	if voiced && marker == turnMarkerNone {
		b.turnCompletion.mu.Unlock()
		return nil
	}

	// Suppress everything after an incomplete verdict, in case the model
	// disobeys the protocol and keeps talking past the marker.
	if marker == TurnMarkerIncomplete {
		b.turnCompletion.mu.Unlock()
		return nil
	}

	// Past a complete verdict the text flows straight through.
	if marker == TurnMarkerComplete {
		b.turnCompletion.mu.Unlock()
		return b.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream)
	}

	b.turnCompletion.buffer += text
	buffer := b.turnCompletion.buffer

	// The short marker is looked for first, matching the order the protocol
	// presents them in.
	incomplete, isIncomplete := IncompleteShort, true
	switch {
	case strings.Contains(buffer, MarkerIncompleteShort):
		incomplete = IncompleteShort
	case strings.Contains(buffer, MarkerIncompleteLong):
		incomplete = IncompleteLong
	default:
		isIncomplete = false
	}

	if isIncomplete {
		marker := MarkerIncompleteShort
		if incomplete == IncompleteLong {
			marker = MarkerIncompleteLong
		}
		b.turnCompletion.marker = TurnMarkerIncomplete
		b.turnCompletion.buffer = ""
		b.turnCompletion.mu.Unlock()

		slog.Debug("an incomplete turn was reported, suppressing the response",
			"kind", incomplete.String(), "marker", marker)

		// Nothing reports the turn complete here: it explicitly is not. The
		// re-prompt below is what drives the turn on.
		//
		// The marker is written to the conversation as an assistant message of
		// its own, since an incomplete turn produces no speech and the marker is
		// therefore the whole entry.
		if err := b.PushFrame(ctx, frames.NewLLMMarkerFrame(marker), processor.Downstream); err != nil {
			return err
		}
		b.startIncompleteTimeout(incomplete)
		return nil
	}

	_, rest, found := strings.Cut(buffer, MarkerComplete)
	if !found {
		b.turnCompletion.mu.Unlock()
		return nil // still buffering, no marker yet
	}

	// This user turn now has its one spoken completion. Any armed re-prompt was
	// already disarmed when this response's start frame was pushed.
	b.turnCompletion.voiced = true
	b.turnCompletion.marker = TurnMarkerComplete
	b.turnCompletion.buffer = ""
	b.turnCompletion.mu.Unlock()

	slog.Debug("a complete turn was reported, pushing the buffered response")

	// Report the turn complete before the marker, so a stop strategy gating
	// finalization on it sees the signal ahead of the response. It is idempotent:
	// a tool call earlier in the turn may already have reported.
	b.broadcastTurnCompletion(ctx)

	// The marker goes to the conversation as a sideband signal the assistant
	// aggregator prepends to the text it is aggregating, so the message it
	// finally writes reads as the marker followed by the response.
	mf := frames.NewLLMMarkerFrame(MarkerComplete)
	mf.AppendToContextImmediately = false
	if err := b.PushFrame(ctx, mf, processor.Downstream); err != nil {
		return err
	}

	// A model may send the marker and the first words in one chunk, so whatever
	// followed it in this chunk is pushed as speech.
	rest = strings.TrimPrefix(rest, " ")
	if rest == "" {
		return nil
	}
	return b.PushFrame(ctx, frames.NewLLMTextFrame(rest), processor.Downstream)
}
