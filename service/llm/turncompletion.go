package llm

import (
	"cmp"
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

// incompleteShortPromptTemplate asks the model to nudge a user who paused
// briefly. The markers are substituted in when it is rendered.
//
//nolint:lll // the protocol text is reproduced verbatim as the model is given it
const incompleteShortPromptTemplate = `The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

IMPORTANT: You MUST respond with {{complete}} followed by your message. Do NOT output {{short}} or {{long}} - the user has already been given time to continue.

Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational
- Be very concise (1 sentence max)
- Gently prompt them to continue

Example format: {{complete}} Go ahead, I'm listening.

Generate your {{complete}} response now.`

// incompleteLongPromptTemplate asks the model to check in on a user who has
// been quiet for a while. The markers are substituted in when it is rendered.
//
//nolint:lll // the protocol text is reproduced verbatim as the model is given it
const incompleteLongPromptTemplate = `The user has been quiet for a while. Generate a friendly check-in message.

IMPORTANT: You MUST respond with {{complete}} followed by your message. Do NOT output {{short}} or {{long}} - the user has already been given plenty of time.

Your response should:
- Acknowledge they might be thinking or busy
- Offer to help or continue when ready
- Be warm and understanding
- Be brief (1 sentence)

Example format: {{complete}} No rush! Let me know when you're ready to continue.

Generate your {{complete}} response now.`

// completionInstructionsTemplate teaches the model the marker protocol. It is
// composed onto the system instruction while turn-completion gating is on, with
// the markers substituted in when it is rendered.
//
//nolint:lll // the protocol text is reproduced verbatim as the model is given it
const completionInstructionsTemplate = `
TURN COMPLETION PROTOCOL (mandatory):
The user's words reach you from speech recognition, usually without punctuation, and sometimes before they have finished talking. Before you reply, decide whether their turn is complete, and start every response with exactly one of these markers as its very first character:

{{complete}}  the user's turn is complete: answer them. Write {{complete}}, a space, then your full reply. Never write {{complete}} on its own.
{{short}}  the user stopped mid-sentence and will continue in a few seconds. Write {{short}} and nothing else.
{{long}}  the user needs time to think or asked you to wait. Write {{long}} and nothing else.

Deciding:
- Complete means conversationally complete, not long. One word can be a complete answer to your question: "yes", "no thanks", "Tuesday", "Japan", "four". A question to you is complete. A correction, or a request to repeat yourself, is complete.
- Grammatically complete is not the same as conversationally complete. If the user has only acknowledged your question or reacted to what you said, without answering it ("that's a good question", "oh wow, okay", "that's interesting", "hmm", "well"), they have not taken their turn yet: {{long}}.
- Cut off ({{short}}): the last words leave a phrase open, in whatever language the user speaks: a sentence that ends on a conjunction, a preposition, an article, or the word for "because"; a list that is still going; a number that is only partly said. A fragment like this is not a request for help. Do not answer it, do not ask what they need, and do not guess the rest: wait with {{short}}.
- Needs time ({{long}}): "hold on", "let me think", "give me a second", "one moment". Filler followed by a real answer, such as "hmm, I'd go to Japan for the food", is complete.
- When the user's latest words continue an earlier fragment (your previous response was {{short}} or {{long}}), judge the fragments together as one turn. If the combined turn still ends open, {{short}}; if it now answers your question, {{complete}}.
- If a tool call is the right response, make the tool call; the turn is complete.

Format rules:
- The marker is the first character. No text, quotes, backticks or explanation before it.
- After {{short}} or {{long}}, output nothing: no words, no explanation, no nudge. The system waits and prompts you again later.
- Exactly one marker per response.

Examples:
- You asked where they would go and the user says "i'd go to japan because i love". Respond with only {{short}}.
- The user says "i need help with". Respond with only {{short}}.
- You asked for their phone number and the user says "it's five five five". Respond with only {{short}}.
- You asked what they want to order and the user says "a large pepperoni pizza a garden salad and". Respond with only {{short}}.
- You asked where they would go and the user says "that's a good question let me think". Respond with only {{long}}.
- You asked where they would go and the user says "that's interesting". Respond with only {{long}}.
- The user says "hold on a second". Respond with only {{long}}.
- You asked where they would go and the user says "japan". Respond with {{complete}} followed by your reply, for example "{{complete}} Japan is a wonderful choice. What draws you there?"
- You asked whether to book it and the user says "yes". Respond with {{complete}} followed by your reply, for example "{{complete}} Done, I'll book it now."
- The user says "can you help me book a flight to new york next week". Respond with {{complete}} followed by your reply, for example "{{complete}} Of course. What day would you like to leave, and from which city?"
- Your previous response was {{short}} after the user said "i'd go to", and now the user says "japan because". Together that is still open: respond with only {{short}}.`

// renderMarkers substitutes the three markers into one of the protocol templates.
//
// The markers are configurable because the protocol is only a convention between
// the application and its model: a model whose tokenizer splits one of the
// defaults, or one already using a character for something else, needs a set of
// its own, and every prompt has to agree on it.
func renderMarkers(tmpl string, complete, short, long string) string {
	return strings.NewReplacer(
		"{{complete}}", complete,
		"{{short}}", short,
		"{{long}}", long,
	).Replace(tmpl)
}

// The protocol as it is worded with the default markers. They are variables
// rather than constants because they are rendered from the templates above.
//
//nolint:gochecknoglobals // rendered once, read only
var (
	// DefaultIncompleteShortPrompt nudges a user who paused briefly.
	DefaultIncompleteShortPrompt = renderMarkers(incompleteShortPromptTemplate,
		MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong)
	// DefaultIncompleteLongPrompt checks in on a user who has been quiet.
	DefaultIncompleteLongPrompt = renderMarkers(incompleteLongPromptTemplate,
		MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong)
	// UserTurnCompletionInstructions teaches the model the marker protocol.
	UserTurnCompletionInstructions = renderMarkers(completionInstructionsTemplate,
		MarkerComplete, MarkerIncompleteShort, MarkerIncompleteLong)
)

// UserTurnCompletionConfig configures turn-completion gating.
type UserTurnCompletionConfig struct {
	// Instructions overrides the marker protocol taught to the model. Empty
	// renders the default protocol from the configured markers.
	Instructions string
	// CompleteMarker is what the model emits when the user's turn is complete.
	// It is generated ahead of any speakable text, so prefer a character that is
	// a single token in the model's tokenizer. Empty uses MarkerComplete.
	CompleteMarker string
	// IncompleteShortMarker marks a turn cut off mid-thought. Empty uses
	// MarkerIncompleteShort.
	IncompleteShortMarker string
	// IncompleteLongMarker marks a user who needs more time. Empty uses
	// MarkerIncompleteLong.
	IncompleteLongMarker string
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

// Markers are the complete, short and long markers in force, in that order:
// the configured ones, and the defaults for whatever was left empty.
func (c UserTurnCompletionConfig) Markers() (complete, short, long string) {
	return cmp.Or(c.CompleteMarker, MarkerComplete),
		cmp.Or(c.IncompleteShortMarker, MarkerIncompleteShort),
		cmp.Or(c.IncompleteLongMarker, MarkerIncompleteLong)
}

// CompletionInstructions is the marker protocol to teach the model: the
// configured one, or the default rendered from the markers in force.
func (c UserTurnCompletionConfig) CompletionInstructions() string {
	if c.Instructions != "" {
		return c.Instructions
	}
	complete, short, long := c.Markers()
	return renderMarkers(completionInstructionsTemplate, complete, short, long)
}

// ShortPrompt is the re-prompt for a short incomplete turn.
func (c UserTurnCompletionConfig) ShortPrompt() string {
	if c.IncompleteShortPrompt != "" {
		return c.IncompleteShortPrompt
	}
	complete, short, long := c.Markers()
	return renderMarkers(incompleteShortPromptTemplate, complete, short, long)
}

// LongPrompt is the re-prompt for a long incomplete turn.
func (c UserTurnCompletionConfig) LongPrompt() string {
	if c.IncompleteLongPrompt != "" {
		return c.IncompleteLongPrompt
	}
	complete, short, long := c.Markers()
	return renderMarkers(incompleteLongPromptTemplate, complete, short, long)
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
	// foundMarker is the marker text the response carried, and foundKind what it
	// meant ("complete", "short" or "long"), for the report made when the
	// response ends. Both are empty until a marker is found.
	foundMarker string
	foundKind   string
	// raw accumulates the response as the model produced it, markers and all,
	// which is what that report carries: it is how a consumer checks the model
	// against the protocol it was given.
	raw strings.Builder
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
	// userSpeaking reports whether the detector currently hears the user. A
	// complete verdict arriving while it does is stale: the user resumed after
	// the inference was triggered, so the turn is not over after all.
	userSpeaking bool
	// cancelTimeout stops the armed re-prompt, and is nil when none is armed.
	cancelTimeout func()
}

// setUserSpeaking records whether the detector currently hears the user.
func (b *Base) setUserSpeaking(v bool) {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.userSpeaking = v
	b.turnCompletion.mu.Unlock()
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
		b.setUserSpeaking(true)
		b.cancelIncompleteTimeout()
		b.clearVoiced()
	case *frames.VADUserStoppedSpeakingFrame:
		b.setUserSpeaking(false)
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

// markerResponse is the report of what the service made of the response that
// just ended: the marker it found, what that meant, and the text the model
// produced before anything was held back. It reports false when gating is off,
// where there is no protocol to report against.
//
// It is a diagnostic for a consumer checking the model against the protocol it
// was given, and plays no part in the conversation.
func (b *Base) markerResponse() (*frames.LLMMarkerResponseFrame, bool) {
	if !b.FilterIncompleteUserTurns() {
		return nil, false
	}
	b.turnCompletion.mu.Lock()
	defer b.turnCompletion.mu.Unlock()
	cfg := b.turnCompletion.config
	return frames.NewLLMMarkerResponseFrame(
		b.turnCompletion.raw.String(),
		b.turnCompletion.foundMarker,
		b.turnCompletion.foundKind,
		protocolMarkers(cfg),
	), true
}

// protocolMarkers is every marker the protocol recognizes, in the order the
// model is taught them.
func protocolMarkers(cfg UserTurnCompletionConfig) []string {
	complete, short, long := cfg.Markers()
	return []string{complete, short, long}
}

// pushMarkerResponse reports what the service made of the response that just
// ended, for a consumer that asked to see it, and then clears the per-response
// state. It is called where the service itself ends a response; a service whose
// responses are bracketed from elsewhere resets on the frame instead.
func (b *Base) pushMarkerResponse(ctx context.Context) {
	report, ok := b.markerResponse()
	if !ok {
		return
	}
	if err := b.PushFrame(ctx, report, processor.Downstream); err != nil {
		slog.Error("pushing the marker report failed", "service", b.Name(), "error", err)
	}
	b.turnReset(ctx)
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
	b.turnCompletion.foundMarker, b.turnCompletion.foundKind = "", ""
	b.turnCompletion.raw.Reset()
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
	// Measured before turn-completion filtering, which can hold text back or drop
	// it entirely. Neither says anything about how fast the model answered.
	b.StopTTFATMetrics()
	if b.FilterIncompleteUserTurns() {
		return b.pushTurnText(ctx, text)
	}
	return b.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream)
}

// suppressStaleCompletion handles a complete verdict that arrives after the user
// has resumed speaking. The inference was triggered when the turn looked over;
// by the time it answers the turn is not over after all, so the response is
// suppressed and the short timeout re-armed, exactly as if the model had
// reported the turn incomplete. The next inference re-evaluates the fuller turn.
//
// The caller holds the lock and hands it over.
func (b *Base) suppressStaleCompletion(ctx context.Context, complete, short string) error {
	b.turnCompletion.marker = TurnMarkerIncomplete
	b.turnCompletion.buffer = ""
	b.turnCompletion.mu.Unlock()

	slog.Debug("a complete turn was reported while the user is speaking, "+
		"treating it as stale and suppressing the response", "marker", complete)

	if err := b.PushFrame(ctx, frames.NewLLMMarkerFrame(short), processor.Downstream); err != nil {
		return err
	}
	b.startIncompleteTimeout(IncompleteShort)
	return nil
}

// pushTurnText emits one chunk of generated text through the gating. The
// service calls it instead of pushing an LLMTextFrame, so a suppressed response
// never becomes a frame.
func (b *Base) pushTurnText(ctx context.Context, text string) error {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.raw.WriteString(text)
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
	completeMarker, shortMarker, longMarker := b.turnCompletion.config.Markers()

	// The short marker is looked for first, matching the order the protocol
	// presents them in.
	incomplete, isIncomplete := IncompleteShort, true
	switch {
	case strings.Contains(buffer, shortMarker):
		incomplete = IncompleteShort
	case strings.Contains(buffer, longMarker):
		incomplete = IncompleteLong
	default:
		isIncomplete = false
	}

	if isIncomplete {
		marker := shortMarker
		if incomplete == IncompleteLong {
			marker = longMarker
		}
		b.turnCompletion.marker = TurnMarkerIncomplete
		b.turnCompletion.buffer = ""
		b.turnCompletion.foundMarker, b.turnCompletion.foundKind = marker, incomplete.String()
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

	_, rest, found := strings.Cut(buffer, completeMarker)
	if !found {
		b.turnCompletion.mu.Unlock()
		return nil // still buffering, no marker yet
	}

	if b.turnCompletion.userSpeaking {
		return b.suppressStaleCompletion(ctx, completeMarker, shortMarker)
	}

	// This user turn now has its one spoken completion. Any armed re-prompt was
	// already disarmed when this response's start frame was pushed.
	b.turnCompletion.voiced = true
	b.turnCompletion.marker = TurnMarkerComplete
	b.turnCompletion.buffer = ""
	b.turnCompletion.foundMarker, b.turnCompletion.foundKind = completeMarker, "complete"
	b.turnCompletion.mu.Unlock()

	slog.Debug("a complete turn was reported, pushing the buffered response")

	// Report the turn complete before the marker, so a stop strategy gating
	// finalization on it sees the signal ahead of the response. It is idempotent:
	// a tool call earlier in the turn may already have reported.
	b.broadcastTurnCompletion(ctx)

	// The marker goes to the conversation as a sideband signal the assistant
	// aggregator prepends to the text it is aggregating, so the message it
	// finally writes reads as the marker followed by the response.
	mf := frames.NewLLMMarkerFrame(completeMarker)
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
