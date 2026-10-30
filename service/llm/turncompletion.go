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

// Reading the turn-completion markers lives on the LLM service rather than
// beside it, because it works on the text the service is about to emit, not on
// the frames that have already left. The service calls pushTurnText instead of
// pushing an LLMTextFrame, so a dropped response never becomes a frame at all,
// and it intercepts its own PushFrame so the response-lifecycle frames it
// generates reach the state machine before anything downstream sees them.
//
// The protocol: the model is told to begin every response with one of three
// markers: complete, the user will continue shortly, or the user needs more
// time. The service strips the marker from the spoken text and reports it:
// upstream, once per inference, as a UserTurnCompletionFrame for the
// LLMTurnCompletionStop strategy to decide on (a tool call before any marker
// reports a complete turn, and a response without a marker reports no kind); in
// both directions as a UserTurnInferenceCompletedFrame when the turn is
// complete; and downstream as an LLMMarkerFrame, which the assistant aggregator
// writes to the conversation. Text after an incomplete marker is not spoken:
// the marker is the whole response. Re-prompting a user who goes quiet after an
// incomplete turn is the stop strategy's job.

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
	// TurnMarkerIncomplete means the rest of the response is dropped. It doubles
	// as a latch: the prompt asks for the marker alone, and a model that
	// disobeys and keeps streaming stays suppressed.
	TurnMarkerIncomplete
)

// IncompleteType is which incomplete marker the response carried.
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

// ShortTimeout is how long a user who left a short incomplete turn is given
// before being re-prompted: the configured wait, or 5s.
func (c UserTurnCompletionConfig) ShortTimeout() time.Duration {
	if c.IncompleteShortTimeout != 0 {
		return c.IncompleteShortTimeout
	}
	return defaultIncompleteShortTimeout
}

// LongTimeout is how long a user who left a long incomplete turn is given
// before being re-prompted: the configured wait, or 10s.
func (c UserTurnCompletionConfig) LongTimeout() time.Duration {
	if c.IncompleteLongTimeout != 0 {
		return c.IncompleteLongTimeout
	}
	return defaultIncompleteLongTimeout
}

// Kinds of turn-completion judgment, as a UserTurnCompletionFrame reports them.
const (
	completionKindComplete = "complete"
	// completionKindNone is a response that carried no marker.
	completionKindNone = ""
)

// turnCompletionState is the marker state of one service. Upstream runs on a
// single event loop and needs no lock; jargo reaches this from the frame
// goroutine and from the generation, so it carries its own.
type turnCompletionState struct {
	mu sync.Mutex
	// enabled reports whether the markers are on. It is set by a settings
	// update, which is how the stop strategy turns them on once the pipeline is
	// running.
	enabled bool
	config  UserTurnCompletionConfig
	// contextFrameID is the LLMContextFrame that started the current inference,
	// and reported whether its turn-completion judgment has been reported
	// upstream yet.
	contextFrameID uint64
	reported       bool
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
}

// FilterIncompleteUserTurns reports whether turn-completion markers are on.
func (b *Base) FilterIncompleteUserTurns() bool {
	b.turnCompletion.mu.Lock()
	defer b.turnCompletion.mu.Unlock()
	return b.turnCompletion.enabled
}

// SetFilterIncompleteUserTurns turns turn-completion markers on or off, and
// rebuilds the system instruction so the marker protocol is taught exactly while
// it is on.
//
// The service only reads and reports the markers. Deciding the turn from them,
// and re-prompting a user who goes quiet after an incomplete one, is the job of
// the turns package's LLMTurnCompletionStop, which turns them on itself.
func (b *Base) SetFilterIncompleteUserTurns(on bool) {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.enabled = on
	b.turnCompletion.mu.Unlock()
	slog.Info("incomplete turn filtering", "service", b.Name(), "enabled", on)
	b.composeSystemInstruction()
}

// SetUserTurnCompletionConfig replaces the marker configuration, and rebuilds
// the system instruction in case the protocol taught to the model changed.
func (b *Base) SetUserTurnCompletionConfig(cfg UserTurnCompletionConfig) {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.config = cfg
	b.turnCompletion.mu.Unlock()
	b.composeSystemInstruction()
}

// UserTurnCompletionConfig is the marker configuration in force.
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
		b.turnReset(ctx)
	case *frames.LLMContextFrame:
		b.turnCompletion.mu.Lock()
		b.turnCompletion.contextFrameID = fr.ID()
		b.turnCompletion.reported = false
		b.turnCompletion.mu.Unlock()
	}
}

// handleTurnCompletionPushFrame follows the response the service is pushing,
// resetting at its end. It lives on the push path because these frames are the
// service's own and never arrive as input, and it runs before anything decides
// whether the frame goes out: a speculative response the gate drops still ends
// here, and its marker is reported while the turn waits for it.
func (b *Base) handleTurnCompletionPushFrame(ctx context.Context, f frames.Frame) {
	switch f.(type) {
	case *frames.FunctionCallsStartedFrame:
		// A tool call is the model judging the turn complete. Reported before
		// the function dispatches, so the turn can end before its result
		// reaches the assistant aggregator.
		b.reportTurnCompletion(ctx, completionKindComplete, "")
	case *frames.LLMFullResponseEndFrame:
		// A response that ends without a marker or a tool call carries no
		// judgment.
		b.reportTurnCompletion(ctx, completionKindNone, "")
		b.turnReset(ctx)
	}
}

// reportTurnCompletion pushes the turn-completion judgment upstream, once per
// inference, as a UserTurnCompletionFrame naming the inference it judges. A
// complete turn is also broadcast as a UserTurnInferenceCompletedFrame.
func (b *Base) reportTurnCompletion(ctx context.Context, kind, marker string) {
	b.turnCompletion.mu.Lock()
	if b.turnCompletion.reported {
		b.turnCompletion.mu.Unlock()
		return
	}
	b.turnCompletion.reported = true
	id := b.turnCompletion.contextFrameID
	b.turnCompletion.mu.Unlock()

	report := frames.NewUserTurnCompletionFrame(kind, marker, id)
	if err := b.PushFrame(ctx, report, processor.Upstream); err != nil {
		slog.Error("reporting the turn-completion judgment failed", "service", b.Name(), "error", err)
	}
	if kind != completionKindComplete {
		return
	}
	if err := b.Broadcast(ctx, func() frames.Frame {
		return frames.NewUserTurnInferenceCompletedFrame()
	}); err != nil {
		slog.Error("reporting the user turn complete failed", "error", err)
	}
}

// pushMarker reports the marker read from the response, and pushes it for the
// conversation. Only a complete turn's marker belongs to the response that
// follows: it is held and flushed with that text, while an incomplete one is
// written as an assistant message of its own, since such a turn produces no
// speech and the marker is the whole entry.
func (b *Base) pushMarker(ctx context.Context, marker, kind string) error {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.foundMarker, b.turnCompletion.foundKind = marker, kind
	b.turnCompletion.mu.Unlock()

	b.reportTurnCompletion(ctx, kind, marker)

	mf := frames.NewLLMMarkerFrame(marker)
	mf.AppendToContextImmediately = kind != completionKindComplete
	return b.PushFrame(ctx, mf, processor.Downstream)
}

// markerResponse is the report of what the service made of the response that
// just ended: the marker it found, what that meant, and the text the model
// produced before anything was held back. It reports false when the markers
// are off, where there is no protocol to report against.
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
func (b *Base) turnReset(ctx context.Context) {
	b.turnCompletion.mu.Lock()
	orphaned := ""
	if b.turnCompletion.marker == turnMarkerNone && b.turnCompletion.buffer != "" {
		orphaned = b.turnCompletion.buffer
	}
	b.turnCompletion.buffer = ""
	b.turnCompletion.marker = turnMarkerNone
	b.turnCompletion.foundMarker, b.turnCompletion.foundKind = "", ""
	b.turnCompletion.raw.Reset()
	b.turnCompletion.mu.Unlock()

	if orphaned != "" {
		slog.Warn("turn-completion markers are on but the response carried none; "+
			"pushing its text anyway, as the system prompt may be missing the protocol",
			"service", b.Name())
		if err := b.PushFrame(ctx, frames.NewLLMTextFrame(orphaned), processor.Downstream); err != nil {
			slog.Error("pushing the unmarked response failed", "error", err)
		}
	}
}

// pushLLMText emits one chunk of generated text, through the turn-completion
// markers when they are on and straight out when they are not.
func (b *Base) pushLLMText(ctx context.Context, text string) error {
	// Measured before the markers are read, which can hold text back or drop it
	// entirely. Neither says anything about how fast the model answered.
	b.StopTTFATMetrics()
	if b.FilterIncompleteUserTurns() {
		return b.pushTurnText(ctx, text)
	}
	return b.PushFrame(ctx, frames.NewLLMTextFrame(text), processor.Downstream)
}

// pushTurnText emits one chunk of generated text through the markers. The
// service calls it instead of pushing an LLMTextFrame, so text after an
// incomplete marker never becomes a frame. It buffers text until a marker
// appears; on the short or long marker it reports it and drops the rest of the
// response, and on the complete marker it reports it and pushes the text after
// it, which then flows straight through.
func (b *Base) pushTurnText(ctx context.Context, text string) error {
	b.turnCompletion.mu.Lock()
	b.turnCompletion.raw.WriteString(text)
	marker := b.turnCompletion.marker

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
		found := shortMarker
		if incomplete == IncompleteLong {
			found = longMarker
		}
		b.turnCompletion.marker = TurnMarkerIncomplete
		b.turnCompletion.buffer = ""
		b.turnCompletion.mu.Unlock()

		slog.Debug("an incomplete turn was reported, dropping the response",
			"kind", incomplete.String(), "marker", found)
		// The user was cut off or needs time, and the marker is the whole
		// response.
		return b.pushMarker(ctx, found, incomplete.String())
	}

	_, rest, found := strings.Cut(buffer, completeMarker)
	if !found {
		b.turnCompletion.mu.Unlock()
		return nil // still buffering, no marker yet
	}

	b.turnCompletion.marker = TurnMarkerComplete
	b.turnCompletion.buffer = ""
	b.turnCompletion.mu.Unlock()

	slog.Debug("a complete turn was reported, pushing the buffered response")

	// The marker goes to the conversation as a sideband signal the assistant
	// aggregator prepends to the text it is aggregating, so the message it
	// finally writes reads as the marker followed by the response.
	if err := b.pushMarker(ctx, completeMarker, completionKindComplete); err != nil {
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
