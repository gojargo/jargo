package llm

import (
	"cmp"
	"context"
	"log/slog"
	"strings"
	"time"

	"github.com/gojargo/jargo/frames"
)

// Turn completion: the markers the model opens each response with, the
// instructions that ask for them, and their configuration. The
// LLMTurnCompletionLLMExtension reads and reports them (see
// turncompletionextension.go), so the turns package's LLMTurnCompletionStop can
// decide whether the user was done.

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

// completionInstructionsTemplate teaches the model the marker protocol. The
// turn-completion extension composes it onto the system instruction, with the
// markers substituted in when it is rendered.
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

// UserTurnCompletionConfig configures turn completion: the markers, the
// instructions, the re-prompts and their timeouts.
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

// FilterIncompleteUserTurns reports whether the service reads turn-completion
// markers: whether the turn-completion extension is attached.
//
// Deprecated: use GetExtension[*LLMTurnCompletionLLMExtension].
func (b *Base) FilterIncompleteUserTurns() bool {
	_, ok := GetExtension[*LLMTurnCompletionLLMExtension](b)
	return ok
}

// SetFilterIncompleteUserTurns attaches the turn-completion extension, built
// with the configuration the settings hold, or removes the one attached.
//
// Deprecated: use turns.FilterIncompleteUserTurnStrategies on the user
// aggregator, whose stop strategy installs the extension.
func (b *Base) SetFilterIncompleteUserTurns(on bool) {
	if err := b.setTurnCompletion(context.Background(), on, b.turnCompletionConfigSetting()); err != nil {
		slog.Error("switching turn completion failed", "service", b.Name(), "error", err)
	}
	b.composeSystemInstruction()
}

// SetUserTurnCompletionConfig reconfigures the turn-completion extension, when
// one is attached.
//
// Deprecated: pass the configuration to turns.FilterIncompleteUserTurnStrategies.
func (b *Base) SetUserTurnCompletionConfig(cfg UserTurnCompletionConfig) {
	if extension, ok := GetExtension[*LLMTurnCompletionLLMExtension](b); ok {
		extension.SetConfig(cfg)
		b.composeSystemInstruction()
	}
}

// UserTurnCompletionConfig is the configuration of the turn-completion
// extension attached, or the default one when none is.
//
// Deprecated: read it from GetExtension[*LLMTurnCompletionLLMExtension].
func (b *Base) UserTurnCompletionConfig() UserTurnCompletionConfig {
	if extension, ok := GetExtension[*LLMTurnCompletionLLMExtension](b); ok {
		return extension.Config()
	}
	return UserTurnCompletionConfig{}
}
