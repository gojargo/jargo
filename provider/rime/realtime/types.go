package realtime

import (
	"math"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/provider/rime/internal/rimepcm"
)

// The values a setting takes on the wire.
const (
	truncationAuto      = "auto"
	truncationDisabled  = "disabled"
	truncationRetention = "retention_ratio"

	fillerNone = "none"
	fillerText = "text"
	fillerAuto = "auto"
)

// fillerWhitespace is the whitespace Prism strips before validating a line.
// The control characters U+001C to U+001F are left out: the engine takes them
// as text.
const fillerWhitespace = "\u0009\u000a\u000b\u000c\u000d \u0085  " +
	"           " +
	"    　"

// Truncation is the policy for a history that outgrows the model's capacity.
type Truncation struct {
	// Type is "auto", "disabled" or "retention_ratio".
	Type string
	// RetentionRatio is the fraction of the history capacity to keep after
	// truncation, above zero and at most one. It applies to the type
	// "retention_ratio" only.
	RetentionRatio float64
}

// TruncationAuto lets Prism truncate the history as it sees fit.
func TruncationAuto() Truncation { return Truncation{Type: truncationAuto} }

// TruncationDisabled keeps Prism from truncating the history.
func TruncationDisabled() Truncation { return Truncation{Type: truncationDisabled} }

// TruncationRetentionRatio keeps the given fraction of the history capacity
// after truncation.
func TruncationRetentionRatio(ratio float64) Truncation {
	return Truncation{Type: truncationRetention, RetentionRatio: ratio}
}

// validate reports whether the policy is one Prism takes.
func (t Truncation) validate() error {
	switch t.Type {
	case truncationAuto, truncationDisabled:
		return nil
	case truncationRetention:
		if !(t.RetentionRatio > 0 && t.RetentionRatio <= 1) {
			return inputError("retention_ratio must be above zero and at most one")
		}
		return nil
	default:
		return inputError("truncation must be auto, disabled, or RetentionRatio")
	}
}

// ReviewSettings is the speech review policy a session reports. The service
// configures none, so it is only ever what the server says it holds.
type ReviewSettings struct {
	// TimeoutMS is how long a review may take.
	TimeoutMS int
	// OnTimeout is what happens to a chunk whose review timed out: "stop",
	// "speak" or "say".
	OnTimeout string
	// TimeoutLine is the line spoken instead, for "say".
	TimeoutLine string
}

// TurnDetectionUpdate holds the interruption rules, which can change while a
// conversation runs. A nil field keeps a setting; an empty list clears one.
type TurnDetectionUpdate struct {
	// InterruptMinWords is how many words, from 1 to 10, the user says over a
	// reply before Prism interrupts it.
	InterruptMinWords *int
	// InterruptPhrases are phrases that interrupt a reply whatever
	// InterruptMinWords says. At most 100.
	InterruptPhrases []string
	// AcknowledgmentPhrases are phrases, such as "uh huh", that do not
	// interrupt a reply. At most 100.
	AcknowledgmentPhrases []string
}

// validate reports whether the rules are ones Prism takes.
func (u TurnDetectionUpdate) validate() error {
	if u.InterruptMinWords != nil && (*u.InterruptMinWords < 1 || *u.InterruptMinWords > 10) {
		return inputError("Interrupt minimum must be an integer from 1 to 10")
	}
	if len(u.InterruptPhrases) > 100 {
		return inputError("interrupt_phrases must contain at most 100 strings")
	}
	if len(u.AcknowledgmentPhrases) > 100 {
		return inputError("acknowledgment_phrases must contain at most 100 strings")
	}
	return nil
}

// TurnDetection holds the server's turn settings for the opening of a
// conversation. A field left unset leaves the setting to the server.
type TurnDetection struct {
	TurnDetectionUpdate

	// Eagerness is how readily Prism ends a turn: "low", "medium", "high" or
	// "auto".
	Eagerness string `validate:"omitempty,oneof=low medium high auto"`
	// EndOfTurnTimeoutMS is the silence that ends a turn, from 1,000 to 10,000
	// milliseconds.
	EndOfTurnTimeoutMS *int
	// EndOfTurnMinSilenceMS is the least silence that can end a turn, from 0
	// to 5,000 milliseconds.
	EndOfTurnMinSilenceMS *int
}

// validate reports whether the settings are ones Prism takes.
func (d TurnDetection) validate() error {
	if err := d.TurnDetectionUpdate.validate(); err != nil {
		return err
	}
	if s := d.EndOfTurnMinSilenceMS; s != nil && (*s < 0 || *s > 5000) {
		return inputError("Minimum silence must be an integer from 0 to 5000 ms")
	}
	if d.Eagerness != "" && !slices.Contains([]string{"low", "medium", "high", "auto"}, d.Eagerness) {
		return inputError("Turn eagerness must be low, medium, high, or auto")
	}
	if v := d.EndOfTurnTimeoutMS; v != nil && (*v < 1000 || *v > 10000) {
		return inputError("End-of-turn timeout must be an integer from 1000 to 10000 ms")
	}
	return nil
}

// TurnDetectionInfo is the turn timing the server reports. A nil field means
// the server did not publish it.
type TurnDetectionInfo struct {
	// Eagerness is "low", "medium", "high", "auto", or "unknown" for a value
	// the client does not know; empty when unpublished.
	Eagerness string
	// EndOfTurnTimeoutMS is the silence that ends a turn.
	EndOfTurnTimeoutMS *int
	// EndOfTurnMinSilenceMS is the least silence that can end a turn.
	EndOfTurnMinSilenceMS *int
	// InterruptMinWords is how many words interrupt a reply.
	InterruptMinWords *int
	// InterruptPhrases are the phrases that interrupt a reply.
	InterruptPhrases []string
	// AcknowledgmentPhrases are the phrases that do not interrupt a reply.
	AcknowledgmentPhrases []string
}

// Timeouts bound the connection, the wait for Prism to be ready for a reply,
// and the wait for an acknowledgment. A zero field uses the default.
type Timeouts struct {
	// Connect bounds opening the connection and its first event; 10 s by
	// default.
	Connect time.Duration `validate:"min=0"`
	// Ready bounds the wait for Prism to admit another reply; 30 s by
	// default.
	Ready time.Duration `validate:"min=0"`
	// Request bounds the wait for a request to be acknowledged; 10 s by
	// default.
	Request time.Duration `validate:"min=0"`
}

// The default timeouts.
const (
	defaultConnectTimeout = 10 * time.Second
	defaultReadyTimeout   = 30 * time.Second
	defaultRequestTimeout = 10 * time.Second
)

// withDefaults fills the timeouts left unset.
func (t Timeouts) withDefaults() Timeouts {
	if t.Connect == 0 {
		t.Connect = defaultConnectTimeout
	}
	if t.Ready == 0 {
		t.Ready = defaultReadyTimeout
	}
	if t.Request == 0 {
		t.Request = defaultRequestTimeout
	}
	return t
}

// audioChunk is a chunk of PCM to send, and the format it is in.
type audioChunk struct {
	data   []byte
	format rimepcm.Format
}

// defaultFormat is the format of audio sent with none named.
func defaultFormat() rimepcm.Format { return rimepcm.Format{SampleRate: rimepcm.WireRate, Channels: 1} }

// validate reports whether the chunk holds whole frames in a format Rime
// takes.
func (c audioChunk) validate() error {
	if !c.format.Valid() {
		return newError(KindAudioFormat, "Use PCM16 at 8, 16, 24 or 48 kHz, with one or two channels")
	}
	if len(c.data)%(2*c.format.Channels) != 0 {
		return newError(KindAudioFormat, "Audio must contain complete PCM16 frames as bytes")
	}
	return nil
}

// Filler is what Prism says while a tool runs. A tool with no filler uses the
// server's default.
type Filler struct {
	// Type is "none" to say nothing, "text" for a fixed line, or "auto" to let
	// Prism write a line.
	Type string `validate:"required,oneof=none text auto"`
	// Text is the fixed line, for the type "text": nonblank, and at most 500
	// characters.
	Text *string
	// Instructions guide the line Prism writes, for the type "auto": nonblank,
	// and at most 500 characters.
	Instructions *string
}

// validate reports whether the filler is one Prism takes.
func (f Filler) validate() error {
	switch {
	case !slices.Contains([]string{fillerNone, fillerText, fillerAuto}, f.Type):
		return inputError("Prism filler type must be none, text, or auto")
	case f.Type == fillerText && f.Text == nil:
		return inputError("Prism text filler requires text")
	case f.Type != fillerText && f.Text != nil:
		return inputError("Prism filler text is only allowed for type text")
	case f.Type != fillerAuto && f.Instructions != nil:
		return inputError("Prism filler instructions are only allowed for type auto")
	}
	for _, value := range []*string{f.Text, f.Instructions} {
		if value == nil {
			continue
		}
		if strings.Trim(*value, fillerWhitespace) == "" || utf8.RuneCountInString(*value) > 500 {
			return inputError("Prism filler text must be nonblank and at most 500 characters")
		}
		if !utf8.ValidString(*value) {
			return inputError("Prism filler text must be valid UTF-8")
		}
	}
	return nil
}

// toolDefinition is a tool as the session advertises it.
type toolDefinition struct {
	name        string
	parameters  map[string]any
	description string
	filler      *Filler
}

// historyItem is an item written to the conversation's history: a message, a
// past tool call, or the result of one.
type historyItem struct {
	// kind is "message", "function_call" or "function_call_output".
	kind string
	// id is the identity the item is given, empty to let the server name it.
	id string
	// role and text are a message's.
	role, text string
	// callID is the call a past call or a result belongs to.
	callID string
	// name and arguments are a past call's.
	name, arguments string
	// output is a result's.
	output string
}

// The kinds of history item.
const (
	itemMessage      = "message"
	itemFunctionCall = "function_call"
	itemFunctionOut  = "function_call_output"
)

// ResponseRef names a reply.
type ResponseRef struct {
	// SessionID is the session the reply belongs to.
	SessionID string
	// ResponseID is the reply's identity.
	ResponseID string
}

// ToolCallRef names a tool call.
type ToolCallRef struct {
	// SessionID is the session the call belongs to.
	SessionID string
	// ResponseID is the reply that made the call.
	ResponseID string
	// CallID is the call's identity.
	CallID string
}

// OutputRef names one generated message of a reply.
type OutputRef struct {
	// Response is the reply the message belongs to.
	Response ResponseRef
	// ItemID is the message's identity in the conversation.
	ItemID string
	// OutputIndex is the message's position in the reply.
	OutputIndex int
	// ContentIndex is the position of the content in the message, always 0.
	ContentIndex int
}

// SessionInfo holds the settings Prism last acknowledged.
type SessionInfo struct {
	// SettingsRevision counts the settings changes; nil when the server does
	// not publish it.
	SettingsRevision *int
	// Truncation is the history capacity policy; nil when unpublished.
	Truncation *Truncation
	// Speed is the speech speed; nil when unpublished.
	Speed *float64
	// Version is the server's version; empty when unpublished.
	Version string
	// Review is the speech review policy; nil when unpublished.
	Review *ReviewSettings
	// SessionID is the session's identity.
	SessionID string
	// Model is always "prism".
	Model string
	// Voice is the voice Prism speaks with.
	Voice string
	// InterruptOnSpeech reports whether the user's speech interrupts a reply.
	InterruptOnSpeech bool
	// ToolResultTimeout is how long Prism waits for a tool's result; 0 when
	// unpublished.
	ToolResultTimeout time.Duration
	// ToolContinuationTimeout is how long Prism waits to be asked to continue
	// after the results arrived; 0 when unpublished.
	ToolContinuationTimeout time.Duration
	// TurnDetection is the turn timing the server reports.
	TurnDetection TurnDetectionInfo
	// RequestID is the identity Prism's logs name the session by; empty when
	// the server does not publish it.
	RequestID string
	// ResponseMetadata are the labels replies carry, as Prism last confirmed
	// them; nil when the server does not publish them.
	ResponseMetadata map[string]string
}

// EventPayload is what a session event carries: one of the payload types
// below.
type EventPayload interface {
	// Kind names the payload, such as "response.started".
	Kind() string
}

// InputReady reports that Prism is ready to take another reply request.
type InputReady struct{}

// Kind implements EventPayload.
func (InputReady) Kind() string { return "input.ready" }

// SpeechHint reports that the user may have started speaking over a reply. It
// comes before SpeechStarted and can be a false alarm.
type SpeechHint struct {
	// AudioStartMS is where in the input audio the speech may have started.
	AudioStartMS int
}

// Kind implements EventPayload.
func (SpeechHint) Kind() string { return "speech.hint" }

// SpeechStarted reports that the user started speaking.
type SpeechStarted struct {
	// ItemID is the identity of the user's turn.
	ItemID string
	// AudioStartMS is where in the input audio the speech started.
	AudioStartMS int
}

// Kind implements EventPayload.
func (SpeechStarted) Kind() string { return "speech.started" }

// SpeechStopped reports that the user stopped speaking.
type SpeechStopped struct {
	// ItemID is the identity of the user's turn.
	ItemID string
	// AudioEndMS is where in the input audio the speech ended.
	AudioEndMS int
	// EndReason is what ended the turn: "semantic", "timeout", "silence",
	// "max_duration", "detector_changed", or "unknown" for a reason the client
	// does not know. Empty when the server did not publish it.
	EndReason string
}

// Kind implements EventPayload.
func (SpeechStopped) Kind() string { return "speech.stopped" }

// InputCommitted reports that the user's turn was committed to the
// conversation.
type InputCommitted struct {
	// ItemID is the identity of the user's turn.
	ItemID string
}

// Kind implements EventPayload.
func (InputCommitted) Kind() string { return "input.committed" }

// TranscriptDelta carries more of the transcript of the user's turn.
type TranscriptDelta struct {
	// ItemID is the identity of the user's turn.
	ItemID string
	// Delta is the text added.
	Delta string
	// Transcript is the whole transcript so far, revisions included; nil when
	// the server sent the addition only.
	Transcript *string
}

// Kind implements EventPayload.
func (TranscriptDelta) Kind() string { return "transcript.delta" }

// TranscriptFinal carries the completed transcript of the user's turn.
type TranscriptFinal struct {
	// ItemID is the identity of the user's turn.
	ItemID string
	// Text is the transcript.
	Text string
}

// Kind implements EventPayload.
func (TranscriptFinal) Kind() string { return "transcript.final" }

// TranscriptFailed reports that the user's turn could not be transcribed.
type TranscriptFailed struct {
	// ItemID is the identity of the user's turn.
	ItemID string
	// Error is the failure.
	Error Fault
}

// Kind implements EventPayload.
func (TranscriptFailed) Kind() string { return "transcript.failed" }

// The causes of a reply.
const (
	causeSpeech    = "speech"
	causeText      = "text"
	causeProactive = "proactive"
	causeTools     = "tools"
)

// ResponseStarted reports that a reply started.
type ResponseStarted struct {
	// SettingsRevision is the settings the reply runs with; nil when the
	// server does not publish it.
	SettingsRevision *int
	// Response names the reply.
	Response ResponseRef
	// Cause is what started the reply: "speech", "text", "proactive", "tools"
	// or "unknown".
	Cause string
	// Parent names the reply this one continues, for a tool continuation.
	Parent *ResponseRef
	// InputItemID is the user item the reply answers, when the server names
	// one.
	InputItemID string
	// InstructionItemID is the item holding the instruction of a requested
	// reply, when it has one.
	InstructionItemID string
	// Metadata are the session's labels when the reply began, Prism's own
	// left out.
	Metadata map[string]string
	// InputItemIDs are the user items the reply answers; empty when the server
	// names none.
	InputItemIDs []string
}

// Kind implements EventPayload.
func (ResponseStarted) Kind() string { return "response.started" }

// MessageStarted reports that a reply started a message.
type MessageStarted struct {
	// Output names the message.
	Output OutputRef
}

// Kind implements EventPayload.
func (MessageStarted) Kind() string { return "message.started" }

// TextDelta carries more of a message's text.
type TextDelta struct {
	// Output names the message.
	Output OutputRef
	// Delta is the text added.
	Delta string
}

// Kind implements EventPayload.
func (TextDelta) Kind() string { return "text.delta" }

// TextDone carries a message's whole text.
type TextDone struct {
	// Output names the message.
	Output OutputRef
	// Text is the message's text.
	Text string
}

// Kind implements EventPayload.
func (TextDone) Kind() string { return "text.done" }

// AudioDelta carries more of a message's audio: 16-bit mono PCM at 24 kHz.
type AudioDelta struct {
	// Output names the message.
	Output OutputRef
	// Audio is the PCM.
	Audio []byte
	// SampleRate is the audio's rate.
	SampleRate int
	// Channels is the audio's channel count.
	Channels int
}

// Kind implements EventPayload.
func (AudioDelta) Kind() string { return "audio.delta" }

// AudioDone reports that a message's audio ended.
type AudioDone struct {
	// Output names the message.
	Output OutputRef
}

// Kind implements EventPayload.
func (AudioDone) Kind() string { return "audio.done" }

// ToolCall reports a tool the model called.
type ToolCall struct {
	// Call names the call.
	Call ToolCallRef
	// ItemID is the call's identity in the conversation.
	ItemID string
	// Name is the tool called.
	Name string
	// Arguments are the call's arguments.
	Arguments map[string]any
	// raw is the arguments as the model wrote them.
	raw string
}

// Kind implements EventPayload.
func (ToolCall) Kind() string { return "tool.call" }

// ResponseAbandoned reports that a request given up on had been accepted: the
// reply it started is being canceled. It is the client's own notice, and can
// follow the reply's end when the two race.
type ResponseAbandoned struct {
	// Response names the reply.
	Response ResponseRef
}

// Kind implements EventPayload.
func (ResponseAbandoned) Kind() string { return "response.abandoned" }

// ResponseTruncation reports the history items left out of a reply's prompt.
// They are not deleted from the history.
type ResponseTruncation struct {
	// ItemsLeftOut counts the items left out.
	ItemsLeftOut int
	// FirstKeptItemID is the first item kept; empty when none earlier was.
	FirstKeptItemID string
}

// ResponseUsage is the token accounting of a reply.
type ResponseUsage struct {
	// InputTokens counts the prompt's tokens.
	InputTokens int
	// OutputTokens counts the reply's tokens.
	OutputTokens int
	// TotalTokens counts both.
	TotalTokens int
	// ContextWindow is the model's capacity.
	ContextWindow int
}

// ResponseEnded reports that a reply ended.
type ResponseEnded struct {
	// SettingsRevision is the settings the reply ran with; nil when the server
	// does not publish it.
	SettingsRevision *int
	// Response names the reply.
	Response ResponseRef
	// Status is "completed", "cancelled", "failed" or "unknown".
	Status string
	// Reason says why the reply ended that way, when the server does.
	Reason string
	// Truncation reports the items left out of the prompt; nil when none were
	// or the server does not say.
	Truncation *ResponseTruncation
	// Usage is the reply's token accounting; nil when the server sent none.
	Usage *ResponseUsage
}

// Kind implements EventPayload.
func (ResponseEnded) Kind() string { return "response.ended" }

// FaultEvent carries a failure Prism reported.
type FaultEvent struct {
	// Error is the failure.
	Error Fault
}

// Kind implements EventPayload.
func (FaultEvent) Kind() string { return "error" }

// ItemTruncated reports that Prism recorded how much of a message was heard.
// The text it keeps can differ from what was audible.
type ItemTruncated struct {
	// ItemID is the message's identity in the conversation.
	ItemID string
	// ContentIndex is the position of the content in the message.
	ContentIndex int
	// AudioEndMS is how much of the audio was heard.
	AudioEndMS int
	// KeptText is the text Prism keeps: nil when it does not say, empty when no
	// words remain.
	KeptText *string
}

// Kind implements EventPayload.
func (ItemTruncated) Kind() string { return "item.truncated" }

// Event is an event of a Prism session.
type Event struct {
	// SessionID is the session the event belongs to.
	SessionID string
	// EventID is the server's identity for the event; empty for the client's
	// own notices.
	EventID string
	// RequestID is the event id of the request the event answers, when it
	// answers one.
	RequestID string
	// Payload is what the event carries.
	Payload EventPayload
	// ServerTimeMS is the server's clock when it sent the event; nil when it
	// does not say.
	ServerTimeMS *int
}

// playbackInterrupted reports how much of a message was heard before it was
// cut off.
type playbackInterrupted struct {
	output     OutputRef
	audioEndMS int
}

// validate reports whether the position is one Prism takes.
func (p playbackInterrupted) validate() error {
	if p.audioEndMS < 0 || p.audioEndMS > math.MaxUint32 {
		return inputError("audio_end_ms must be an unsigned 32-bit integer")
	}
	return nil
}

// playbackFinished reports how much of a reply was played once it ended.
type playbackFinished struct {
	response      ResponseRef
	playedMS      *float64
	audibleTailMS *float64
}

// validate reports whether the times are ones Prism takes.
func (p playbackFinished) validate() error {
	for _, value := range []*float64{p.playedMS, p.audibleTailMS} {
		if value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0) || *value < 0) {
			return inputError("Playback times must be finite and nonnegative")
		}
	}
	return nil
}
