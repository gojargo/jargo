package frames

import "fmt"

// TextFrame is a chunk of text flowing through the pipeline — emitted by LLM
// services and consumed by aggregators, TTS services and more. It is a data
// frame.
type TextFrame struct {
	BaseDataFrame
	// Text is the text content.
	Text string
	// SkipTTS reports whether a TTS service should skip this text. A nil value
	// means "unset": the decision is left to the frame flow.
	SkipTTS *bool
	// IncludesInterFrameSpaces reports whether any leading/trailing spaces
	// needed between adjacent frames are already part of Text.
	IncludesInterFrameSpaces bool
	// AppendToContext reports whether this text should be appended to the LLM
	// context. Defaults to true.
	AppendToContext bool
}

// NewTextFrame builds a TextFrame with the default field values.
func NewTextFrame(text string) *TextFrame {
	return &TextFrame{
		BaseDataFrame:   NewBaseDataFrame("TextFrame"),
		Text:            text,
		AppendToContext: true,
	}
}

// String implements fmt.Stringer.
func (f *TextFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, text: [%s])", f.Name(), formatPTS(f), f.Text)
}

// LLMTextFrame is a TextFrame produced by an LLM service. LLM output already
// includes any necessary inter-frame spaces.
type LLMTextFrame struct {
	TextFrame
}

// NewLLMTextFrame builds an LLMTextFrame.
func NewLLMTextFrame(text string) *LLMTextFrame {
	return &LLMTextFrame{
		BaseDataFrame:            NewBaseDataFrame("LLMTextFrame"),
		Text:                     text,
		AppendToContext:          true,
		IncludesInterFrameSpaces: true,
	}
}

// TTSTextFrame is a chunk of text a TTS service is speaking, aligned to audio
// playback. Text is the token as it was sent to the synthesizer; RawText, when
// set, is that same span in its original written form (for example "$42.50" for
// a token spoken as "forty two dollars and fifty cents"), so the assistant
// context records what was written rather than what was pronounced. A TTS
// service that reports word timings emits one per spoken word as its audio
// plays; because they flow in step with playback, an interruption leaves only
// the frames already emitted, the words actually spoken, in the context.
//
// It is an aggregated frame: TextType says what unit the text is, a whole
// sentence for a service that reports no timings and a single word for one that
// does, which is how a consumer tells a caption to render from a word to
// highlight within it.
type TTSTextFrame struct {
	AggregatedTextFrame
	// SegmentID is the ID of the AggregatedTextFrame this text is part of, when
	// the TTS service built it from one: a word carries the ID of the sentence
	// or other segment it was spoken in. Zero means it belongs to none.
	SegmentID uint64
	// Delivery is the delivery this text is spoken in, such as ["cheerful"],
	// and "emphasis" for stressed words; empty for the voice's baseline. Nil
	// when the text carries no delivery. Part of expressive conversation
	// (alpha).
	Delivery []string
}

// NewTTSTextFrame builds a TTSTextFrame for the spoken text, of the text type by
// and appended to the LLM context by default. Word tokens do not carry their
// own inter-frame spacing, so a consumer joins them with a separator.
func NewTTSTextFrame(text string, by TextType) *TTSTextFrame {
	f := &TTSTextFrame{AggregatedTextFrame: *NewAggregatedTextFrame(text, by)}
	f.BaseDataFrame = NewBaseDataFrame("TTSTextFrame")
	return f
}

// Original returns the text to record in the context: RawText when set,
// otherwise Text.
func (f *TTSTextFrame) Original() string {
	if f.RawText != "" {
		return f.RawText
	}
	return f.Text
}

// String implements fmt.Stringer.
func (f *TTSTextFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, text: [%s], raw: [%s])", f.Name(), formatPTS(f), f.Text, f.RawText)
}

// TTSSpeakFrame carries fixed text for the TTS service to speak directly,
// bypassing the LLM and the TTS sentence aggregator — the way to make the bot
// say a set phrase (a greeting, an acknowledgement). It is a data frame.
type TTSSpeakFrame struct {
	BaseDataFrame
	// Text is the exact text to speak.
	Text string
	// AppendToContext reports whether the spoken text is appended to the LLM
	// context as an assistant message. Defaults to true; set it false for
	// utterances that should not become part of the conversation (e.g. a wake
	// acknowledgement, which would otherwise start the context on an assistant
	// turn).
	AppendToContext bool
	// TextType is the text type of the text frames the TTS service pushes for
	// the spoken text, a sentence by default. A custom type lets them be told
	// apart, for example to keep them from the client with
	// rtvi.ObserverParams.SkipTextTypes. Empty is a sentence.
	TextType TextType
}

// NewTTSSpeakFrame builds a TTSSpeakFrame that speaks text as a sentence,
// appending it to the LLM context by default.
func NewTTSSpeakFrame(text string) *TTSSpeakFrame {
	return &TTSSpeakFrame{
		BaseDataFrame:   NewBaseDataFrame("TTSSpeakFrame"),
		Text:            text,
		AppendToContext: true,
		TextType:        TextTypeSentence,
	}
}

// BotBackchannelFrame is a short acknowledgment for the bot to say while the
// user talks, such as "Mm-hmm.".
//
// The TTS service speaks it like any TTSSpeakFrame, and its text frames have
// the text type TextTypeBackchannel. It stays out of the LLM context unless
// AppendToContext is set, and interruptions do not cut it off unless the frame
// is made interruptible with SetInterruptible(true).
type BotBackchannelFrame struct {
	TTSSpeakFrame
	UninterruptibleMixin
}

// NewBotBackchannelFrame builds a BotBackchannelFrame that says text.
func NewBotBackchannelFrame(text string) *BotBackchannelFrame {
	return &BotBackchannelFrame{
		BaseDataFrame: NewBaseDataFrame("BotBackchannelFrame"),
		Text:          text,
		TextType:      TextTypeBackchannel,
	}
}

// String implements fmt.Stringer.
func (f *TTSSpeakFrame) String() string {
	return fmt.Sprintf("%s(text: [%s])", f.Name(), f.Text)
}

// TranscriptionFrame carries a finalized speech transcription for a user.
type TranscriptionFrame struct {
	TextFrame
	// UserID identifies the user who spoke.
	UserID string
	// Timestamp is when the transcription occurred.
	Timestamp string
	// Language is the detected or specified language as a BCP-47 tag; "" when
	// unset.
	Language string
	// Result is the raw result from the STT service, if available.
	Result any
	// Finalized reports whether this is the final transcription for an
	// utterance, for STT services that signal commit/finalize.
	Finalized bool
}

// NewTranscriptionFrame builds a TranscriptionFrame.
func NewTranscriptionFrame(text, userID, timestamp string) *TranscriptionFrame {
	return &TranscriptionFrame{
		BaseDataFrame:   NewBaseDataFrame("TranscriptionFrame"),
		Text:            text,
		AppendToContext: true,
		UserID:          userID,
		Timestamp:       timestamp,
	}
}

// String implements fmt.Stringer.
func (f *TranscriptionFrame) String() string {
	return fmt.Sprintf("%s(user: %s, text: [%s], language: %s, timestamp: %s)",
		f.Name(), f.UserID, f.Text, f.Language, f.Timestamp)
}

// EagerTranscriptionFrame carries the transcript of a turn a service predicts
// has ended, before it commits to that.
//
// Some transcribers report a provisional end of turn ahead of the real one, so a
// reply can be generated during the gap. The prediction may be withdrawn (see
// EagerEndOfTurnCancelFrame) and the committed transcript may differ from this
// one, so nothing produced from it may reach the user or the conversation until
// it is confirmed.
//
// It is not a TranscriptionFrame: nothing that acts on a committed transcript
// should act on a predicted one.
type EagerTranscriptionFrame struct {
	TextFrame
	// UserID identifies the user who spoke.
	UserID string
	// Timestamp is when the eager end of turn occurred.
	Timestamp string
	// Language is the detected or specified language as a BCP-47 tag; "" when
	// unset.
	Language string
	// Result is the raw result from the STT service, if available.
	Result any
}

// NewEagerTranscriptionFrame builds an EagerTranscriptionFrame.
func NewEagerTranscriptionFrame(text, userID, timestamp string) *EagerTranscriptionFrame {
	return &EagerTranscriptionFrame{
		BaseDataFrame: NewBaseDataFrame("EagerTranscriptionFrame"),
		Text:          text,
		UserID:        userID,
		Timestamp:     timestamp,
	}
}

// String implements fmt.Stringer.
func (f *EagerTranscriptionFrame) String() string {
	return fmt.Sprintf("%s(user: %s, text: [%s], language: %s, timestamp: %s)",
		f.Name(), f.UserID, f.Text, f.Language, f.Timestamp)
}

// UserBackchannelFrame is a short acknowledgment the user said while the bot
// talks, such as "mhm". It is pushed in place of a transcription, so the bot
// keeps talking and the acknowledgment stays out of the LLM context. It is a
// data frame.
type UserBackchannelFrame struct {
	BaseDataFrame
	// Text is what the user said.
	Text string
	// UserID identifies the user who spoke.
	UserID string
	// Timestamp is when the user said it.
	Timestamp string
}

// NewUserBackchannelFrame builds a UserBackchannelFrame.
func NewUserBackchannelFrame(text, userID, timestamp string) *UserBackchannelFrame {
	return &UserBackchannelFrame{
		BaseDataFrame: NewBaseDataFrame("UserBackchannelFrame"),
		Text:          text,
		UserID:        userID,
		Timestamp:     timestamp,
	}
}

// String implements fmt.Stringer.
func (f *UserBackchannelFrame) String() string {
	return fmt.Sprintf("%s(user: %s, text: [%s], timestamp: %s)", f.Name(), f.UserID, f.Text, f.Timestamp)
}

// InterimTranscriptionFrame carries a partial (non-final) speech transcription
// for a user.
type InterimTranscriptionFrame struct {
	TextFrame
	// UserID identifies the user who spoke.
	UserID string
	// Timestamp is when the interim transcription occurred.
	Timestamp string
	// Language is the detected or specified language as a BCP-47 tag; "" when
	// unset.
	Language string
	// Result is the raw result from the STT service, if available.
	Result any
}

// NewInterimTranscriptionFrame builds an InterimTranscriptionFrame.
func NewInterimTranscriptionFrame(text, userID, timestamp string) *InterimTranscriptionFrame {
	return &InterimTranscriptionFrame{
		BaseDataFrame:   NewBaseDataFrame("InterimTranscriptionFrame"),
		Text:            text,
		AppendToContext: true,
		UserID:          userID,
		Timestamp:       timestamp,
	}
}

// String implements fmt.Stringer.
func (f *InterimTranscriptionFrame) String() string {
	return fmt.Sprintf("%s(user: %s, text: [%s], language: %s, timestamp: %s)",
		f.Name(), f.UserID, f.Text, f.Language, f.Timestamp)
}

// Compile-time interface checks.
var (
	_ DataFrame = (*TextFrame)(nil)
	_ DataFrame = (*LLMTextFrame)(nil)
	_ DataFrame = (*TTSTextFrame)(nil)
	_ DataFrame = (*TTSSpeakFrame)(nil)
	_ DataFrame = (*BotBackchannelFrame)(nil)
	_ DataFrame = (*UserBackchannelFrame)(nil)
	_ DataFrame = (*TranscriptionFrame)(nil)
	_ DataFrame = (*InterimTranscriptionFrame)(nil)
	_ DataFrame = (*EagerTranscriptionFrame)(nil)
)

// TextType is the kind of text a frame holds: a sentence, a word, or a custom
// type such as "code", which a pattern-matching aggregator gives the text it
// cuts out.
//
// TextTypeExpressionSound and TextTypeExpressionPause are part of expressive
// conversation (alpha): a sound or a silence an LLM writes into its reply,
// which isn't spoken as words. The text is the sound's name, or the pause's
// duration in milliseconds.
type TextType string

// The built-in text types.
const (
	// TextTypeSentence is text aggregated up to a sentence boundary.
	TextTypeSentence TextType = "sentence"
	// TextTypeToken is each token passed through as it arrives.
	TextTypeToken TextType = "token"
	// TextTypeWord is text a word at a time.
	TextTypeWord TextType = "word"
	// TextTypeBackchannel is a short acknowledgment the bot says while the user
	// talks, such as "Mm-hmm.".
	TextTypeBackchannel TextType = "backchannel"
	// TextTypeExpressionSound is a sound an LLM wrote into its reply, such as a
	// laugh, which the TTS service performed. The text is the sound's name.
	TextTypeExpressionSound TextType = "expression_sound"
	// TextTypeExpressionPause is a pause an LLM wrote into its reply, which the
	// TTS service performed. The text is its duration in milliseconds.
	TextTypeExpressionPause TextType = "expression_pause"
	// AnyTextType is not a kind of text but a wildcard standing for every one
	// of them, for something registered against a type that should apply
	// whatever the text is.
	AnyTextType TextType = "*"
)

// AggregationType is the former name of TextType.
//
// Deprecated: use TextType.
type AggregationType = TextType

// The former names of the built-in text types.
const (
	// Deprecated: use TextTypeSentence.
	AggregationSentence = TextTypeSentence
	// Deprecated: use TextTypeToken.
	AggregationToken = TextTypeToken
	// Deprecated: use TextTypeWord.
	AggregationWord = TextTypeWord
	// Deprecated: use AnyTextType.
	AnyAggregation = AnyTextType
)

// AggregatedTextFrame is a run of TextFrames aggregated into one unit for
// synthesis, carrying what kind of text they make.
type AggregatedTextFrame struct {
	TextFrame
	// TextType is the kind of text, such as a sentence, a word or a custom type
	// like "code".
	TextType TextType
	// ContextID identifies the synthesis context this text was sent on.
	ContextID string
	// RawText is the full matched text including any pattern delimiters, set
	// when the frame came from a pattern match such as a code block. Empty for
	// an ordinary sentence aggregation.
	RawText string
	// WillBeSpoken reports whether the TTS service will speak this frame. The
	// service sets it just before synthesis.
	WillBeSpoken bool
}

// NewAggregatedTextFrame builds an AggregatedTextFrame of the given text type.
func NewAggregatedTextFrame(text string, by TextType) *AggregatedTextFrame {
	f := &AggregatedTextFrame{
		TextFrame: *NewTextFrame(text),
		TextType:  by,
	}
	f.BaseDataFrame = NewBaseDataFrame("AggregatedTextFrame")
	return f
}

// String implements fmt.Stringer.
func (f *AggregatedTextFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, text: [%s], text_type: %s)", f.Name(), formatPTS(f), f.Text, f.TextType)
}

// AggregatedTextProgressFrame accompanies each TTSTextFrame during
// word-timestamp playback, carrying the spoken-so-far and remaining text of the
// AggregatedTextFrame being spoken. It lets a consumer highlight words as they
// are heard without reaching into the sequencer's state.
type AggregatedTextProgressFrame struct {
	BaseDataFrame
	// SegmentID identifies the AggregatedTextFrame being spoken.
	SegmentID uint64
	// ContextID is the synthesis context the text belongs to.
	ContextID string
	// Text is the full original text of the frame being spoken.
	Text string
	// TextType is the text type of that frame.
	TextType TextType
	// AccumulatedText is what has been spoken so far, including the current word.
	AccumulatedText string
	// RemainingText is what has not been spoken yet.
	RemainingText string
	// Delivery is the delivery of the word just spoken, which changes within
	// the text where its delivery does. Nil when the text carries no delivery.
	// Part of expressive conversation (alpha).
	Delivery []string
}

// NewAggregatedTextProgressFrame builds an AggregatedTextProgressFrame.
func NewAggregatedTextProgressFrame(segmentID uint64, contextID, text string,
	by TextType, accumulated, remaining string,
) *AggregatedTextProgressFrame {
	return &AggregatedTextProgressFrame{
		BaseDataFrame:   NewBaseDataFrame("AggregatedTextProgressFrame"),
		SegmentID:       segmentID,
		ContextID:       contextID,
		Text:            text,
		TextType:        by,
		AccumulatedText: accumulated,
		RemainingText:   remaining,
	}
}

// String implements fmt.Stringer.
func (f *AggregatedTextProgressFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, spoken: [%s], remaining: [%s])",
		f.Name(), formatPTS(f), f.AccumulatedText, f.RemainingText)
}

// TranslationFrame carries a translated transcription for a user, distinct from
// the transcription it was translated from: a provider that transcribes and
// translates reports both, and only the transcription is the user's own words.
type TranslationFrame struct {
	TextFrame
	// UserID identifies the user who spoke.
	UserID string
	// Timestamp is when the translation occurred.
	Timestamp string
	// Language is the language translated into, as a BCP-47 tag; "" when unset.
	Language string
}

// NewTranslationFrame builds a TranslationFrame.
func NewTranslationFrame(text, userID, timestamp string) *TranslationFrame {
	return &TranslationFrame{
		BaseDataFrame:   NewBaseDataFrame("TranslationFrame"),
		Text:            text,
		AppendToContext: true,
		UserID:          userID,
		Timestamp:       timestamp,
	}
}

// String implements fmt.Stringer.
func (f *TranslationFrame) String() string {
	return fmt.Sprintf("%s(user: %s, text: [%s], language: %s, timestamp: %s)",
		f.Name(), f.UserID, f.Text, f.Language, f.Timestamp)
}

// InputTextRawFrame is text arriving from a transport as input, usually because
// the user typed it or an application injected it, and meant for the LLM the
// same way spoken input is. It is the text counterpart of InputAudioRawFrame,
// and a system frame so it reaches the pipeline ahead of the queued conversation.
type InputTextRawFrame struct {
	BaseSystemFrame
	// Text is the text that arrived.
	Text string
}

// NewInputTextRawFrame builds an InputTextRawFrame.
func NewInputTextRawFrame(text string) *InputTextRawFrame {
	return &InputTextRawFrame{
		BaseSystemFrame: NewBaseSystemFrame("InputTextRawFrame"),
		Text:            text,
	}
}

// String implements fmt.Stringer.
func (f *InputTextRawFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, source: %s, text: [%s])",
		f.Name(), formatPTS(f), f.TransportSource(), f.Text)
}
