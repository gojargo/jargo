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
// It is an aggregated frame: AggregatedBy says what unit the text is, a whole
// sentence for a service that reports no timings and a single word for one that
// does, which is how a consumer tells a caption to render from a word to
// highlight within it.
type TTSTextFrame struct {
	AggregatedTextFrame
}

// NewTTSTextFrame builds a TTSTextFrame for the spoken text, aggregated by by
// and appended to the LLM context by default. Word tokens do not carry their
// own inter-frame spacing, so a consumer joins them with a separator.
func NewTTSTextFrame(text string, by AggregationType) *TTSTextFrame {
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
}

// NewTTSSpeakFrame builds a TTSSpeakFrame that speaks text, appending it to the
// LLM context by default.
func NewTTSSpeakFrame(text string) *TTSSpeakFrame {
	return &TTSSpeakFrame{
		BaseDataFrame:   NewBaseDataFrame("TTSSpeakFrame"),
		Text:            text,
		AppendToContext: true,
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
	_ DataFrame = (*TranscriptionFrame)(nil)
	_ DataFrame = (*InterimTranscriptionFrame)(nil)
)

// AggregationType names how a stream of text was aggregated before synthesis.
type AggregationType string

// The built-in aggregation types.
const (
	// AggregationSentence aggregates text up to a sentence boundary.
	AggregationSentence AggregationType = "sentence"
	// AggregationToken passes each token through as it arrives.
	AggregationToken AggregationType = "token"
	// AggregationWord aggregates text a word at a time.
	AggregationWord AggregationType = "word"
	// AnyAggregation is not a way of aggregating text but a wildcard standing for
	// every one of them, for something registered against a type that should
	// apply whatever the unit was grouped by.
	AnyAggregation AggregationType = "*"
)

// AggregatedTextFrame is a run of TextFrames aggregated into one unit for
// synthesis, carrying how they were aggregated.
type AggregatedTextFrame struct {
	TextFrame
	// AggregatedBy is how the text was aggregated.
	AggregatedBy AggregationType
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

// NewAggregatedTextFrame builds an AggregatedTextFrame aggregated by by.
func NewAggregatedTextFrame(text string, by AggregationType) *AggregatedTextFrame {
	f := &AggregatedTextFrame{
		TextFrame:    *NewTextFrame(text),
		AggregatedBy: by,
	}
	f.BaseDataFrame = NewBaseDataFrame("AggregatedTextFrame")
	return f
}

// String implements fmt.Stringer.
func (f *AggregatedTextFrame) String() string {
	return fmt.Sprintf("%s(pts: %s, text: [%s], by: %s)", f.Name(), formatPTS(f), f.Text, f.AggregatedBy)
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
	// AggregatedBy is how that text was aggregated.
	AggregatedBy AggregationType
	// AccumulatedText is what has been spoken so far, including the current word.
	AccumulatedText string
	// RemainingText is what has not been spoken yet.
	RemainingText string
}

// NewAggregatedTextProgressFrame builds an AggregatedTextProgressFrame.
func NewAggregatedTextProgressFrame(segmentID uint64, contextID, text string,
	by AggregationType, accumulated, remaining string,
) *AggregatedTextProgressFrame {
	return &AggregatedTextProgressFrame{
		BaseDataFrame:   NewBaseDataFrame("AggregatedTextProgressFrame"),
		SegmentID:       segmentID,
		ContextID:       contextID,
		Text:            text,
		AggregatedBy:    by,
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
