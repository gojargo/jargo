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
		TextFrame: TextFrame{
			BaseDataFrame:            NewBaseDataFrame("LLMTextFrame"),
			Text:                     text,
			AppendToContext:          true,
			IncludesInterFrameSpaces: true,
		},
	}
}

// TTSTextFrame is a chunk of text a TTS service is speaking, aligned to audio
// playback. Text is the token as it was sent to the synthesizer; RawText, when
// set, is that same span in its original written form (for example "$42.50" for
// a token spoken as "forty two dollars and fifty cents"), so the assistant
// context records what was written rather than what was pronounced. A TTS
// service that reports word timings emits one per spoken word as its audio
// plays; because they flow in step with playback, an interruption leaves only
// the frames already emitted — the words actually spoken — in the context.
type TTSTextFrame struct {
	TextFrame
	// RawText is the original written form of this span; "" means use Text.
	RawText string
	// ContextID identifies the TTS context that produced this text; "" when unset.
	ContextID string
}

// NewTTSTextFrame builds a TTSTextFrame for the spoken token text, appending it
// to the LLM context by default. Word tokens do not carry their own inter-frame
// spacing, so a consumer joins them with a separator.
func NewTTSTextFrame(text string) *TTSTextFrame {
	return &TTSTextFrame{
		TextFrame: TextFrame{
			BaseDataFrame:   NewBaseDataFrame("TTSTextFrame"),
			Text:            text,
			AppendToContext: true,
		},
	}
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
		TextFrame: TextFrame{
			BaseDataFrame:   NewBaseDataFrame("TranscriptionFrame"),
			Text:            text,
			AppendToContext: true,
		},
		UserID:    userID,
		Timestamp: timestamp,
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
		TextFrame: TextFrame{
			BaseDataFrame:   NewBaseDataFrame("InterimTranscriptionFrame"),
			Text:            text,
			AppendToContext: true,
		},
		UserID:    userID,
		Timestamp: timestamp,
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
