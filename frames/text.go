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
	_ DataFrame = (*TranscriptionFrame)(nil)
	_ DataFrame = (*InterimTranscriptionFrame)(nil)
)
