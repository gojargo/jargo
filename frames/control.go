package frames

import "fmt"

// EndFrame indicates the pipeline has ended and processors should shut down. As
// a control frame it is received in order, after preceding frames are flushed.
// It is uninterruptible so it survives an interruption and the pipeline always
// shuts down cleanly.
type EndFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// Reason is an optional reason for ending the pipeline.
	Reason any
}

// NewEndFrame builds an EndFrame.
func NewEndFrame() *EndFrame {
	return &EndFrame{BaseControlFrame: NewBaseControlFrame("EndFrame")}
}

// String implements fmt.Stringer.
func (f *EndFrame) String() string {
	return fmt.Sprintf("%s(reason: %v)", f.Name(), f.Reason)
}

// LLMFullResponseStartFrame marks the beginning of an LLM response, followed by
// one or more TextFrames and a final LLMFullResponseEndFrame. It is a control
// frame.
type LLMFullResponseStartFrame struct {
	BaseControlFrame
	// SkipTTS, when set, reports whether the response should skip TTS. A nil
	// value means "unset".
	SkipTTS *bool
}

// NewLLMFullResponseStartFrame builds an LLMFullResponseStartFrame.
func NewLLMFullResponseStartFrame() *LLMFullResponseStartFrame {
	return &LLMFullResponseStartFrame{BaseControlFrame: NewBaseControlFrame("LLMFullResponseStartFrame")}
}

// LLMFullResponseEndFrame marks the end of an LLM response. It is a control
// frame.
type LLMFullResponseEndFrame struct {
	BaseControlFrame
	// SkipTTS, when set, reports whether the response should skip TTS. A nil
	// value means "unset".
	SkipTTS *bool
}

// NewLLMFullResponseEndFrame builds an LLMFullResponseEndFrame.
func NewLLMFullResponseEndFrame() *LLMFullResponseEndFrame {
	return &LLMFullResponseEndFrame{BaseControlFrame: NewBaseControlFrame("LLMFullResponseEndFrame")}
}

// TTSStartedFrame marks the beginning of a TTS response. The following
// TTSAudioRawFrames are part of the response until a TTSStoppedFrame. It is a
// control frame.
type TTSStartedFrame struct {
	BaseControlFrame
	// ContextID identifies this TTS context; "" when unset.
	ContextID string
	// AppendToContext reports whether the spoken text for this response will be
	// appended to the LLM context. Defaults to true.
	AppendToContext bool
}

// NewTTSStartedFrame builds a TTSStartedFrame.
func NewTTSStartedFrame() *TTSStartedFrame {
	return &TTSStartedFrame{BaseControlFrame: NewBaseControlFrame("TTSStartedFrame"), AppendToContext: true}
}

// TTSStoppedFrame marks the end of a TTS response. It is a control frame.
type TTSStoppedFrame struct {
	BaseControlFrame
	// ContextID identifies this TTS context; "" when unset.
	ContextID string
}

// NewTTSStoppedFrame builds a TTSStoppedFrame.
func NewTTSStoppedFrame() *TTSStoppedFrame {
	return &TTSStoppedFrame{BaseControlFrame: NewBaseControlFrame("TTSStoppedFrame")}
}

// Compile-time interface checks.
var (
	_ ControlFrame    = (*EndFrame)(nil)
	_ Uninterruptible = (*EndFrame)(nil)
	_ ControlFrame    = (*LLMFullResponseStartFrame)(nil)
	_ ControlFrame    = (*LLMFullResponseEndFrame)(nil)
	_ ControlFrame    = (*TTSStartedFrame)(nil)
	_ ControlFrame    = (*TTSStoppedFrame)(nil)
)
