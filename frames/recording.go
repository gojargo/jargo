package frames

// AudioBufferStartRecordingFrame instructs audio-buffer processors to start
// recording. It is a control frame and is uninterruptible so a barge-in does not
// drop it.
type AudioBufferStartRecordingFrame struct {
	BaseControlFrame
	UninterruptibleMixin
}

// NewAudioBufferStartRecordingFrame builds an AudioBufferStartRecordingFrame.
func NewAudioBufferStartRecordingFrame() *AudioBufferStartRecordingFrame {
	return &AudioBufferStartRecordingFrame{
		BaseControlFrame: NewBaseControlFrame("AudioBufferStartRecordingFrame"),
	}
}

// AudioBufferStopRecordingFrame instructs audio-buffer processors to stop
// recording and flush the buffered audio. It is a control frame and is
// uninterruptible.
type AudioBufferStopRecordingFrame struct {
	BaseControlFrame
	UninterruptibleMixin
}

// NewAudioBufferStopRecordingFrame builds an AudioBufferStopRecordingFrame.
func NewAudioBufferStopRecordingFrame() *AudioBufferStopRecordingFrame {
	return &AudioBufferStopRecordingFrame{
		BaseControlFrame: NewBaseControlFrame("AudioBufferStopRecordingFrame"),
	}
}

// Compile-time interface checks.
var (
	_ ControlFrame    = (*AudioBufferStartRecordingFrame)(nil)
	_ Uninterruptible = (*AudioBufferStartRecordingFrame)(nil)
	_ ControlFrame    = (*AudioBufferStopRecordingFrame)(nil)
	_ Uninterruptible = (*AudioBufferStopRecordingFrame)(nil)
)
