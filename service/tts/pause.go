package tts

import (
	"time"
)

// PauseOptions configures pausing the service's frame handling while the audio
// for a turn is generated. It is off unless asked for.
type PauseOptions struct {
	// Enabled pauses frame handling once the text of a turn has been sent to
	// the provider, until the audio for it has finished playing. It stops the
	// next turn's text being synthesized over the audio for this one. Frames
	// stay queued, in order, while the service is paused.
	Enabled bool

	// WatchdogTimeout is unused.
	//
	// Deprecated: it has no replacement. Frame handling is paused only while
	// there is audio still to be played, so the pause is lifted by the bot
	// falling silent or by the audio context completing with nothing in it, and
	// no timer is needed to break it.
	WatchdogTimeout time.Duration
}

// SetPauseFrameProcessing configures pausing frame handling while a turn's audio
// is generated. Call it before the pipeline starts.
func (b *Base) SetPauseFrameProcessing(o PauseOptions) {
	b.pauseStateMu.Lock()
	b.pauseOpts = o
	b.pauseStateMu.Unlock()
}

// maybePauseFrameProcessing holds incoming frames until the audio for this turn
// has played.
//
// The pause waits for the bot to stop speaking, so it is only taken while there
// is playback to wait for: the bot speaking, or an audio context still open that
// may yet produce audio. A turn with neither, one whose contexts all completed in
// silence or whose playback finished before its text ran out, has nothing left to
// resume it, and pausing would hold frame handling for good.
//
// Nothing was sent for a turn that was only a function call, so there is no audio
// to wait for there either.
func (b *Base) maybePauseFrameProcessing() {
	b.pauseStateMu.Lock()
	enabled, processing, speaking := b.pauseOpts.Enabled, b.processingText, b.botSpeaking
	b.pauseStateMu.Unlock()

	if !processing || !enabled {
		return
	}
	// With no audio playing and none on its way, nothing will report the bot
	// falling silent, so the pause would never be lifted.
	if !speaking && !b.hasOpenAudioContexts() {
		return
	}
	b.PauseProcessingFrames()
}

// maybeResumeFrameProcessing releases frame handling paused for a turn's audio.
func (b *Base) maybeResumeFrameProcessing() {
	b.pauseStateMu.Lock()
	enabled := b.pauseOpts.Enabled
	b.pauseStateMu.Unlock()
	if enabled {
		b.ResumeProcessingFrames()
	}
}

// setProcessingText records whether text for this turn was sent to the provider,
// which is what decides there is audio worth pausing for.
func (b *Base) setProcessingText(v bool) {
	b.pauseStateMu.Lock()
	b.processingText = v
	b.pauseStateMu.Unlock()
}

// isProcessingText reports whether text for this turn was sent to the provider.
func (b *Base) isProcessingText() bool {
	b.pauseStateMu.Lock()
	defer b.pauseStateMu.Unlock()
	return b.processingText
}

// setBotSpeaking records whether the bot's audio is confirmed playing.
func (b *Base) setBotSpeaking(v bool) {
	b.pauseStateMu.Lock()
	b.botSpeaking = v
	b.pauseStateMu.Unlock()
}
