package tts

// PauseOptions configures pausing the service's frame handling while the audio
// for a turn is generated. It is off unless asked for.
type PauseOptions struct {
	// Enabled holds incoming frames after a reply until every audio context
	// open for it has ended. Services whose audio does not carry its request's
	// context ID need it: it stops the next request's audio being taken for
	// this one's. Frames stay queued, in order, while the service is paused.
	Enabled bool
}

// SetPauseFrameProcessing configures pausing frame handling while a turn's audio
// is generated. Call it before the pipeline starts.
func (b *Base) SetPauseFrameProcessing(o PauseOptions) {
	b.pauseStateMu.Lock()
	b.pauseOpts = o
	b.pauseStateMu.Unlock()
}

// holdFramesWhileAudioContextsOpen holds incoming frames while an audio context
// is open.
//
// Pausing is for services whose audio does not carry its request's context ID:
// they label incoming audio with the context still open, so the next request
// waits for every open context to end. Contexts play in order, so audio from
// different requests never overlaps. A context ends once its audio has all
// arrived, or once it has gone the stop-frame timeout without audio for a
// provider that never ends it, so the frames held are always released.
func (b *Base) holdFramesWhileAudioContextsOpen() {
	b.pauseStateMu.Lock()
	enabled := b.pauseOpts.Enabled
	b.pauseStateMu.Unlock()
	if enabled && b.hasOpenAudioContexts() {
		b.PauseProcessingFrames()
	}
}

// releaseFramesIfAudioContextsEnded lets held frames through once no audio
// context is open.
func (b *Base) releaseFramesIfAudioContextsEnded() {
	b.pauseStateMu.Lock()
	enabled := b.pauseOpts.Enabled
	b.pauseStateMu.Unlock()
	if enabled && !b.hasOpenAudioContexts() {
		b.ResumeProcessingFrames()
	}
}
