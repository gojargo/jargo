package tts

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// defaultPauseWatchdogTimeout is how long the watchdog waits for audio to be
// confirmed playing before it force-resumes frame handling.
const defaultPauseWatchdogTimeout = 3 * time.Second

// PauseOptions configures pausing the service's frame handling while the audio
// for a turn is generated. It is off unless asked for.
type PauseOptions struct {
	// Enabled pauses frame handling once the text of a turn has been sent to
	// the provider, until the audio for it has finished playing. It stops the
	// next turn's text being synthesized over the audio for this one. Frames
	// stay queued, in order, while the service is paused.
	Enabled bool
	// WatchdogTimeout force-resumes frame handling, and reports a non-fatal
	// error, when nothing confirms the audio is playing within this long of the
	// pause. It covers a turn that produces no audio at all, which would
	// otherwise leave the service paused for good. Zero defaults to 3s. It is
	// not armed when the audio was already confirmed playing before the pause,
	// which is the usual case for a streaming provider.
	WatchdogTimeout time.Duration
}

// SetPauseFrameProcessing configures pausing frame handling while a turn's audio
// is generated. Call it before the pipeline starts.
func (b *Base) SetPauseFrameProcessing(o PauseOptions) {
	if o.WatchdogTimeout <= 0 {
		o.WatchdogTimeout = defaultPauseWatchdogTimeout
	}
	b.pauseStateMu.Lock()
	b.pauseOpts = o
	b.pauseStateMu.Unlock()
}

// maybePauseFrameProcessing pauses frame handling if text was sent to the
// provider for this turn. Nothing was sent for a turn that was only a function
// call, and there is no audio to wait for.
func (b *Base) maybePauseFrameProcessing(ctx context.Context) {
	b.pauseStateMu.Lock()
	opts, processing, speaking := b.pauseOpts, b.processingText, b.botSpeaking
	b.pauseStateMu.Unlock()

	if !processing || !opts.Enabled {
		return
	}
	b.PauseProcessingFrames()
	b.cancelPauseWatchdog()
	if !speaking {
		// Audio for this turn is not confirmed playing. A streaming provider
		// usually starts playback while the model is still generating, so the
		// ordinary path resumes once playback finishes. Otherwise arm the
		// watchdog, so a turn that produces no audio cannot pause the service
		// for good.
		b.startPauseWatchdog(ctx, opts.WatchdogTimeout)
	}
}

// maybeResumeFrameProcessing releases frame handling paused for a turn's audio.
func (b *Base) maybeResumeFrameProcessing() {
	b.cancelPauseWatchdog()

	b.pauseStateMu.Lock()
	enabled := b.pauseOpts.Enabled
	b.pauseStateMu.Unlock()
	if enabled {
		b.ResumeProcessingFrames()
	}
}

// startPauseWatchdog arms the watchdog that force-resumes frame handling when
// nothing confirms the audio is playing.
func (b *Base) startPauseWatchdog(ctx context.Context, timeout time.Duration) {
	// Detached from the frame's context: the watchdog outlives the frame that
	// armed it, and is stopped by cancelPauseWatchdog instead.
	wctx, cancel := context.WithCancel(context.WithoutCancel(ctx))

	b.pauseStateMu.Lock()
	b.watchdogCancel = cancel
	b.pauseStateMu.Unlock()

	go func() {
		select {
		case <-time.After(timeout):
		case <-wctx.Done():
			return
		}

		b.pauseStateMu.Lock()
		b.watchdogCancel = nil
		b.pauseStateMu.Unlock()

		msg := fmt.Sprintf("nothing confirmed the bot was speaking within %s of pausing frame processing "+
			"(a turn that produced no audio, say), force-resuming", timeout)
		slog.Warn("tts pause watchdog fired", "service", b.Name(), "timeout", timeout)
		b.ResumeProcessingFrames()
		b.PushError(wctx, msg, nil, false)
	}()
}

// cancelPauseWatchdog stops the watchdog, if one is armed.
func (b *Base) cancelPauseWatchdog() {
	b.pauseStateMu.Lock()
	cancel := b.watchdogCancel
	b.watchdogCancel = nil
	b.pauseStateMu.Unlock()
	if cancel != nil {
		cancel()
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
