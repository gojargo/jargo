package tts

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/gojargo/jargo/processor"
)

// defaultZeroAudioContextLimit is how many contexts in a row may complete with
// no audio before the service is taken to be unable to speak at all.
const defaultZeroAudioContextLimit = 3

// SetZeroAudioContextLimit sets how many audio contexts in a row may complete
// without producing any audio before the service is reported unable to do its
// job. It catches a provider that accepts every request and stays silent, an
// unknown voice id being the usual case, which no error ever surfaces.
//
// Every silent context is reported whatever the limit is; on reaching it the
// service reports a permanent error instead, stops being given work, and the
// pipeline worker applies its unusable-processor policy. Zero reports silent
// contexts without ever writing the service off. It defaults to 3.
func (b *Base) SetZeroAudioContextLimit(n int) {
	b.silentMu.Lock()
	b.zeroAudioLimit = n
	b.silentMu.Unlock()
}

// SetUsable sets whether the service can be given work, and clears the run of
// silent contexts when it can. Whatever silence wrote the service off has been
// dealt with, so the count starts over.
func (b *Base) SetUsable(ctx context.Context, usable bool) {
	if usable {
		b.silentMu.Lock()
		b.zeroAudioRun = 0
		b.silentMu.Unlock()
	}
	b.Base.SetUsable(ctx, usable)
}

// recordContextAudioOutcome tracks whether contexts are producing audio, and
// acts when they stop.
//
// A provider can accept every request and return no audio at all, an unknown
// voice id say, without ever reporting an error. Every context that completes in
// silence is reported as an error, so application code hears about a turn that
// produced no speech as it happens. Enough of them in a row means the service is
// not going to speak again, so it is reported unable to do its job: it stops
// being given work, a switcher can fail over to another provider, and the
// pipeline worker applies its unusable-processor policy.
//
// Only contexts that complete are counted. An interrupted context is abandoned
// before it gets here.
func (b *Base) recordContextAudioOutcome(ctx context.Context, contextID string, receivedAudio bool) {
	if receivedAudio {
		b.silentMu.Lock()
		b.zeroAudioRun = 0
		b.silentMu.Unlock()
		return
	}

	// This context played nothing, so nothing will report the bot falling silent
	// to lift a pause taken for it, which happens when the pause was taken while
	// the context was still open. A bot still speaking is playing audio from
	// another context, whose own stopped frame is still to come.
	b.pauseStateMu.Lock()
	speaking := b.botSpeaking
	b.pauseStateMu.Unlock()
	if !speaking {
		b.maybeResumeFrameProcessing()
	}

	// An unusable service is deliberately not given work, so its silent contexts
	// say nothing new.
	if !b.Usable() {
		return
	}

	b.silentMu.Lock()
	b.zeroAudioRun++
	run, limit := b.zeroAudioRun, b.zeroAudioLimit
	b.silentMu.Unlock()

	slog.WarnContext(ctx, "audio context completed with no audio",
		"service", b.Name(), "context", contextID, "in_a_row", run)

	if limit > 0 && run >= limit {
		b.PushError(ctx, fmt.Sprintf("%d consecutive TTS contexts completed with no audio", run),
			nil, false, processor.ForceTreatAsPermanent())
		return
	}
	// A single silent context says nothing about whether the service will speak
	// again, so it is reported without costing the service its usability.
	b.PushError(ctx, fmt.Sprintf("TTS context %s completed with no audio", contextID), nil, false)
}
