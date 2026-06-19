// Package turntaking turns voice activity and end-of-turn detection into the
// frames that drive a conversation. It is a single pipeline processor placed
// just after the input transport: it watches incoming audio, decides when the
// user starts and finishes a turn, and interrupts the bot when the user barges
// in.
//
// It coordinates two analyzers from the audio packages: a vad.Analyzer for
// fast speech onset and silence, and an optional turn.Analyzer that tells a
// real end-of-turn from a mid-sentence pause. A "logical turn" spans from the
// first speech onset until the turn analyzer (or, as a fallback, sustained
// silence) declares it complete; pauses within a turn do not end it.
//
// Frames emitted downstream:
//
//   - UserStartedSpeakingFrame, once at the start of each turn.
//   - InterruptionFrame, alongside the start frame, so the output transport and
//     the rest of the pipeline drop any in-progress bot response (barge-in).
//   - UserStoppedSpeakingFrame, once the turn is complete.
//
// The analyzers run on 16 kHz mono audio; incoming audio at another rate is
// resampled before analysis. The original input audio is always forwarded
// downstream unchanged so the STT service sees it.
package turntaking

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// analyzerSampleRate is the rate the analyzers run at. Silero VAD and Smart Turn
// both operate on 16 kHz mono audio.
const analyzerSampleRate = 16000

// Config configures a Detector.
type Config struct {
	// VAD detects voice activity. Required.
	VAD vad.Analyzer
	// Turn detects end-of-turn. Optional: when nil, a turn ends as soon as VAD
	// reports sustained silence, with no model in the loop.
	Turn turn.Analyzer
}

// Detector is the turn-taking pipeline processor.
type Detector struct {
	*processor.Base

	vad  vad.Analyzer
	turn turn.Analyzer

	resampler *resample.Resampler
	inRate    int

	// turnActive spans a logical turn; vadSpeaking tracks the VAD's confirmed
	// speaking state, which also feeds the turn analyzer's speech flag.
	turnActive  bool
	vadSpeaking bool
}

// New builds a turn-taking Detector. The VAD analyzer is required.
func New(cfg Config) *Detector {
	d := &Detector{vad: cfg.VAD, turn: cfg.Turn}
	d.Base = processor.New("TurnTaking", d)
	return d
}

// ProcessFrame drives the analyzers from incoming audio and forwards frames.
func (d *Detector) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := d.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := d.start(fr); err != nil {
			return err
		}
		return d.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		if dir == processor.Downstream {
			return d.handleAudio(ctx, fr)
		}
		return d.PushFrame(ctx, f, dir)
	default:
		return d.PushFrame(ctx, f, dir)
	}
}

// Cleanup closes the analyzers.
func (d *Detector) Cleanup(ctx context.Context) error {
	if d.vad != nil {
		_ = d.vad.Close()
	}
	if d.turn != nil {
		_ = d.turn.Close()
	}
	return d.Base.Cleanup(ctx)
}

func (d *Detector) start(*frames.StartFrame) error {
	if err := d.vad.SetSampleRate(analyzerSampleRate); err != nil {
		return err
	}
	if d.turn != nil {
		d.turn.SetSampleRate(analyzerSampleRate)
	}
	return nil
}

// handleAudio forwards the audio frame, resamples a copy to the analyzer rate,
// and advances the VAD and turn analyzers.
func (d *Detector) handleAudio(ctx context.Context, f *frames.InputAudioRawFrame) error {
	// Always forward the original audio downstream so STT sees it.
	if err := d.PushFrame(ctx, f, processor.Downstream); err != nil {
		return err
	}

	pcm := d.toAnalyzerRate(f)

	// The speech flag fed to the turn analyzer is the VAD's confirmed state
	// before this frame's transition, matching how the analyzers compose.
	isSpeech := d.vadSpeaking

	state := d.vad.AnalyzeAudio(pcm)

	silenceComplete := false
	if d.turn != nil {
		if d.turn.AppendAudio(pcm, isSpeech) == turn.Complete {
			silenceComplete = true
		}
	}

	switch {
	case state == vad.StateSpeaking && !d.vadSpeaking:
		d.onSpeechStarted(ctx)
	case state == vad.StateQuiet && d.vadSpeaking:
		d.onSpeechStopped(ctx)
	}

	// The silence safety net can complete a turn even if VAD has not flipped
	// (for example a long trailing silence the model already saw).
	if silenceComplete && d.turnActive {
		d.endTurn(ctx)
	}
	return nil
}

// onSpeechStarted handles a VAD speech onset.
func (d *Detector) onSpeechStarted(ctx context.Context) {
	d.vadSpeaking = true
	if d.turnActive {
		return // a resume within the same turn; no new turn boundary
	}
	d.turnActive = true
	if d.turn != nil {
		d.turn.UpdateVADStartSecs(d.vad.Params().StartSecs)
	}
	// Start the turn and barge in: the interruption flushes any in-progress
	// bot response so the user can take over mid-sentence.
	_ = d.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	_ = d.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
}

// onSpeechStopped handles a VAD transition to silence: the turn ends only if
// the turn analyzer agrees (or there is no analyzer).
func (d *Detector) onSpeechStopped(ctx context.Context) {
	d.vadSpeaking = false
	if !d.turnActive {
		return
	}
	if d.turn == nil {
		d.endTurn(ctx)
		return
	}
	state, _, err := d.turn.AnalyzeEndOfTurn()
	if err != nil {
		slog.Error("turn analysis", "processor", d.Name(), "err", err)
		// Fall back to ending the turn so the conversation does not stall.
		d.endTurn(ctx)
		return
	}
	if state == turn.Complete {
		d.endTurn(ctx)
	}
	// Incomplete: the user paused mid-thought. Keep the turn open; it ends on a
	// later onset+stop or the silence safety net.
}

// endTurn closes the current logical turn.
func (d *Detector) endTurn(ctx context.Context) {
	d.turnActive = false
	d.vadSpeaking = false
	_ = d.PushFrame(ctx, frames.NewUserStoppedSpeakingFrame(), processor.Downstream)
}

// toAnalyzerRate returns the frame's audio resampled to the analyzer rate, mono.
func (d *Detector) toAnalyzerRate(f *frames.InputAudioRawFrame) []byte {
	if f.SampleRate == analyzerSampleRate {
		return f.Audio
	}
	if d.resampler == nil || d.inRate != f.SampleRate {
		d.resampler = resample.New(f.SampleRate, analyzerSampleRate, 1)
		d.inRate = f.SampleRate
	}
	return d.resampler.Process(f.Audio)
}
