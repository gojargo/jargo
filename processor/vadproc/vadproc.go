// Package vadproc is the voice-activity-detection pipeline processor. It drives
// a vad.Analyzer over incoming audio and emits the raw VAD frames the turn
// subsystem consumes: VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame
// on speech onset/offset, plus a periodic UserSpeakingFrame while the user
// speaks. It does not decide turns — that is the turns package's job.
//
// Place it just after the input transport. The analyzer runs at the input sample
// rate when it supports it — so Silero runs natively on 8 kHz telephony audio,
// not an upsampled copy — and only resamples to 16 kHz when the input rate is
// rejected. The original input audio is always forwarded downstream unchanged so
// STT and the turn analyzer still see it.
package vadproc

import (
	"context"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// analyzerSampleRate is the fallback analyzer rate, used when the input rate is
// not one the analyzer accepts (Silero also runs natively at 8 kHz).
const analyzerSampleRate = 16000

// defaultSpeechActivityPeriod is how often a UserSpeakingFrame is emitted while
// the user is speaking.
const defaultSpeechActivityPeriod = 200 * time.Millisecond

// Config configures a Processor.
type Config struct {
	// VAD detects voice activity. Required.
	VAD vad.Analyzer
	// SpeechActivityPeriod is how often a UserSpeakingFrame is emitted while the
	// user is speaking; 0 uses 200ms, a negative value disables the keepalive.
	SpeechActivityPeriod time.Duration
}

// Processor is the VAD pipeline processor.
type Processor struct {
	*processor.Base

	vad          vad.Analyzer
	speechPeriod time.Duration

	resampler    *resample.Resampler
	inRate       int
	analyzerRate int

	speaking      bool
	speakingAccum time.Duration
}

// New builds a VAD Processor. The VAD analyzer is required.
func New(cfg Config) *Processor {
	period := cfg.SpeechActivityPeriod
	if period == 0 {
		period = defaultSpeechActivityPeriod
	}
	p := &Processor{vad: cfg.VAD, speechPeriod: period}
	p.Base = processor.New("VAD", p)
	return p
}

// ProcessFrame drives the analyzer from incoming audio and forwards frames.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Prefer the input rate so no resampling is needed — Silero runs natively
		// at 8 kHz as well as 16 kHz. Fall back to the default rate (and resample)
		// only if the analyzer rejects the input rate.
		rate := fr.AudioInSampleRate
		if rate <= 0 {
			rate = analyzerSampleRate
		}
		if err := p.vad.SetSampleRate(rate); err != nil {
			rate = analyzerSampleRate
			if err := p.vad.SetSampleRate(rate); err != nil {
				return err
			}
		}
		p.analyzerRate = rate
		return p.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		if dir == processor.Downstream {
			return p.handleAudio(ctx, fr)
		}
		return p.PushFrame(ctx, f, dir)
	default:
		return p.PushFrame(ctx, f, dir)
	}
}

// Cleanup closes the analyzer and resampler.
func (p *Processor) Cleanup(ctx context.Context) error {
	if p.vad != nil {
		_ = p.vad.Close()
	}
	err := p.Base.Cleanup(ctx)
	if p.resampler != nil {
		p.resampler.Close()
		p.resampler = nil
	}
	return err
}

// handleAudio forwards the audio, runs the VAD, and emits VAD frames.
func (p *Processor) handleAudio(ctx context.Context, f *frames.InputAudioRawFrame) error {
	if err := p.PushFrame(ctx, f, processor.Downstream); err != nil {
		return err
	}
	state := p.vad.AnalyzeAudio(p.toAnalyzerRate(f))

	switch {
	case state == vad.StateSpeaking && !p.speaking:
		p.speaking = true
		p.speakingAccum = 0
		_ = p.PushFrame(ctx, frames.NewVADUserStartedSpeakingFrame(p.vad.Params().StartSecs), processor.Downstream)
	case state == vad.StateQuiet && p.speaking:
		p.speaking = false
		ts := time.Now().UTC().Format(time.RFC3339)
		_ = p.PushFrame(ctx, frames.NewVADUserStoppedSpeakingFrame(p.vad.Params().StopSecs, ts), processor.Downstream)
	}

	if p.speaking && p.speechPeriod > 0 {
		p.speakingAccum += frameDuration(f)
		if p.speakingAccum >= p.speechPeriod {
			p.speakingAccum = 0
			_ = p.PushFrame(ctx, frames.NewUserSpeakingFrame(), processor.Downstream)
		}
	}
	return nil
}

// frameDuration is the wall-clock duration of one audio frame.
func frameDuration(f *frames.InputAudioRawFrame) time.Duration {
	if f.SampleRate == 0 {
		return 0
	}
	return time.Duration(f.NumFrames()) * time.Second / time.Duration(f.SampleRate)
}

// toAnalyzerRate returns the frame's audio resampled to the analyzer rate, mono.
func (p *Processor) toAnalyzerRate(f *frames.InputAudioRawFrame) []byte {
	if f.SampleRate == p.analyzerRate {
		return f.Audio
	}
	if p.resampler == nil || p.inRate != f.SampleRate {
		if p.resampler != nil {
			p.resampler.Close()
			p.resampler = nil
		}
		r, err := resample.New(f.SampleRate, p.analyzerRate, 1)
		if err != nil {
			slog.Error("vadproc: create resampler", "from", f.SampleRate, "to", p.analyzerRate, "err", err)
			return f.Audio
		}
		p.resampler = r
		p.inRate = f.SampleRate
	}
	return p.resampler.Process(f.Audio)
}
