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
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// analyzerSampleRate is the fallback analyzer rate, used when the input rate is
// not one the analyzer accepts (Silero also runs natively at 8 kHz).
const analyzerSampleRate = 16000

// defaultAudioIdleTimeout is how long the audio can stop arriving mid-speech
// before the user is taken to have stopped.
const defaultAudioIdleTimeout = time.Second

// Config configures a Processor.
type Config struct {
	// VAD detects voice activity. Required.
	VAD vad.Analyzer
	// AudioIdleTimeout is how long to wait, with the user speaking and no audio
	// arriving at all, before taking the speech to have stopped. It covers the
	// audio going away mid-utterance, a muted microphone being the usual case:
	// the detector never sees the silence that would have ended the speech, so
	// without this the user is left speaking for good. 0 uses one second, a
	// negative value disables it.
	AudioIdleTimeout time.Duration
}

// Processor is the VAD pipeline processor.
type Processor struct {
	*processor.Base

	vad         vad.Analyzer
	idleTimeout time.Duration

	resampler    *resample.Resampler
	inRate       int
	analyzerRate int

	// mu guards the speaking state, which the idle watcher reads and writes
	// alongside the goroutine processing frames.
	mu          sync.Mutex
	speaking    bool
	lastAudioAt time.Time

	idleCancel context.CancelFunc
	idleWG     sync.WaitGroup
}

// New builds a VAD Processor. The VAD analyzer is required.
func New(cfg Config) *Processor {
	idle := cfg.AudioIdleTimeout
	if idle == 0 {
		idle = defaultAudioIdleTimeout
	}
	p := &Processor{vad: cfg.VAD, idleTimeout: idle}
	p.Base = processor.New("VAD", p)
	return p
}

// ProcessFrame forwards the frame and then drives the analyzer from it.
//
// It forwards first so the StartFrame reaches everything downstream before the
// parameters are reported, and so audio keeps flowing without waiting on
// detection.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}

	switch fr := f.(type) {
	case *frames.StartFrame:
		return p.start(ctx, fr)
	case *frames.InputAudioRawFrame:
		p.handleAudio(ctx, fr)
	case *frames.VADParamsUpdateFrame:
		// Travels upstream from whatever asked for the change, so it is acted on
		// whichever way it was going.
		p.vad.SetParams(fr.Params)
		p.reportParams(ctx)
	}
	return nil
}

// start configures the analyzer for the pipeline's input rate and reports the
// parameters it will run with.
func (p *Processor) start(ctx context.Context, f *frames.StartFrame) error {
	// Prefer the input rate so no resampling is needed — Silero runs natively
	// at 8 kHz as well as 16 kHz. Fall back to the default rate (and resample)
	// only if the analyzer rejects the input rate.
	rate := f.AudioInSampleRate
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
	p.startIdleWatch(ctx)
	p.reportParams(ctx)
	return nil
}

// reportParams broadcasts the detection parameters in force, so a processor
// downstream can size its own behavior to them.
func (p *Processor) reportParams(ctx context.Context) {
	params := p.vad.Params()
	_ = p.Broadcast(ctx, func() frames.Frame {
		vp := params
		return frames.NewSpeechControlParamsFrame(&vp, nil)
	})
}

// Cleanup closes the analyzer and resampler.
func (p *Processor) Cleanup(ctx context.Context) error {
	p.stopIdleWatch()
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

// handleAudio runs detection over one chunk and reports what it heard. The
// frames go both ways: an interruption decision upstream and a transcription one
// downstream are both driven by them.
func (p *Processor) handleAudio(ctx context.Context, f *frames.InputAudioRawFrame) {
	state := p.vad.AnalyzeAudio(p.toAnalyzerRate(f))

	p.mu.Lock()
	p.lastAudioAt = time.Now()
	started := state == vad.StateSpeaking && !p.speaking
	stopped := state == vad.StateQuiet && p.speaking
	switch {
	case started:
		p.speaking = true
	case stopped:
		p.speaking = false
	}
	speaking := p.speaking
	p.mu.Unlock()

	switch {
	case started:
		startSecs := p.vad.Params().StartSecs
		_ = p.Broadcast(ctx, func() frames.Frame {
			return frames.NewVADUserStartedSpeakingFrame(startSecs)
		})
	case stopped:
		p.broadcastStopped(ctx)
	}

	// Reported for every chunk heard as speech, the one that started it
	// included, so anything counting on the user still being there keeps hearing
	// about it.
	if speaking {
		_ = p.Broadcast(ctx, func() frames.Frame { return frames.NewUserSpeakingFrame() })
	}
}

// broadcastStopped reports that the user's speech has ended.
func (p *Processor) broadcastStopped(ctx context.Context) {
	stopSecs := p.vad.Params().StopSecs
	_ = p.Broadcast(ctx, func() frames.Frame {
		ts := time.Now().UTC().Format(time.RFC3339)
		return frames.NewVADUserStoppedSpeakingFrame(stopSecs, ts)
	})
}

// startIdleWatch brings up the watcher that ends speech when the audio stops
// arriving altogether.
func (p *Processor) startIdleWatch(ctx context.Context) {
	if p.idleTimeout <= 0 {
		return
	}
	p.stopIdleWatch()

	p.mu.Lock()
	p.lastAudioAt = time.Now()
	p.mu.Unlock()

	// Detached from the frame's context, which does not outlive the frame.
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	p.idleCancel = cancel
	p.idleWG.Add(1)
	go p.idleWatch(runCtx)
}

// stopIdleWatch tears the watcher down.
func (p *Processor) stopIdleWatch() {
	cancel := p.idleCancel
	p.idleCancel = nil
	if cancel == nil {
		return
	}
	cancel()
	p.idleWG.Wait()
}

// idleWatch ends the user's speech when no audio has arrived for the idle
// timeout. The detector only ever hears silence as speech ending, so audio that
// stops mid-utterance (a microphone muted part-way through, typically) would
// otherwise leave the user speaking for good, and the turn would never close.
func (p *Processor) idleWatch(ctx context.Context) {
	defer p.idleWG.Done()
	for {
		p.mu.Lock()
		deadline := p.lastAudioAt.Add(p.idleTimeout)
		p.mu.Unlock()

		if remaining := time.Until(deadline); remaining > 0 {
			// Audio is still recent, so wait out only what is left of the window.
			if !sleepCtx(ctx, remaining) {
				return
			}
			continue
		}

		p.mu.Lock()
		idled := p.speaking
		if idled {
			p.speaking = false
		}
		p.mu.Unlock()

		if idled {
			slog.Warn("vadproc: no audio while the user was speaking, ending the speech",
				"timeout", p.idleTimeout)
			p.broadcastStopped(ctx)
		}

		if !sleepCtx(ctx, p.idleTimeout) {
			return
		}
	}
}

// sleepCtx waits for d, reporting false if ctx was done first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
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
