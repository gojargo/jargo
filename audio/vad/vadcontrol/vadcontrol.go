// Package vadcontrol drives a voice-activity detector and reports what it hears.
// It owns the detection state, the resampling the detector needs, and the watch
// for audio that stops arriving, and it reports speech starting, continuing and
// stopping through handlers the caller supplies.
//
// It is separate from the pipeline processor that usually hosts it (see
// processor/vadproc) so that anything else needing the same detection can drive
// it too: a speech-to-speech service running its own detection alongside the
// provider's, say. It sits beside the detector rather than inside a processor
// for that reason.
//
// It lives in its own package rather than in audio/vad because it works in
// frames, and the frames package refers to vad.Params.
package vadcontrol

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

// DefaultAudioIdleTimeout is how long the audio can stop arriving mid-speech
// before the user is taken to have stopped.
const DefaultAudioIdleTimeout = time.Second

// Handlers receives what the controller decides. Any of them may be nil.
type Handlers struct {
	// OnSpeechStarted reports that the user began speaking.
	OnSpeechStarted func(ctx context.Context)
	// OnSpeechStopped reports that the user stopped speaking, including when the
	// audio stopped arriving rather than going quiet.
	OnSpeechStopped func(ctx context.Context)
	// OnSpeechActivity reports one more chunk heard as speech. It fires for every
	// such chunk, the one that started the speech included.
	OnSpeechActivity func(ctx context.Context)
	// OnPushFrame sends a frame through whatever hosts the controller.
	OnPushFrame func(ctx context.Context, f frames.Frame, dir processor.Direction)
	// OnBroadcastFrame sends a frame both ways through whatever hosts the
	// controller. build is called once per direction.
	OnBroadcastFrame func(ctx context.Context, build func() frames.Frame)
}

// Config configures a Controller.
type Config struct {
	// AudioIdleTimeout is how long to wait, with the user speaking and no audio
	// arriving at all, before taking the speech to have stopped. It covers the
	// audio going away mid-utterance, a muted microphone being the usual case:
	// the detector never sees the silence that would have ended the speech, so
	// without this the user is left speaking for good. 0 uses one second, a
	// negative value disables it.
	AudioIdleTimeout time.Duration
}

// Controller drives a detector over incoming audio and reports what it hears.
type Controller struct {
	analyzer    vad.Analyzer
	handlers    Handlers
	idleTimeout time.Duration

	resampler    *resample.Resampler
	inRate       int
	analyzerRate int

	// mu guards the speaking state, which the idle watch reads and writes
	// alongside whatever is feeding in audio.
	mu          sync.Mutex
	speaking    bool
	lastAudioAt time.Time

	idleCancel context.CancelFunc
	idleWG     sync.WaitGroup
}

// New builds a Controller around analyzer. The analyzer is required.
func New(analyzer vad.Analyzer, handlers Handlers, cfg Config) *Controller {
	idle := cfg.AudioIdleTimeout
	if idle == 0 {
		idle = DefaultAudioIdleTimeout
	}
	return &Controller{analyzer: analyzer, handlers: handlers, idleTimeout: idle}
}

// Params returns the detection parameters in force.
func (c *Controller) Params() vad.Params { return c.analyzer.Params() }

// ProcessFrame drives the detector from one frame. It acts on the start of the
// pipeline, on incoming audio, and on a request to change the parameters;
// anything else it ignores.
func (c *Controller) ProcessFrame(ctx context.Context, f frames.Frame) error {
	switch fr := f.(type) {
	case *frames.StartFrame:
		return c.start(ctx, fr)
	case *frames.InputAudioRawFrame:
		c.handleAudio(ctx, fr)
	case *frames.VADParamsUpdateFrame:
		c.analyzer.SetParams(fr.Params)
		c.ReportParams(ctx)
	}
	return nil
}

// Cleanup stops the idle watch and releases the detector and resampler.
func (c *Controller) Cleanup() {
	c.stopIdleWatch()
	if c.analyzer != nil {
		_ = c.analyzer.Close()
	}
	if c.resampler != nil {
		c.resampler.Close()
		c.resampler = nil
	}
}

// PushFrame sends a frame through whatever hosts the controller.
func (c *Controller) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) {
	if c.handlers.OnPushFrame != nil {
		c.handlers.OnPushFrame(ctx, f, dir)
	}
}

// BroadcastFrame sends a frame both ways through whatever hosts the controller.
func (c *Controller) BroadcastFrame(ctx context.Context, build func() frames.Frame) {
	if c.handlers.OnBroadcastFrame != nil {
		c.handlers.OnBroadcastFrame(ctx, build)
	}
}

// ReportParams broadcasts the detection parameters in force, so a processor
// downstream can size its own behavior to them.
func (c *Controller) ReportParams(ctx context.Context) {
	params := c.analyzer.Params()
	c.BroadcastFrame(ctx, func() frames.Frame {
		vp := params
		return frames.NewSpeechControlParamsFrame(&vp, nil)
	})
}

// start configures the detector for the pipeline's input rate and reports the
// parameters it will run with.
func (c *Controller) start(ctx context.Context, f *frames.StartFrame) error {
	// Prefer the input rate so no resampling is needed: Silero runs natively at
	// 8 kHz as well as 16 kHz. Fall back to the default rate, and resample, only
	// if the detector rejects the input rate.
	rate := f.AudioInSampleRate
	if rate <= 0 {
		rate = analyzerSampleRate
	}
	if err := c.analyzer.SetSampleRate(rate); err != nil {
		rate = analyzerSampleRate
		if err := c.analyzer.SetSampleRate(rate); err != nil {
			return err
		}
	}
	c.analyzerRate = rate
	c.startIdleWatch(ctx)
	c.ReportParams(ctx)
	return nil
}

// handleAudio runs detection over one chunk and reports what it heard.
func (c *Controller) handleAudio(ctx context.Context, f *frames.InputAudioRawFrame) {
	state := c.analyzer.AnalyzeAudio(c.toAnalyzerRate(f))

	c.mu.Lock()
	c.lastAudioAt = time.Now()
	started := state == vad.StateSpeaking && !c.speaking
	stopped := state == vad.StateQuiet && c.speaking
	switch {
	case started:
		c.speaking = true
	case stopped:
		c.speaking = false
	}
	speaking := c.speaking
	c.mu.Unlock()

	switch {
	case started:
		if c.handlers.OnSpeechStarted != nil {
			c.handlers.OnSpeechStarted(ctx)
		}
	case stopped:
		if c.handlers.OnSpeechStopped != nil {
			c.handlers.OnSpeechStopped(ctx)
		}
	}

	// Reported for every chunk heard as speech, the one that started it
	// included, so anything counting on the user still being there keeps hearing
	// about it.
	if speaking && c.handlers.OnSpeechActivity != nil {
		c.handlers.OnSpeechActivity(ctx)
	}
}

// startIdleWatch brings up the watch that ends speech when the audio stops
// arriving altogether.
func (c *Controller) startIdleWatch(ctx context.Context) {
	if c.idleTimeout <= 0 {
		return
	}
	c.stopIdleWatch()

	c.mu.Lock()
	c.lastAudioAt = time.Now()
	c.mu.Unlock()

	// Detached from the frame's context, which does not outlive the frame.
	runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	c.idleCancel = cancel
	c.idleWG.Add(1)
	go c.idleWatch(runCtx)
}

// stopIdleWatch tears the watch down.
func (c *Controller) stopIdleWatch() {
	cancel := c.idleCancel
	c.idleCancel = nil
	if cancel == nil {
		return
	}
	cancel()
	c.idleWG.Wait()
}

// idleWatch ends the user's speech when no audio has arrived for the idle
// timeout. The detector only ever hears silence as speech ending, so audio that
// stops mid-utterance (a microphone muted part-way through, typically) would
// otherwise leave the user speaking for good, and the turn would never close.
func (c *Controller) idleWatch(ctx context.Context) {
	defer c.idleWG.Done()
	for {
		c.mu.Lock()
		deadline := c.lastAudioAt.Add(c.idleTimeout)
		c.mu.Unlock()

		if remaining := time.Until(deadline); remaining > 0 {
			// Audio is still recent, so wait out only what is left of the window.
			if !sleepCtx(ctx, remaining) {
				return
			}
			continue
		}

		c.mu.Lock()
		idled := c.speaking
		if idled {
			c.speaking = false
		}
		c.mu.Unlock()

		if idled {
			slog.Warn("vadcontrol: no audio while the user was speaking, ending the speech",
				"timeout", c.idleTimeout)
			if c.handlers.OnSpeechStopped != nil {
				c.handlers.OnSpeechStopped(ctx)
			}
		}

		if !sleepCtx(ctx, c.idleTimeout) {
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
func (c *Controller) toAnalyzerRate(f *frames.InputAudioRawFrame) []byte {
	if f.SampleRate == c.analyzerRate {
		return f.Audio
	}
	if c.resampler == nil || c.inRate != f.SampleRate {
		if c.resampler != nil {
			c.resampler.Close()
			c.resampler = nil
		}
		r, err := resample.New(f.SampleRate, c.analyzerRate, 1)
		if err != nil {
			slog.Error("vadcontrol: create resampler",
				"from", f.SampleRate, "to", c.analyzerRate, "err", err)
			return f.Audio
		}
		c.resampler = r
		c.inRate = f.SampleRate
	}
	return c.resampler.Process(f.Audio)
}
