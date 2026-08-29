// Package vadproc is the voice-activity-detection pipeline processor. It hosts a
// controller.Controller and turns what it hears into the raw VAD frames the turn
// subsystem consumes: VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame
// on speech onset and offset, and UserSpeakingFrame for every chunk heard as
// speech. It does not decide turns, which is the turns package's job.
//
// Place it just after the input transport. The detection itself, along with the
// resampling it needs and the watch for audio that stops arriving, lives in
// audio/vad/controller, so anything else needing the same detection can drive it
// without going through a pipeline. Frames are forwarded before detection runs
// over them, so audio keeps flowing and the rest of the pipeline is started
// before the parameters are reported.
package vadproc

import (
	"context"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vad/controller"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Config configures a Processor.
type Config struct {
	// VAD detects voice activity. Required.
	VAD vad.Analyzer
	// AudioIdleTimeout is how long to wait, with the user speaking and no audio
	// arriving at all, before taking the speech to have stopped. It covers the
	// audio going away mid-utterance, a muted microphone being the usual case:
	// the detector never sees the silence that would have ended the speech, so
	// without this the user is left speaking for good.
	//
	// Leave it nil for one second. A zero duration turns the watch off.
	AudioIdleTimeout *time.Duration
}

// Processor is the VAD pipeline processor.
type Processor struct {
	*processor.Base

	controller *controller.Controller
}

// New builds a VAD Processor. The VAD analyzer is required.
func New(cfg Config) *Processor {
	p := &Processor{}
	p.Base = processor.New("VAD", p)

	// The frames go both ways: an interruption decision is made upstream of the
	// detector and a transcription one downstream, and both are driven by them.
	p.controller = controller.New(cfg.VAD, controller.Handlers{
		OnSpeechStarted: func(ctx context.Context) {
			startSecs := p.controller.Params().StartSecs
			// Taken once, outside the builder, so the frame sent each way
			// reports the same moment.
			ts := time.Now()
			_ = p.Broadcast(ctx, func() frames.Frame {
				return frames.NewVADUserStartedSpeakingFrame(startSecs, ts)
			})
		},
		OnSpeechStopped: func(ctx context.Context) {
			stopSecs := p.controller.Params().StopSecs
			// Taken once, outside the builder, so the frame sent each way
			// reports the same moment.
			ts := time.Now()
			_ = p.Broadcast(ctx, func() frames.Frame {
				return frames.NewVADUserStoppedSpeakingFrame(stopSecs, ts)
			})
		},
		OnSpeechActivity: func(ctx context.Context) {
			_ = p.Broadcast(ctx, func() frames.Frame { return frames.NewUserSpeakingFrame() })
		},
		OnPushFrame: func(ctx context.Context, f frames.Frame, dir processor.Direction) {
			_ = p.PushFrame(ctx, f, dir)
		},
		OnBroadcastFrame: func(ctx context.Context, build func() frames.Frame) {
			_ = p.Broadcast(ctx, build)
		},
	}, controller.Config{AudioIdleTimeout: cfg.AudioIdleTimeout})

	return p
}

// Setup hands the controller its configuration, so the detector is told the
// pipeline's input rate before any audio arrives.
func (p *Processor) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	return p.controller.Setup(s)
}

// ProcessFrame forwards the frame and then lets the controller run over it.
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
	switch f.(type) {
	case *frames.StartFrame:
		p.controller.Start(ctx)
	case *frames.EndFrame, *frames.CancelFrame:
		// The session is over, so the idle watch stops here rather than waiting
		// for teardown: left running it would report the silence that only means
		// the session ended. What the controller holds is released in Cleanup.
		p.controller.Stop()
	}
	return p.controller.ProcessFrame(ctx, f)
}

// Cleanup releases the controller and the processor.
func (p *Processor) Cleanup(ctx context.Context) error {
	p.controller.Cleanup()
	return p.Base.Cleanup(ctx)
}
