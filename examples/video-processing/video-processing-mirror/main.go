// Command video-processing-mirror sends the user's audio and video straight
// back to them.
//
// A processor between the transport's input and output turns each frame of
// received audio and video into a frame to send, so the user hears and sees
// themselves. No service is involved, so no API key is needed.
//
//	go run ./examples/video-processing/video-processing-mirror
//
// Then open http://localhost:7860 and connect with your camera on.
package main

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// mirrorProcessor turns received audio and video into audio and video to send,
// and passes every other frame on.
type mirrorProcessor struct {
	*processor.Base
}

func newMirrorProcessor() *mirrorProcessor {
	m := &mirrorProcessor{}
	m.Base = processor.New("MirrorProcessor", m)
	return m
}

// ProcessFrame pushes received audio and video back out, and every other frame
// on.
func (m *mirrorProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := m.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch fr := f.(type) {
	case frames.InputAudioFrame:
		audio := fr.AudioData()
		return m.PushFrame(ctx,
			frames.NewOutputAudioRawFrame(audio.Audio, audio.SampleRate, audio.NumChannels), processor.Downstream)
	case frames.InputImageFrame:
		image := fr.ImageData()
		return m.PushFrame(ctx,
			frames.NewOutputImageRawFrame(image.Image, image.Width, image.Height, image.Format), processor.Downstream)
	default:
		return m.PushFrame(ctx, f, dir)
	}
}

// mirrorParams are the parameters of a transport that mirrors the user: audio
// both ways, video in, and 1280 by 720 live video out.
func mirrorParams() transport.Params {
	p := transport.DefaultParams()
	p.VideoInEnabled = true
	p.VideoOutEnabled = true
	p.VideoOutIsLive = true
	p.VideoOutWidth = 1280
	p.VideoOutHeight = 720
	return p
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: mirrorParams,
	WebRTC:  mirrorParams,
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	p := pipeline.New(t.Input(), newMirrorProcessor(), t.Output())

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

func bot(ctx context.Context, args runner.Arguments) error {
	t, err := runner.CreateTransport(ctx, args, transportParams)
	if err != nil {
		return err
	}
	return runBot(ctx, t, args)
}

func main() {
	runner.Main(bot)
}
