// Command 01-say-one-thing says one thing to whoever connects, then hangs up.
//
// It is the smallest bot there is: a TTS service and the transport's output.
// When a client connects, the bot queues the sentence to speak and an EndFrame,
// so the pipeline ends once the sentence has been spoken.
//
//	CARTESIA_API_KEY=… go run ./examples/getting-started/01-say-one-thing
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// outputOnly is a transport's parameters with only audio output enabled.
func outputOnly(p transport.Params) transport.Params {
	p.AudioInEnabled = false
	p.AudioOutEnabled = true
	return p
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: func() transport.Params { return outputOnly(transport.DefaultParams()) },
	Twilio: func() runner.WebSocketParams {
		p := wsserver.DefaultParams()
		p.Params = outputOnly(p.Params)
		return runner.WebSocketParams{Params: p}
	},
	WebRTC: func() transport.Params { return outputOnly(transport.DefaultParams()) },
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	worker := pipeline.NewWorker(pipeline.New(tts, t.Output()), pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	// Register an event handler so we can play the audio when the client joins.
	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		worker.QueueFrames([]frames.Frame{frames.NewTTSSpeakFrame("Hello there!"), frames.NewEndFrame()})
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
