package runner_test

import (
	"context"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
)

// A bot is one function, called for every session. It builds the session's
// transport, its pipeline around it, and runs it. This one says one sentence
// to whoever connects, then hangs up. Main serves it, with a browser client at
// http://localhost:7860.
func Example() {
	bot := func(ctx context.Context, args runner.Arguments) error {
		t, err := runner.CreateTransport(ctx, args, runner.TransportParams{
			WebRTC: transport.DefaultParams,
		})
		if err != nil {
			return err
		}
		tts := cartesia.NewTTS(cartesia.Config{
			APIKey:  os.Getenv("CARTESIA_API_KEY"),
			VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
		})
		worker := pipeline.NewWorker(pipeline.New(tts, t.Output()), pipeline.WorkerConfig{
			IdleTimeout: args.Base().PipelineIdleTimeout,
		})
		events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
			worker.QueueFrames([]frames.Frame{frames.NewTTSSpeakFrame("Hello there!"), frames.NewEndFrame()})
		})
		return worker.Run(ctx)
	}

	runner.Main(bot)
}
