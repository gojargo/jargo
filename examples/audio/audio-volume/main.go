// Command audio-volume changes the volume of the bot's speech with volume
// frames.
//
// A VolumeGainFrame scales the audio between it and the next gain frame, and a
// gain of 1.0 restores the volume. A VolumeFrame sets the whole output volume,
// which holds until the next volume frame. The bot speaks a few sentences at
// different volumes, then ends.
//
//	DEEPGRAM_API_KEY=… go run ./examples/audio/audio-volume
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// audioOut sends the bot's audio and takes none from the user.
func audioOut(p transport.Params) transport.Params {
	p.AudioInEnabled = false
	return p
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: func() transport.Params { return audioOut(transport.DefaultParams()) },
	Twilio: func() runner.WebSocketParams {
		params := wsserver.DefaultParams()
		params.Params = audioOut(params.Params)
		return runner.WebSocketParams{Params: params}
	},
	WebRTC: func() transport.Params { return audioOut(transport.DefaultParams()) },
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	tts := deepgram.NewWebsocketTTS(deepgram.WebsocketTTSConfig{
		APIKey: os.Getenv("DEEPGRAM_API_KEY"),
		Voice:  "aura-2-andromeda-en",
	})

	worker := pipeline.NewWorker(pipeline.New(tts, t.Output()), pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		worker.QueueFrames([]frames.Frame{
			frames.NewTTSSpeakFrame("This is my normal volume."),
			// A gain changes the volume for the audio between two gain frames,
			// and resetting it to 1.0 restores the volume.
			frames.NewVolumeGainFrame(0.1),
			frames.NewTTSSpeakFrame("This sentence plays at a tenth of the volume."),
			frames.NewVolumeGainFrame(1.0),
			frames.NewTTSSpeakFrame("And this one is back to normal."),
			frames.NewVolumeGainFrame(3.0),
			frames.NewTTSSpeakFrame("This one plays three times as loud."),
			frames.NewVolumeGainFrame(1.0),
			// The volume holds until the next volume frame.
			frames.NewVolumeFrame(0.2),
			frames.NewTTSSpeakFrame("Now my whole volume is turned down."),
			frames.NewTTSSpeakFrame("It stays down until the volume changes again."),
			frames.NewVolumeFrame(1.0),
			frames.NewTTSSpeakFrame("And I'm back to my normal volume. Goodbye!"),
			frames.NewEndFrame(),
		})
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
