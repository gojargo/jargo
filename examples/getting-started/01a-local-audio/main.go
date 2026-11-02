// Command 01a-local-audio says one thing through the local speaker, then exits.
//
// It is 01-say-one-thing on the local audio transport: no server, no browser.
// It needs a running PulseAudio or PipeWire server.
//
//	CARTESIA_API_KEY=… go run ./examples/getting-started/01a-local-audio
package main

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/localaudio"
	"github.com/gojargo/jargo/workers"
	"github.com/joho/godotenv"
)

func run(ctx context.Context) error {
	params := transport.DefaultParams()
	params.AudioInEnabled = false
	t, err := localaudio.New(localaudio.Config{AppName: "01a-local-audio", Params: params})
	if err != nil {
		return err
	}
	defer t.Close()

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	worker := pipeline.NewWorker(pipeline.New(tts, t.Output()), pipeline.WorkerConfig{
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := true
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	go func() {
		time.Sleep(time.Second)
		worker.QueueFrames([]frames.Frame{
			frames.NewTTSSpeakFrame("Hello there, how is it going!"), frames.NewEndFrame(),
		})
	}()

	return r.Run(ctx, workers.RunOptions{})
}

func main() {
	_ = godotenv.Overload()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	if err := run(context.Background()); err != nil {
		slog.Error("01a-local-audio", "err", err)
		os.Exit(1)
	}
}
