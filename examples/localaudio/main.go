// Command localaudio echoes the local microphone straight back to the local
// speaker using jargo's pure-Go local-audio transport. It needs no API keys and
// no browser — just a running PulseAudio or PipeWire server — so it is a quick
// way to check that capture and playback work. Wear headphones: echoing the mic
// to the speaker will otherwise feed back.
//
// Run with: go run ./examples/localaudio   (Ctrl-C to stop)
package main

import (
	"context"
	"errors"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/localaudio"
)

const sampleRate = 24000

func main() {
	params := transport.DefaultParams()
	params.AudioInSampleRate = sampleRate
	params.AudioOutSampleRate = sampleRate

	t, err := localaudio.New(localaudio.Config{AppName: "jargo-echo", Params: params})
	if err != nil {
		log.Fatal(err)
	}
	defer t.Close()

	pipe := pipeline.New(t.Input(), newEcho(), t.Output())
	task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
		Params: pipeline.Params{
			AudioInSampleRate:  sampleRate,
			AudioOutSampleRate: sampleRate,
		},
	})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	slog.Info("local echo started; speak into the mic (Ctrl-C to stop)")
	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("pipeline ended", "err", err)
	}
	slog.Info("local echo stopped")
}

// echoProcessor turns each captured audio frame into an outgoing audio frame,
// so the pipeline plays the microphone straight back.
type echoProcessor struct {
	*processor.Base
}

func newEcho() *echoProcessor {
	e := &echoProcessor{}
	e.Base = processor.New("Echo", e)
	return e
}

func (e *echoProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if in, ok := f.(*frames.InputAudioRawFrame); ok {
		out := frames.NewOutputAudioRawFrame(in.Audio, in.SampleRate, in.NumChannels)
		return e.PushFrame(ctx, out, processor.Downstream)
	}
	return e.PushFrame(ctx, f, dir)
}
