// Command transcription-together transcribes what the user says and prints it: Together AI.
//
// A VAD processor ahead of the STT service marks where the user speaks, and a
// transcriptionLogger after it prints each transcript, interim and final. The
// bot never speaks.
//
//	TOGETHER_API_KEY=… \
//	  go run ./examples/transcription/transcription-together
//
// Then open http://localhost:7860, connect, and speak.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/vadproc"
	"github.com/gojargo/jargo/provider/together"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// transcriptionLogger prints every transcript passing through it.
type transcriptionLogger struct {
	*processor.Base
}

func newTranscriptionLogger() *transcriptionLogger {
	p := &transcriptionLogger{}
	p.Base = processor.New("TranscriptionLogger", p)
	return p
}

// ProcessFrame prints the transcripts and pushes every frame on.
func (p *transcriptionLogger) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch t := f.(type) {
	case *frames.TranscriptionFrame:
		_, _ = fmt.Fprintln(os.Stdout, "Transcription: "+t.Text)
	case *frames.InterimTranscriptionFrame:
		_, _ = fmt.Fprintln(os.Stdout, "Interim transcription: "+t.Text)
	}

	// Push all frames through.
	return p.PushFrame(ctx, f, dir)
}

// audioIn takes the user's audio and sends none back.
func audioIn(p transport.Params) transport.Params {
	p.AudioOutEnabled = false
	return p
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: func() transport.Params { return audioIn(transport.DefaultParams()) },
	Twilio: func() runner.WebSocketParams {
		params := wsserver.DefaultParams()
		params.Params = audioIn(params.Params)
		return runner.WebSocketParams{Params: params}
	},
	WebRTC: func() transport.Params { return audioIn(transport.DefaultParams()) },
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := together.NewSTT(together.STTConfig{APIKey: os.Getenv("TOGETHER_API_KEY")})

	tl := newTranscriptionLogger()
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	vadProcessor := vadproc.New(vadproc.Config{VAD: silero})

	p := pipeline.New(t.Input(), vadProcessor, stt, tl, t.Output())

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

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
