// Command 02-llm-say-one-thing has an LLM say one thing to whoever connects,
// then hangs up.
//
// When a client connects, the bot hands the LLM a conversation with a single
// instruction and an EndFrame. The LLM's reply streams to the TTS service and
// out, and the pipeline ends once it has been spoken.
//
//	OPENAI_API_KEY=… CARTESIA_API_KEY=… go run ./examples/getting-started/02-llm-say-one-thing
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
	"github.com/gojargo/jargo/provider/openai/chat"
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

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction("You are a helpful assistant in a voice conversation. Your responses will be " +
		"spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond " +
		"to what the user said in a creative, helpful, and brief way.")

	worker := pipeline.NewWorker(pipeline.New(llm, tts, t.Output()), pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	// Register an event handler so we can play the audio when the client joins.
	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		convo := frames.NewLLMContext("")
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello to the world."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMContextFrame(convo), frames.NewEndFrame()})
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
