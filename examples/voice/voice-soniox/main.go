// Command voice-soniox is a voice agent: Soniox speech-to-text and text-to-
// speech, with an OpenAI LLM.
//
//	SONIOX_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/voice/voice-soniox
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/soniox"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
	" to what the user said in a creative, helpful, and brief way."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := soniox.NewSTT(soniox.Config{
		APIKey: os.Getenv("SONIOX_API_KEY"),
		// Language is the language hint; LanguageHintsStrict enforces it.
		Language:                     language.English,
		LanguageHintsStrict:          new(true),
		EnableLanguageIdentification: new(true),
	})

	tts := soniox.NewTTS(soniox.TTSConfig{
		APIKey: os.Getenv("SONIOX_API_KEY"),
		Voice:  "Nora",
	})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		llm,             // LLM
		tts,             // TTS
		t.Output(),      // Transport bot output
		agg.Assistant(), // Assistant spoken responses
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
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
