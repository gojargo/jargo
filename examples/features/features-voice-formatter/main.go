// Command features-voice-formatter shows the VoiceFormatter bundle, which
// applies a chain of built-in text transforms before TTS synthesis so that
// currency amounts, phone numbers, dates, acronyms and other special text are
// spoken naturally.
//
// Without voice formatting a TTS service might read:
//
//	"$42.50" as "dollar sign four two point five zero"
//	"API"    as a single word rather than "A P I"
//	"3/15"   as "three slash fifteen"
//
// With the VoiceFormatter these are rewritten before the audio is synthesized:
//
//	"$42.50"  →  "forty-two dollars and fifty cents"
//	"API"     →  "A P I"
//	"3/15/25" →  "March 15th, two thousand and twenty-five"
//
//	CARTESIA_API_KEY=… ELEVENLABS_API_KEY=… ELEVENLABS_VOICE_ID=… \
//	OPENAI_API_KEY=… go run ./examples/features/features-voice-formatter
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	ttsservice "github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/text"
	"github.com/gojargo/jargo/workers"
)

const systemInstruction = "You are a billing support assistant for a telecom company. " +
	"Your responses are spoken aloud. Use natural formatting in your " +
	"answers: currency amounts like $42.50, percentages like 3.5%, " +
	"email addresses like support@example.com, and abbreviations like " +
	"Dr. or St. as you normally would in writing; the voice formatter " +
	"will convert them to natural speech before synthesis."

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := cartesia.NewSTT(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})

	// The VoiceFormatter with every default transform on. Toggle fields of
	// the options to turn individual transforms on or off, e.g.
	// ExpandNumbers on and NormalizeAcronyms off.
	voiceFormatter, err := text.NewVoiceFormatter(text.DefaultFormatterOptions())
	if err != nil {
		return err
	}

	tts := elevenlabs.NewTTS(elevenlabs.Config{
		APIKey:  os.Getenv("ELEVENLABS_API_KEY"),
		VoiceID: os.Getenv("ELEVENLABS_VOICE_ID"),
	})
	// The VoiceFormatter is a text transform. AnyTextType means it runs on
	// every text frame, whatever its type.
	tts.SetTextTransformers(ttsservice.TextTransformer{TextType: frames.AnyTextType, Transform: voiceFormatter.Transform})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		tts,
		t.Output(),
		agg.Assistant(),
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
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Greet the user and let them know they've reached billing support. " +
				"Offer to help with their account balance, recent charges, or " +
				"payment options. Give a sample balance such as $127.50 due on " +
				"3/15/2025 and a support email like billing@telecom.example.com.",
		})
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
