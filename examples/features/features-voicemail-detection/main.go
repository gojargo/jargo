// Command features-voicemail-detection is a voice agent that tells a person
// from a voicemail greeting, and leaves a message when it reaches voicemail.
//
// A voicemail detector placed after the STT service asks a Jev classifier
// whether the other side is a person or a voicemail greeting, and its gate after
// the TTS service holds the bot's speech back until the answer is in. On
// voicemail the bot leaves a message and ends the call.
//
// Speech-to-text is Deepgram, the LLM OpenAI and text-to-speech Cartesia.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… TYPESAFE_API_KEY=… \
//	  go run ./examples/features/features-voicemail-detection
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/classifier/jev"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/voicemail"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
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

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	classifier, err := jev.New(jev.Config{APIKey: os.Getenv("TYPESAFE_API_KEY")})
	if err != nil {
		return err
	}
	detector, err := voicemail.New(voicemail.Config{Classifier: classifier})
	if err != nil {
		return err
	}

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		detector.Detector(), // Voicemail detection, between STT and the user aggregator
		agg.User(),
		llm,
		tts,
		detector.Gate(), // TTS gating, right after the TTS service
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
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	events.On(detector.Events(), voicemail.EventConversationDetected, func(context.Context, *voicemail.Detector) {
		slog.Info("Conversation detected!")
	})

	events.On(detector.Events(), voicemail.EventVoicemailDetected, func(ctx context.Context, p *voicemail.Detector) {
		slog.Info("Voicemail detected! Leaving a message...")

		message := frames.NewTTSSpeakFrame("Hello, this is Alex calling about your appointment. " +
			"Please call me back at 555-0123 when you get this.")
		if err := p.PushFrame(ctx, message, processor.Downstream); err != nil {
			slog.Error("leaving the voicemail message", "error", err)
			return
		}
		// The verdict is final and the message is left, so the call is over.
		if err := p.PushFrame(ctx, frames.NewEndWorkerFrame(), processor.Downstream); err != nil {
			slog.Error("ending the call", "error", err)
		}
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
