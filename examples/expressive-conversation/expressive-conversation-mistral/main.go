// Command expressive-conversation-mistral is an expressive conversation:
// Deepgram speech-to-text, an OpenAI LLM and Mistral text-to-speech. The LLM
// writes expression tags into its replies, and Voxtral 3 performs them.
//
//	DEEPGRAM_API_KEY=… MISTRAL_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/expressive-conversation/expressive-conversation-mistral
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/mistral"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

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

const systemInstruction = "You are a warm, expressive assistant in a voice conversation. Your responses " +
	"will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. " +
	"Keep your responses brief."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	// Voxtral 3 performs emotion and pace tags, and sounds written in the text.
	tts := mistral.NewTTS(mistral.TTSConfig{
		APIKey: os.Getenv("MISTRAL_API_KEY"),
		Model:  "voxtral-mini-tts-3-0",
		Voice:  "voice_478",
	})

	// With expressive conversation on, the LLM is taught the expression tags the TTS
	// performs. Voxtral 3's emotions are a fixed list, so the policy offers the ones
	// this conversation calls for.
	expressionPolicy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{
			expressive.GuidanceMatchStyle,
			expressive.GuidanceSlowForDetails,
			expressive.GuidanceBreathe,
		},
		Delivery: []expressive.ExpressionDelivery{
			expressive.DeliverySlow,
			expressive.DeliveryHappy,
			expressive.DeliveryExcited,
			expressive.DeliveryCurious,
			expressive.DeliveryWarm,
			expressive.DeliverySympathetic,
			expressive.DeliveryCalm,
			expressive.DeliverySad,
			expressive.DeliveryAmused,
			expressive.DeliveryPlayful,
			expressive.DeliverySarcastic,
		},
		Sounds: []expressive.ExpressionSound{
			expressive.SoundLaugh,
			expressive.SoundChuckle,
			expressive.SoundSigh,
			expressive.SoundLightBreath,
			expressive.SoundClearThroat,
		},
	}

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
			// The LLM writes expression tags into its responses, and the TTS
			// performs them.
			EnableExpressiveConversation: true,
			ExpressionPolicy:             expressionPolicy,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

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
