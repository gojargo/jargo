// Command expressive-conversation-cartesia is an expressive conversation:
// Cartesia speech-to-text and text-to-speech, with an OpenAI LLM. The LLM
// writes expression tags into its replies, and Cartesia performs them.
//
//	CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/expressive-conversation/expressive-conversation-cartesia
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
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

	stt := cartesia.NewSTT(cartesia.STTConfig{
		APIKey: os.Getenv("CARTESIA_API_KEY"),
		Model:  "ink-2",
	})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey: os.Getenv("CARTESIA_API_KEY"),
		// An "Emotive" voice, which Cartesia recommends for emotion.
		VoiceID: "57dcab65-68ac-45a6-8480-6c4c52ec1cd1",
	})

	// With expressive conversation on, the LLM is taught the expression tags the TTS
	// performs. Cartesia has many more emotions (see CartesiaEmotion), but offered
	// more than about 15, the LLM starts inventing names of its own as a
	// conversation goes on, so the policy offers the ones this conversation calls
	// for, and says which to use for feelings that have none.
	expressionPolicy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{
			expressive.GuidanceMatchStyle,
			expressive.GuidanceSlowForDetails,
			expressive.GuidanceLine("Some feelings have no delivery of their own: for relief, use happy or " +
				"content; to reassure, comfort or encourage, use calm or sympathetic; for concern or " +
				"support, use sympathetic."),
		},
		Delivery: []expressive.ExpressionDelivery{
			expressive.DeliverySlow,
			expressive.DeliveryCalm,
			expressive.DeliveryContent,
			expressive.DeliverySympathetic,
			expressive.DeliveryApologetic,
			expressive.DeliveryGrateful,
			expressive.DeliveryHappy,
			expressive.DeliveryExcited,
			expressive.DeliveryCurious,
			expressive.DeliveryConfident,
			expressive.DeliverySad,
			expressive.DeliverySurprised,
			expressive.DeliveryAmazed,
			expressive.DeliveryHesitant,
			expressive.DeliveryAnxious,
			expressive.DeliveryProud,
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
