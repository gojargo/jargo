// Command features-concurrent-llm-evaluation runs two LLMs on the same user
// turns, side by side in a ParallelPipeline.
//
// The OpenAI branch speaks to the user; the Groq branch answers the same turns
// with a different prompt and stays silent, so the two can be compared. One
// UserTurnProcessor decides the turns for both branches, fed by an external
// VAD processor, and both aggregators adopt its decisions.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… GROQ_API_KEY=… \
//	  go run ./examples/features/features-concurrent-llm-evaluation
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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/processor/vadproc"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/groq"
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

const (
	openaiInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
		"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
		" to what the user said in a creative, helpful, and brief way."
	groqInstruction = "You are a very helpful assistant. Your goal is to demonstrate your capabilities " +
		"in detail in a creative and helpful way."
)

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	openaiLLM := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	openaiLLM.SetSystemInstruction(openaiInstruction)

	groqLLM := groq.NewLLM(groq.LLMConfig{APIKey: os.Getenv("GROQ_API_KEY")})
	groqLLM.SetSystemInstruction(groqInstruction)

	openaiContext := frames.NewLLMContext("")
	groqContext := frames.NewLLMContext("")

	// An external VAD processor, because the UserTurnProcessor is shared across
	// several aggregators. It pushes the VAD speaking frames the
	// UserTurnProcessor manages the turns from.
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	vadProcessor := vadproc.New(vadproc.Config{VAD: silero})

	// The external user turn processor pushes the user started and stopped
	// speaking frames, and the interruptions. It is for advanced cases with
	// several aggregators in the pipeline.
	userTurnProcessor := turns.NewUserTurnProcessor(turns.Config{})

	// Both aggregators take external strategies, since the shared
	// UserTurnProcessor manages the turns.
	openaiAgg := aggregators.New(openaiContext, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))
	groqAgg := aggregators.New(groqContext, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))

	parallel, err := pipeline.NewParallel(
		[]processor.Processor{
			openaiAgg.User(),      // User responses
			openaiLLM,             // LLM
			tts,                   // TTS
			t.Output(),            // Transport bot output
			openaiAgg.Assistant(), // Assistant spoken responses
		},
		[]processor.Processor{
			groqAgg.User(),      // User responses
			groqLLM,             // LLM
			groqAgg.Assistant(), // Assistant responses
		},
	)
	if err != nil {
		return err
	}

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,       // STT
		vadProcessor,
		userTurnProcessor,
		parallel,
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
		openaiContext.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		groqContext.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
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
