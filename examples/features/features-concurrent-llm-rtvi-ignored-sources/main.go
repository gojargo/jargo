// Command features-concurrent-llm-rtvi-ignored-sources keeps the RTVI messages
// of one pipeline processor from reaching the client, so that a secondary
// branch does not leak events to it.
//
// A ParallelPipeline runs two LLMs on the same user turns. The main branch
// speaks to the user; the evaluator branch silently grades each user message,
// and rtvi.ObserverParams.IgnoredSources keeps its events away from the
// client. One UserTurnProcessor decides the turns for both branches, fed by
// an external VAD processor, and both aggregators adopt its decisions.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-concurrent-llm-rtvi-ignored-sources
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
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/processor/vadproc"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
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
	mainInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
		"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
		" to what the user said in a creative, helpful, and brief way."
	evaluatorInstruction = "You are a silent quality evaluator. When given a user message, respond with " +
		"a single JSON object: {'score': <1-5>, 'reason': '<brief reason>'}. Do not respond conversationally."
)

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	// The main LLM drives the conversation. Its RTVI events reach the client.
	mainLLM := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	mainLLM.SetSystemInstruction(mainInstruction)

	// The evaluator LLM silently grades the user's message in the background.
	// Its RTVI events are suppressed, so the client is unaware of this branch.
	// It is OpenAI's own endpoint under a name of its own, so the logs tell the
	// two LLMs apart.
	evaluatorLLM := chat.NewCompatLLM(
		chat.Compat{Name: "EvaluatorLLM", BaseURL: "https://api.openai.com/v1", DefaultModel: "gpt-4o-mini"},
		chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")},
	)
	evaluatorLLM.SetSystemInstruction(evaluatorInstruction)

	mainContext := frames.NewLLMContext("")
	evaluatorContext := frames.NewLLMContext("")

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
	mainAgg := aggregators.New(mainContext, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))
	evaluatorAgg := aggregators.New(evaluatorContext, aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	}))

	parallel, err := pipeline.NewParallel(
		// Main branch: speaks to the user.
		[]processor.Processor{
			mainAgg.User(),
			mainLLM,
			tts,
			t.Output(),
			mainAgg.Assistant(),
		},
		// Evaluator branch: silent background scoring, no audio output.
		[]processor.Processor{
			evaluatorAgg.User(),
			evaluatorLLM,
			evaluatorAgg.Assistant(),
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

	rtviParams := rtvi.DefaultObserverParams()
	rtviParams.IgnoredSources = []processor.Processor{evaluatorLLM}

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		RTVIObserverParams:      &rtviParams,
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		mainContext.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		evaluatorContext.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Ready to evaluate user messages."})
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
