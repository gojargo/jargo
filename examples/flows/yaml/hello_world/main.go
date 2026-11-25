// Command hello_world is a "Hello world" introduction to conversation flows,
// with the flow in YAML.
//
// The same bot as examples/flows/hello_world, split along the seam flows offer
// for runtime configuration: flow.yaml holds the two nodes and the transition
// between them, and handlers.go holds the one tool the flow calls.
//
//	CARTESIA_API_KEY=… GOOGLE_API_KEY=… \
//	  go run ./examples/flows/yaml/hello_world
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
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

// flowConfigPath is the flow.yaml beside this source file.
func flowConfigPath() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "flow.yaml"
	}
	return filepath.Join(filepath.Dir(file), "flow.yaml")
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	stt := cartesia.NewSTT(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "32b3f3c5-7171-46aa-abe7-b598964aa793",
	})
	model := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		aggregators.WithTurns(turns.Config{
			Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{}),
		}),
	)

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		model,           // LLM
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

	// The flow is data: load it, then join it to the tool it names.
	config, err := flows.FlowConfigFromFile(flowConfigPath())
	if err != nil {
		return err
	}
	flow, err := flows.NewFlow(config, handlers())
	if err != nil {
		return err
	}

	flowManager, err := flows.New(flows.Config{
		Enqueuer:        worker,
		Watcher:         worker,
		Aggregators:     agg,
		LLM:             model,
		GlobalFunctions: flow.GlobalFunctions(),
	})
	if err != nil {
		return err
	}

	event.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		if err := flowManager.Initialize(ctx, flow.InitialNode()); err != nil {
			slog.Error("initialize flow", "err", err)
		}
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
