// Command podcast_interview is the podcast interview flow, configured from
// YAML at run time.
//
// An interview conversation split along the seam flows offer for runtime
// configuration:
//
//   - flow.yaml holds the graph: introduction, topic, interview, conclusion,
//     and final nodes, and where each tool leads. The interview node
//     transitions back to itself once per topic aspect.
//   - handlers.go holds the tools: NodeFunctions that carry their own
//     description and parameters.
//
// Run it with:
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/flows/yaml/podcast_interview
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
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/responses"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
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

// flowConfigPath is the flow.yaml beside this source file.
func flowConfigPath() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "flow.yaml"
	}
	return filepath.Join(filepath.Dir(file), "flow.yaml")
}

// runBot runs the podcast interview bot.
func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})
	model := responses.NewLLM(responses.Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  "gpt-4.1",
	})

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
		t.Input(),
		stt,
		agg.User(),
		model,
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

	// Load the flow graph and join it to the handlers. The config is validated
	// as it loads; building the Flow checks that every tool it names exists and
	// is a valid NodeFunction.
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

	events.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		if err := flowManager.Initialize(ctx, flow.InitialNode()); err != nil {
			slog.Error("initialize flow", "err", err)
		}
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
