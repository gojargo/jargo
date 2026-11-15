// Command main is the main transport worker of the redis-handoff example, run
// on machine A.
//
// It handles the audio (speech-to-text and text-to-speech) and bridges frames
// to the bus. The LLM workers run as separate processes (the llm command of
// the example), possibly on other machines, and connect to the same
// Redis-backed worker bus.
//
// The example is a module of its own, so only it depends on a Redis client.
// From this directory's parent:
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./main -redis-url redis://localhost:6379
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"sync"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/bus"
	redisbus "github.com/gojargo/jargo/bus/network/redis"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
	goredis "github.com/redis/go-redis/v9"
)

const mainName = "acme"

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	WebRTC:  transport.DefaultParams,
}

// busOptions are the command-line options that say which bus to join.
type busOptions struct {
	redisURL string
	channel  string
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments, opts busOptions) error {
	redisOpts, err := goredis.ParseURL(opts.redisURL)
	if err != nil {
		return err
	}
	b := redisbus.NewBus(redisbus.Config{Client: goredis.NewClient(redisOpts), Channel: opts.channel})
	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{Bus: b.Bus, HandleInterrupt: &handleSigint})

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", // Jacqueline
	})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	bridge := bus.NewBridgeProcessor(bus.BridgeConfig{
		Bus:        r.Bus(),
		WorkerName: mainName,
	})

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		bridge,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Name: mainName,
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	r.AddWorkers(ctx, worker)

	// The remote LLM workers may take a moment to register on the bus. The
	// greeter is activated only once both the client is connected and the
	// worker has been seen in the registry.
	var (
		mu              sync.Mutex
		clientConnected bool
		greeterReady    bool
	)

	maybeActivate := func(ctx context.Context) {
		mu.Lock()
		ready := clientConnected && greeterReady
		mu.Unlock()
		if !ready {
			return
		}
		worker.ActivateWorker(ctx, "greeter", workers.ActivateOptions{
			Args: llmworker.ActivationArgs{
				Messages: []map[string]any{{
					"role": "developer",
					"content": "Welcome the user to Acme Corp, mention the available " +
						"products and ask how you can help.",
				}},
			},
		})
	}

	r.WorkerRegistry().Watch(ctx, "greeter", "activate-greeter",
		func(ctx context.Context, _ registry.WorkerReadyData) {
			mu.Lock()
			greeterReady = true
			mu.Unlock()
			maybeActivate(ctx)
		})

	events.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		mu.Lock()
		clientConnected = true
		mu.Unlock()
		maybeActivate(ctx)
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

func main() {
	redisURL := flag.String("redis-url", "redis://localhost:6379", "Redis URL")
	channel := flag.String("channel", "jargo:acme", "Redis pub/sub channel")

	runner.Main(func(ctx context.Context, args runner.Arguments) error {
		t, err := runner.CreateTransport(ctx, args, transportParams)
		if err != nil {
			return err
		}
		return runBot(ctx, t, args, busOptions{redisURL: *redisURL, channel: *channel})
	})
}
