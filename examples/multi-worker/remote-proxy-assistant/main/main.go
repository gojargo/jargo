// Command main is the main transport worker of the remote-proxy-assistant
// example, with a WebSocket proxy to a remote LLM worker.
//
// It runs the transport with speech-to-text, text-to-speech and a bus bridge,
// and a client proxy that connects to the remote assistant server (the
// assistant command of the example) and forwards the bus frame messages. No
// shared bus is needed: the proxy workers forward bus messages point to point
// over the WebSocket.
//
// Start the assistant server first, then:
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/multi-worker/remote-proxy-assistant/main -remote-url ws://localhost:8765/ws
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/bus"
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
	"github.com/gojargo/jargo/workers/proxy"
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments, remoteURL string) error {
	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})

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

	// Forward the bus frame messages over the WebSocket, so the remote
	// assistant sees the user-side context and can ship back its replies.
	client := proxy.NewClient(proxy.ClientConfig{
		Name:            "proxy",
		URL:             remoteURL,
		LocalWorker:     mainName,
		RemoteWorker:    "assistant",
		ForwardMessages: []bus.Message{&bus.FrameMessage{}},
	})

	r.AddWorkers(ctx, client, worker)

	r.WorkerRegistry().Watch(ctx, "assistant", "activate-assistant",
		func(ctx context.Context, _ registry.WorkerReadyData) {
			slog.Info("Remote assistant ready, activating")
			worker.ActivateWorker(ctx, "assistant", workers.ActivateOptions{
				Args: llmworker.ActivationArgs{
					Messages: []map[string]any{{
						"role": "developer",
						"content": "Welcome the user to Acme Corp, mention the available " +
							"products and ask how you can help.",
					}},
				},
			})
		})

	events.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected, activating proxy")
		worker.ActivateWorker(ctx, "proxy", workers.ActivateOptions{})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

func main() {
	remoteURL := flag.String("remote-url", "ws://localhost:8765/ws", "WebSocket URL of the remote LLM server")

	runner.Main(func(ctx context.Context, args runner.Arguments) error {
		t, err := runner.CreateTransport(ctx, args, transportParams)
		if err != nil {
			return err
		}
		return runBot(ctx, t, args, *remoteURL)
	})
}
