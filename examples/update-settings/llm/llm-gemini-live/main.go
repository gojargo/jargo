// Command llm-gemini-live changes the settings of a Gemini Live
// speech-to-speech model while the conversation runs: ten seconds after the
// client connects, it sets the temperature to 0.1.
//
//	GOOGLE_API_KEY=… go run ./examples/update-settings/llm/llm-gemini-live
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/settings"
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
	" to what the user said in a creative, helpful, and brief way."

// updateAfter is how long the bot waits before each settings change.
const updateAfter = 10 * time.Second

// sleep waits for updateAfter, and reports false when ctx ends first.
func sleep(ctx context.Context) bool {
	select {
	case <-time.After(updateAfter):
		return true
	case <-ctx.Done():
		return false
	}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	llm := live.New(live.Config{
		APIKey:       os.Getenv("GOOGLE_API_KEY"),
		Instructions: systemInstruction,
	})

	// Gemini Live doesn't emit user-turn frames. Server-side VAD is enabled by
	// default; to surface turn frames (for RTVI speech events, turn observers,
	// etc.), give the user aggregator local turn detection with
	// aggregators.WithVAD. See realtime-gemini-live for the full discussion.

	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),       // Transport user input
		agg.User(),      // User responses
		llm,             // LLM
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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Gemini Live LLM settings: temperature=0.1")
			worker.QueueFrame(frames.NewLLMUpdateSettingsFrame(&live.Settings{
				Temperature: settings.Set(0.1),
			}))
		}()
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
