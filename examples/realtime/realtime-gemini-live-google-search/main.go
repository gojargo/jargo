// Command realtime-gemini-live-google-search is a voice agent on Gemini Live, a
// speech-to-speech model, that answers with the latest news through Google
// Search, which Gemini runs itself.
//
//	GOOGLE_API_KEY=… go run ./examples/realtime/realtime-gemini-live-google-search
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// tools offers Google Search, in Gemini's own format.
func tools() frames.ToolsSchema {
	return frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeGemini: {map[string]any{"google_search": map[string]any{}}},
		},
	}
}

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

const systemInstruction = `
You are an expert at providing the most recent news from any place. ` +
	`Your responses will be converted to audio, so avoid using special characters or overly complex formatting.

Always use the google search API to retrieve the latest news. You must also use it to check which day is today.

You can:
- Use the Google search API to check the current date.
- Provide the most recent and relevant news from any place by using the google search API.
- Answer any questions the user may have, ensuring your responses are accurate and concise.

Start each interaction by asking the user about which place they would like to know the information.
`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	s2s := live.New(live.Config{
		APIKey:       os.Getenv("GOOGLE_API_KEY"),
		Voice:        "Puck", // Aoede, Charon, Fenrir, Kore, Puck
		Instructions: systemInstruction,
		Tools:        tools(),
	})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{
		Role: frames.RoleDeveloper,
		Text: "Start by greeting the user warmly, introducing yourself, and mentioning the current day. " +
			"Be friendly and engaging to set a positive tone for the interaction.",
	})

	// Gemini Live doesn't emit user-turn frames. Server-side VAD is on by
	// default; to surface turn frames (for RTVI speech events, turn observers,
	// and so on) give the user aggregator local turn detection with
	// aggregators.WithVAD. See realtime-gemini-live for the full discussion.
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),       // Transport user input
		agg.User(),      // User responses
		s2s,             // LLM
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
