// Command realtime-aws-nova-sonic is a voice agent on Amazon Nova Sonic, a
// speech-to-speech model: it hears the caller, speaks its answers, and calls a
// tool, with turn detection done server-side. Sessions are rotated before the
// AWS time limit, so a conversation can run past it.
//
//	AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1 \
//	  go run ./examples/realtime/realtime-aws-nova-sonic
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/aws/novasonic"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Location string `json:"location"`
		Format   string `json:"format"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	temperature := 15 + rand.IntN(16) //nolint:gosec // a made-up reading
	if args.Format == "fahrenheit" {
		temperature = 60 + rand.IntN(26) //nolint:gosec // a made-up reading
	}
	result, err := json.Marshal(map[string]any{
		"conditions":  "nice",
		"temperature": temperature,
		"location":    args.Location,
		"format":      args.Format,
		"timestamp":   time.Now().Format("20060102_150405"),
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// tools describes the function to the model, carrying its handler.
func tools() []frames.Tool {
	return []frames.Tool{{
		Name:        "get_current_weather",
		Description: "Get the current weather.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"location": {
					"type": "string",
					"description": "The city and state, e.g. \"San Francisco, CA\"."
				},
				"format": {
					"type": "string",
					"description": "The temperature unit to use. Must be either \"celsius\" or \"fahrenheit\". ` +
			`Infer this from the user's location."
				}
			},
			"required": ["location", "format"]
		}`),
		Handler: llm.FunctionCallHandler(getCurrentWeather),
	}}
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

// systemInstruction is the initial system instruction.
const systemInstruction = "You are a friendly assistant. The user and you will engage in a spoken dialog " +
	"exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally " +
	"two or three sentences for chatty scenarios."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	// Session continuation is on by default, so a conversation can run past
	// the AWS session limit of about eight minutes: the service rotates
	// sessions in the background with no gap the user can hear. Tune the
	// threshold here, or turn it off with Enabled: false.
	continuation := novasonic.DefaultSessionContinuationParams()
	// When to start preparing the next session (default six minutes). Lower it
	// (to 20 seconds, say) to see a handover happen quickly while testing.
	continuation.TransitionThreshold = 360 * time.Second

	s2s := novasonic.New(novasonic.Config{
		AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
		// As of 2026-10-08, the regions serving the default model (Nova 2.5
		// Sonic) are us-east-1, us-west-2, eu-north-1 and ap-northeast-1.
		Region:              os.Getenv("AWS_REGION"),
		Voice:               "tiffany",
		Instructions:        systemInstruction,
		SessionContinuation: &continuation,
		// The tools could be given here rather than through the conversation:
		// Tools: frames.ToolsSchema{Standard: tools()},
	})

	// Nova Sonic drives the conversation server-side. It does not, however,
	// emit turn frames (UserStartedSpeakingFrame, UserStoppedSpeakingFrame).
	// The conversation is aggregated without them, but processors that expect
	// them (RTVI, turn observers) need local turn detection in the user
	// aggregator, whose turns are approximate: they may not align with the
	// turns the server decides.
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),
		agg.User(),
		s2s,
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

	// The user's finalized text arrives with the user turn message being added;
	// the assistant message is final when its turn stops.
	events.On(agg.User().Events(), aggregators.EventUserTurnMessageAdded,
		func(_ context.Context, m aggregators.UserTurnMessageAdded) {
			slog.Info("Transcript: " + stamp(m.Timestamp) + "user: " + m.Content)
		})
	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			slog.Info("Transcript: " + stamp(m.Timestamp) + "assistant: " + m.Content)
		})

	return r.Run(ctx, workers.RunOptions{})
}

// stamp formats a transcript line's timestamp, or nothing without one.
func stamp(timestamp string) string {
	if timestamp == "" {
		return ""
	}
	return fmt.Sprintf("[%s] ", timestamp)
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
