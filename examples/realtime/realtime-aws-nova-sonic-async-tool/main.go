// Command realtime-aws-nova-sonic-async-tool is a voice agent on Amazon Nova
// Sonic, a speech-to-speech model, whose tool is asynchronous.
//
// The get_current_weather tool is not canceled when the user interrupts, and
// simulates a slow API call (ten seconds). While the call is in flight the
// conversation continues; the result arrives later and is handed to Nova Sonic
// as the call's tool result, so the model can work it into its next turn.
//
//	AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1 \
//	  go run ./examples/realtime/realtime-aws-nova-sonic-async-tool
//
// Then open http://localhost:7860, connect, and ask about the weather.
package main

import (
	"context"
	"encoding/json"
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
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// getCurrentWeather gets the current weather. It simulates a long-running API
// call, to show that the conversation continues while the tool is in flight.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Location string `json:"location"`
		Format   string `json:"format"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	select {
	case <-time.After(10 * time.Second):
	case <-ctx.Done():
		return ctx.Err()
	}
	temperature := 15 + rand.IntN(16) //nolint:gosec // a made-up temperature
	if args.Format == "fahrenheit" {
		temperature = 60 + rand.IntN(26) //nolint:gosec // a made-up temperature
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

// tools describes the function to the model, carrying its handler. It is not
// canceled when the user interrupts, which makes its calls asynchronous.
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
		Handler:              llm.FunctionCallHandler(getCurrentWeather),
		CancelOnInterruption: new(false),
	}}
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

const systemInstruction = "You are a friendly assistant. The user and you will engage in a spoken " +
	"dialog exchanging the transcripts of a natural real-time conversation. " +
	"Keep your responses short, generally two or three sentences for chatty " +
	"scenarios. When the user asks for the weather, call get_current_weather. " +
	"While you wait for the result, keep chatting with the user. When the " +
	"result arrives, share it with the user naturally."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	s2s := novasonic.New(novasonic.Config{
		AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
		Region:          os.Getenv("AWS_REGION"),
		Voice:           "tiffany",
		Instructions:    systemInstruction,
	})

	// Nova Sonic doesn't emit user-turn frames. To get them (for RTVI speech
	// events, turn observers, and so on) give the user aggregator local turn
	// detection with aggregators.WithVAD. See realtime-aws-nova-sonic for the
	// full discussion.
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
