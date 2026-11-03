// Command realtime-gemini-live-vertex is a voice agent on Gemini Live through
// Vertex AI, a speech-to-speech model: it hears the caller, speaks its answers,
// and calls tools, with turn detection done server-side.
//
// The tools are configured on the service rather than advertised through the
// conversation, so their handlers register on their own.
//
// GOOGLE_VERTEX_TEST_CREDENTIALS holds a service-account key, the contents of
// the key file.
//
//	GOOGLE_VERTEX_TEST_CREDENTIALS="$(cat key.json)" GOOGLE_CLOUD_PROJECT_ID=… \
//	GOOGLE_CLOUD_LOCATION=us-east4 go run ./examples/realtime/realtime-gemini-live-vertex
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/google/vertex"
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
	temperature := 24
	if args.Format == "fahrenheit" {
		temperature = 75
	}
	result, err := json.Marshal(map[string]any{
		"conditions":  "nice",
		"temperature": temperature,
		"format":      args.Format,
		"timestamp":   time.Now().Format("20060102_150405"),
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// getRestaurantRecommendation gets a restaurant recommendation.
func getRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	return params.Result(ctx, `{"name": "The Golden Dragon"}`, nil)
}

// standardTools describes the functions to the model,
// each carrying its own handler.
func standardTools() []frames.Tool {
	return []frames.Tool{
		{
			Name:        "get_current_weather",
			Description: "Get the current weather",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. San Francisco, CA"
					},
					"format": {
						"type": "string",
						"enum": ["celsius", "fahrenheit"],
						"description": "The temperature unit to use. Infer this from the user's location."
					}
				},
				"required": ["location", "format"]
			}`),
			Handler: llm.FunctionCallHandler(getCurrentWeather),
		},
		{
			Name:        "get_restaurant_recommendation",
			Description: "Get a restaurant recommendation",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. San Francisco, CA"
					}
				},
				"required": ["location"]
			}`),
			Handler: llm.FunctionCallHandler(getRestaurantRecommendation),
		},
	}
}

// tools are the functions above.
func tools() frames.ToolsSchema {
	return frames.ToolsSchema{Standard: standardTools()}
}

// errNoCredentials is returned when no service-account key is given.
var errNoCredentials = errors.New("GOOGLE_VERTEX_TEST_CREDENTIALS is not set")

// credentials are the service-account key in GOOGLE_VERTEX_TEST_CREDENTIALS.
// For Application Default Credentials, set vertex.Credentials.TokenSource to
// the token source golang.org/x/oauth2/google gives instead.
func credentials() (vertex.Credentials, error) {
	key := os.Getenv("GOOGLE_VERTEX_TEST_CREDENTIALS")
	if key == "" {
		return vertex.Credentials{}, errNoCredentials
	}
	return vertex.Credentials{JSON: []byte(key)}, nil
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

const systemInstruction = `
You are a helpful assistant who can answer questions and use tools.

You have three tools available to you:
1. get_current_weather: Use this tool to get the current weather in a specific location.
2. get_restaurant_recommendation: Use this tool to get a restaurant recommendation in a specific location.
`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	creds, err := credentials()
	if err != nil {
		return err
	}
	s2s := vertex.NewS2S(vertex.S2SConfig{
		ProjectID:    os.Getenv("GOOGLE_CLOUD_PROJECT_ID"),
		Location:     os.Getenv("GOOGLE_CLOUD_LOCATION"),
		Credentials:  creds,
		Instructions: systemInstruction,
		Voice:        "Puck", // Aoede, Charon, Fenrir, Kore, Puck
		Tools:        tools(),
	})

	// Gemini Live doesn't emit user-turn frames. Server-side VAD is enabled by
	// default; processors that expect turn frames (RTVI speech events, turn
	// observers) need local turn detection in the user aggregator. See
	// realtime-gemini-live for the full discussion.
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello."})
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
