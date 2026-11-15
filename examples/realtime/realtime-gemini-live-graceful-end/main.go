// Command realtime-gemini-live-graceful-end is a voice agent on Gemini Live, a
// speech-to-speech model, that ends the conversation itself: after three
// answers it says that is all the time it has, calls end_conversation, and
// says goodbye once the tool reports success.
//
// The goodbye can follow the tool call because the Gemini Live service holds
// the end of the pipeline back until the bot finishes its current turn.
//
//	GOOGLE_API_KEY=… go run ./examples/realtime/realtime-gemini-live-graceful-end
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// fetchWeatherFromAPI answers get_current_weather.
func fetchWeatherFromAPI(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Format string `json:"format"`
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

// fetchRestaurantRecommendation answers get_restaurant_recommendation.
func fetchRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	return params.Result(ctx, `{"name": "The Golden Dragon"}`, nil)
}

// endConversation reports success, then ends the pipeline.
func endConversation(ctx context.Context, params llm.FunctionCallParams) error {
	if err := params.Result(ctx, `{"success": true}`, nil); err != nil {
		return err
	}
	return params.LLM.PushFrame(ctx, frames.NewEndWorkerFrame(), processor.Downstream)
}

// tools are the functions above, each carrying its own handler, and Google
// Search, which Gemini runs itself.
func tools() frames.ToolsSchema {
	return frames.ToolsSchema{
		Standard: []frames.Tool{
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
				Handler: llm.FunctionCallHandler(fetchWeatherFromAPI),
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
				Handler: llm.FunctionCallHandler(fetchRestaurantRecommendation),
			},
			{
				Name:        "end_conversation",
				Description: "Gracefully end the conversation",
				Parameters:  json.RawMessage(`{"type": "object", "properties": {}, "required": []}`),
				Handler:     llm.FunctionCallHandler(endConversation),
			},
		},
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

// systemInstruction has the model say something after calling
// end_conversation. That works because the Gemini Live service holds the end of
// the pipeline back until the bot finishes its current turn; and with Gemini
// 3.1 Flash Live, the model won't reliably report ending its turn until it says
// something after the tool call, which is why the instruction is structured
// the way it is.
const systemInstruction = `
You are a helpful assistant who can answer questions and use tools.

You have three tools available to you:
1. get_current_weather: Use this tool to get the current weather in a specific location.
2. get_restaurant_recommendation: Use this tool to get a restaurant recommendation in a specific location.
3. end_conversation: Use this tool to gracefully end the conversation.

After you've responded to the user three times, do the following:
1. Politely let them know that that's all the time you have today (but don't say "goodbye" yet).
2. Then immediately call the end_conversation function. *DO NOT FORGET TO DO THIS STEP.*
3. After the tool reports success, say goodbye.
`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	s2s := live.New(live.Config{
		APIKey:       os.Getenv("GOOGLE_API_KEY"),
		Instructions: systemInstruction,
		Tools:        tools(),
	})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello."})

	// Gemini Live doesn't emit user-turn frames. Server-side VAD is on by
	// default; to surface turn frames (for RTVI speech events, turn observers,
	// and so on) give the user aggregator local turn detection with
	// aggregators.WithVAD. See realtime-gemini-live for the full discussion.
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
