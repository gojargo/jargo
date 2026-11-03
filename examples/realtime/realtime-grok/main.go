// Command realtime-grok is a voice agent on xAI's Grok Voice Agent API, a
// speech-to-speech model: it hears the caller, speaks its answers, and calls
// tools, with turn detection done server-side.
//
// The Grok Voice Agent provides real-time audio streaming with low latency,
// built-in voice activity detection, built-in and custom voices, built-in tools
// (web search, X search, file search), and custom function calling.
//
//	XAI_API_KEY=… go run ./examples/realtime/realtime-grok
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/xai/realtime"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// --- Function handlers ---

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

// getCurrentTime gets the current time.
func getCurrentTime(ctx context.Context, params llm.FunctionCallParams) error {
	now := time.Now()
	result, err := json.Marshal(map[string]any{
		"time":     now.Format("15:04:05"),
		"date":     now.Format("2006-01-02"),
		"timezone": "local",
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// getRestaurantRecommendation gets a restaurant recommendation.
func getRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Location string `json:"location"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	result, err := json.Marshal(map[string]any{
		"name":     "The Golden Dragon",
		"cuisine":  "Chinese",
		"location": args.Location,
		"rating":   4.5,
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// tools describes the functions to the model, each carrying its own handler.
func tools() []frames.Tool {
	return []frames.Tool{
		{
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
		},
		{
			Name:        "get_current_time",
			Description: "Get the current time.",
			Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
			Handler:     llm.FunctionCallHandler(getCurrentTime),
		},
		{
			Name:        "get_restaurant_recommendation",
			Description: "Get a restaurant recommendation.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. \"San Francisco, CA\"."
					}
				},
				"required": ["location"]
			}`),
			Handler: llm.FunctionCallHandler(getRestaurantRecommendation),
		},
	}
}

// --- Transport configuration ---

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on. No local VAD is needed: Grok detects turns
// server-side.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = `You are a helpful and friendly AI assistant powered by Grok.

    You have access to several tools:
    - Weather information
    - Current time
    - Restaurant recommendations
    - Web search (built-in)
    - X/Twitter search (built-in)

    Your voice and personality should be warm and engaging. Keep your responses
    concise and conversational since this is a voice interaction.

    If the user asks about current events or news, use web search.
    If they ask about what people are saying on social media, use X search.

    Always be helpful and proactive in offering assistance.`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting Grok Voice Agent bot")

	// Grok-specific built-in tools can be turned on here, with WebSearch and
	// XSearch.
	s2s := realtime.New(realtime.Config{
		APIKey:       os.Getenv("XAI_API_KEY"),
		Voice:        "rex",
		Instructions: systemInstruction,
	})

	// The conversation starts with an instruction and the tools.
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello and introduce yourself!"})
	convo.SetTools(tools())

	agg := aggregators.New(convo)

	// In realtime mode the user's transcription comes from Grok, upstream, so
	// the user aggregator goes before the model.
	p := pipeline.New(
		t.Input(),       // Transport user input (audio)
		agg.User(),      // User responses
		s2s,             // Grok Realtime (handles STT + LLM + TTS)
		t.Output(),      // Transport bot output (audio)
		agg.Assistant(), // Assistant spoken responses
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		Observers:               []pipeline.Observer{observers.NewTranscriptionLog(observers.TranscriptionLogConfig{})},
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

	// Grok emits its own user-turn frames from server VAD, so the user turn
	// stopping fires at the turn boundary. In realtime mode its message carries
	// no content, because the transcript is not final yet: the finalized text
	// arrives with the user turn message being added, when the response begins.
	// The assistant message is final when its turn stops.
	events.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("User turn stopped at " + m.Timestamp)
		})
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
