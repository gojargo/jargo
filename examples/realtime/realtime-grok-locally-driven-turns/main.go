// Command realtime-grok-locally-driven-turns is a voice agent on xAI's Grok
// Voice Agent API, a speech-to-speech model, with turn detection driven
// locally.
//
// By default Grok drives the conversation with its own server-side VAD (see
// realtime-grok). This variant turns the server's turn detection off (the
// manual mode of Grok's session) and drives turn boundaries locally instead,
// with a Silero VAD in the user aggregator. Use it to have a turn analyzer such
// as Smart Turn decide when the user is done speaking, or to have
// UserStartedSpeakingFrame and UserStoppedSpeakingFrame fire from the same
// source as InterruptionFrame.
//
// Locally decided turn boundaries are a heuristic and may not match the turns
// the server would decide. Prefer the server's turns (the realtime-grok
// example) without a specific reason to drive them locally.
//
//	XAI_API_KEY=… go run ./examples/realtime/realtime-grok-locally-driven-turns
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

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/xai/realtime"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
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
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
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

	s2s := realtime.New(realtime.Config{
		APIKey:       os.Getenv("XAI_API_KEY"),
		Voice:        "rex",
		Instructions: systemInstruction,
		// Turn Grok's server-side turn detection off (manual mode). This example
		// drives turn boundaries locally, with the VAD in the user aggregator
		// below.
		ServerVAD: new(false),
	})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello and introduce yourself!"})
	convo.SetTools(tools())

	// Drive turn detection locally, with a VAD in the user aggregator. Realtime
	// service mode is detected on its own and (by default) drops the wait for a
	// transcript at the end of a turn, so the local VAD can drive turn
	// boundaries on the latency-critical path.
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

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
		Observers:               []pipeline.Observer{observers.NewTranscriptionLog(observers.TranscriptionLogConfig{})},
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	// In realtime mode the user transcript isn't final when the user's turn
	// stops, so the turn-stopped event carries no text; the finalized text
	// arrives with the user turn message being added.
	event.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("User turn stopped at " + m.Timestamp)
		})
	// In realtime mode this is the canonical "user said X" event, decoupled
	// from the turn stopping.
	event.On(agg.User().Events(), aggregators.EventUserTurnMessageAdded,
		func(_ context.Context, m aggregators.UserTurnMessageAdded) {
			slog.Info("Transcript: " + stamp(m.Timestamp) + "user: " + m.Content)
		})
	event.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
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
