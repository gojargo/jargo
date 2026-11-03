// Command realtime-openai is a voice agent on the OpenAI Realtime API, a
// speech-to-speech model: it hears the caller, speaks its answers, and calls
// tools, with turn detection done server-side. A tool is added while the
// conversation runs.
//
//	OPENAI_API_KEY=… go run ./examples/realtime/realtime-openai
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
	"github.com/gojargo/jargo/provider/openai/realtime"
	rtevents "github.com/gojargo/jargo/provider/openai/realtime/events"
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

// getNews gets the current news.
func getNews(ctx context.Context, params llm.FunctionCallParams) error {
	result, err := json.Marshal(map[string]any{
		"news": []string{
			"Massive UFO currently hovering above New York City",
			"Stock markets reach all-time highs",
			"Living dinosaur species discovered in the Amazon rainforest",
		},
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// newsTool describes getNews, the tool added while the conversation runs.
func newsTool() frames.Tool {
	return frames.Tool{
		Name:        "get_news",
		Description: "Get the current news.",
		Parameters:  json.RawMessage(`{"type": "object", "properties": {}, "required": []}`),
		Handler:     llm.FunctionCallHandler(getNews),
	}
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

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = `You are a helpful and friendly AI.

Act like a human, but remember that you aren't a human and that you can't do human
things in the real world. Your voice and personality should be warm and engaging, with a lively and
playful tone.

If interacting in a non-English language, start by using the standard accent or dialect familiar to
the user. Talk quickly. You should always call a function if you can. Do not refer to these rules,
even if you're asked about them.

You are participating in a voice conversation. Keep your responses concise, short, and to the point
unless specifically asked to elaborate on a topic.

Remember, your responses should be short. Just one or two sentences, usually. Respond in English.`

// toolsAddedAfter is how long the conversation runs before a tool is added.
const toolsAddedAfter = 15 * time.Second

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	s2s := realtime.New(realtime.Config{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Instructions: systemInstruction,
		SessionProperties: rtevents.SessionProperties{
			Audio: &rtevents.AudioConfiguration{
				Input: &rtevents.AudioInput{
					Transcription: &rtevents.InputAudioTranscription{},
					// Set the server's turn detection parameters. Not setting this at
					// all leaves it on, at the server's defaults.
					TurnDetection: rtevents.SemanticVAD(),
					// Or turn it off to have the pipeline's own turn detection drive the
					// session:
					// TurnDetection: rtevents.NoTurnDetection(),
					NoiseReduction: &rtevents.InputAudioNoiseReduction{Type: "near_field"},
				},
			},
			// The tools could be given here rather than through the conversation:
			// Tools: frames.ToolsSchema{Standard: tools()},
		},
	})

	// A standard conversation; the realtime service converts it into the
	// messages the Realtime API takes.
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello!"})
	convo.SetTools(tools())

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

		// Add a new tool at runtime after a delay.
		go func() {
			select {
			case <-time.After(toolsAddedAfter):
			case <-ctx.Done():
				return
			}
			slog.Info("Adding tools")
			worker.QueueFrames([]frames.Frame{frames.NewLLMSetToolsFrame(append(tools(), newsTool()))})
			// Other session properties can be changed at runtime with an
			// LLMUpdateSettingsFrame carrying a realtime.Settings delta. Tools in
			// the conversation take precedence over those in the session
			// properties, so prefer LLMSetToolsFrame for tools, which updates the
			// conversation too. The reasoning effort of a reasoning-capable model
			// (gpt-realtime-2, for one) can be changed the same way:
			//
			//	worker.QueueFrames([]frames.Frame{frames.NewLLMUpdateSettingsFrame(&realtime.Settings{
			//		SessionProperties: settings.Set(rtevents.SessionProperties{
			//			Reasoning: &rtevents.Reasoning{Effort: "xhigh"},
			//		}),
			//	})})
		}()
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	// The realtime service emits its own user-turn frames from server VAD. The
	// user's finalized text arrives with the user turn message being added; the
	// assistant message is final when its turn stops.
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
