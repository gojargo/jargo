// Command realtime-openai-locally-driven-turns is a voice agent on the OpenAI
// Realtime API, a speech-to-speech model, with turn detection driven locally.
//
// By default the Realtime API drives the conversation with its own server-side
// VAD (see realtime-openai). This variant turns the server's turn detection off
// and drives turn boundaries locally, with a Silero VAD in the user aggregator.
// Take this path to have a turn analyzer such as Smart Turn decide when the
// user is done speaking, or to have UserStartedSpeakingFrame and
// UserStoppedSpeakingFrame fire from the same source as InterruptionFrame.
//
// Locally decided turn boundaries are a heuristic and may not match the turns
// the server would decide. The Realtime API generally expects its own turn
// detection to drive the conversation, and turning it off puts that
// responsibility on the application. Prefer the server's turns (the
// realtime-openai example) without a specific reason to drive them locally.
//
//	OPENAI_API_KEY=… go run ./examples/realtime/realtime-openai-locally-driven-turns
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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/provider/openai/realtime/protocol"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
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
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
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
		SessionProperties: protocol.SessionProperties{
			Audio: &protocol.AudioConfiguration{
				Input: &protocol.AudioInput{
					Transcription: &protocol.InputAudioTranscription{},
					// Turn the server's turn detection off: this example drives turn
					// boundaries locally, with the VAD in the user aggregator below.
					TurnDetection:  protocol.NoTurnDetection(),
					NoiseReduction: &protocol.InputAudioNoiseReduction{Type: "near_field"},
				},
			},
		},
	})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello!"})
	convo.SetTools(tools())

	// Drive turn detection locally, with a VAD in the user aggregator. Realtime
	// service mode is detected on its own and (by default) drops the wait for a
	// transcript at the end of a turn, so the local VAD can drive turn
	// boundaries on the latency-critical path. The stop window is longer than
	// the 0.2s default on purpose: with the server's detection off, padding the
	// end of speech with a bit more silence seems to do a bit better.
	vadParams := vad.DefaultParams()
	vadParams.StopSecs = 0.5
	silero, err := vad.NewSilero(vad.WithParams(vadParams))
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
		Observers:               []processor.Observer{observers.NewTranscriptionLog(observers.TranscriptionLogConfig{})},
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			select {
			case <-time.After(toolsAddedAfter):
			case <-ctx.Done():
				return
			}
			worker.QueueFrames([]frames.Frame{frames.NewLLMSetToolsFrame(append(tools(), newsTool()))})
		}()
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
