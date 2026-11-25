// Command features-add-tool-change-messages is a validation harness for
// aggregators.WithToolChangeMessages.
//
// When tools change mid-conversation, LLMs can hallucinate around tool calls in
// a few ways: calling a tool that has been removed, refusing to call a tool
// that has been added back (because recent context is full of "I can't"
// answers), or making up an answer (or tool-call-shaped text) when no tool is
// available. With tool change messages on, the aggregators append a developer
// message whenever an LLMSetToolsFrame changes the advertised tools, so the
// LLM stays in sync with what is actually available.
//
// The harness flips the advertised tools on a turn counter:
//
//	Phase 0 (turns 1 to 4):  weather tool ACTIVE, confirm the baseline.
//	Phase 1 (turns 5 to 8):  tool REMOVED, keep asking for the weather.
//	Phase 2 (turn 9 onward): tool ADDED BACK, does the LLM call it again?
//
// Set ADD_TOOL_CHANGE_MESSAGES=0 to turn the mitigation off and see the
// unmitigated behavior. It is on by default.
//
// The LLM is Llama 3.1 8B Instruct on a local Ollama, one of the more
// hallucination-prone of the easily accessible models. Pull it once with
// `ollama pull llama3.1:8b` and make sure `ollama serve` is running.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/features/features-add-tool-change-messages
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// The turns at which the weather tool is removed and then added back.
const (
	removeAtTurn = 5 // tool gone from this turn on
	readdAtTurn  = 9 // tool back from this turn on
)

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
}

// tools describes the functions to the LLM, each carrying its own handler.
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses " +
	"will be spoken aloud, so avoid emojis, bullet points, or other " +
	"formatting that can't be spoken. Respond briefly and naturally. " +
	"If the user asks for the current weather, use the `get_current_weather` " +
	"function if it's available. IMPORTANT: if you do not have access to the function, " +
	"say something along the lines of 'Sorry, I can't check the weather right now.'."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	addToolChangeMessages, set := os.LookupEnv("ADD_TOOL_CHANGE_MESSAGES")
	enabled := !set || addToolChangeMessages == "1"
	slog.Info("Starting add_tool_change_messages demo bot", "ADD_TOOL_CHANGE_MESSAGES", enabled)

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	// Llama 3.1 8B Instruct is one of the more hallucination-prone of the
	// easily accessible models, which is what this harness wants.
	model := ollama.NewLLM(chat.LLMConfig{Model: "llama3.1:8b"})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	// The tools advertised in the context carry their handlers, and so do the
	// ones advertised later through an LLMSetToolsFrame.
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	opts := []aggregators.Option{aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero})}
	if enabled {
		opts = append(opts, aggregators.WithToolChangeMessages())
	}
	agg := aggregators.New(convo, opts...)

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		model,
		tts,
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

	// Phase controller: roughly four turns per phase.
	userTurnCount := 0
	event.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(context.Context, aggregators.UserTurnStopped) {
			userTurnCount++
			slog.Info("=== User turn complete ===", "turn", userTurnCount)

			switch userTurnCount {
			case removeAtTurn - 1:
				slog.Info("=== Phase 1: weather tool REMOVED. Keep asking about the weather " +
					"to exercise hallucination scenarios. ===")
				worker.QueueFrame(frames.NewLLMSetToolsFrame(nil))
			case readdAtTurn - 1:
				slog.Info("=== Phase 2: weather tool RE-ADDED. Ask for the weather again. " +
					"Does the LLM call it, or keep refusing? (THIS IS THE TEST.) ===")
				worker.QueueFrame(frames.NewLLMSetToolsFrame(tools()))
			}
		})

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		slog.Info("=== Phase 0: weather tool ACTIVE. Ask for the weather a few times " +
			"to confirm it's working. ===")
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself briefly to the user, then invite them " +
				"to ask about the weather.",
		})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
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
