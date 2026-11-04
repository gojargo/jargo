// Command function-calling-missing-handler demonstrates, on purpose, a tool the
// LLM is told about but that has no handler.
//
// When a tool is advertised to the LLM in the context but has no handler
// (neither set on its frames.Tool nor registered with RegisterFunction), the
// LLM still emits a tool call, and nothing would answer it, leaving the
// conversation stuck. The LLM service recovers:
//
//   - It logs an error telling a developer error (a tool advertised with no
//     handler) apart from a hallucination (a tool not advertised), naming the
//     tool.
//   - It answers with a neutral terminal tool result ("The function `X` is not
//     currently available."), so the call still ends with a normal tool result.
//
// The weather tool here deliberately has no handler. Ask the bot about the
// weather and observe:
//
//  1. The LLM emits a tool call for get_current_weather.
//  2. An error is logged: "tool is advertised to the model but has no handler".
//  3. The terminal tool result is fed back to the LLM.
//  4. The LLM answers in voice from that result (typically something like "the
//     weather function isn't available right now").
//
// It uses the OpenAI LLM service with defaults. Swap in another provider to
// check the behavior elsewhere. Speech-to-text is Deepgram and text-to-speech
// Cartesia.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/function-calling/function-calling-missing-handler
//
// Then open http://localhost:7860, connect, and ask about the weather.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// weatherTool describes the weather function to the LLM. Its Handler is left
// unset on purpose.
//
// *** DELIBERATELY OMITTED ***
// The whole point of this example is to demonstrate the missing-handler
// recovery path. To wire the tool up correctly, either set the handler on the
// tool (recommended):
//
//	Handler: llm.FunctionCallHandler(fetchWeatherFromAPI),
//
// or register it on the LLM service:
//
//	model.RegisterFunction("get_current_weather", fetchWeatherFromAPI)
func weatherTool() frames.Tool {
	return frames.Tool{
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
	}
}

// tools are the tools the LLM is told about.
func tools() []frames.Tool {
	return []frames.Tool{weatherTool()}
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses " +
	"will be spoken aloud, so avoid emojis, bullet points, or other " +
	"formatting that can't be spoken. Respond briefly and naturally. " +
	"Always use the get_current_weather function to answer questions " +
	"about the current weather."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting missing-handler demo bot (no handler is registered on purpose)")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		slog.Info("=== Ask for the weather. Watch for an error about the missing " +
			"handler, and listen for the LLM's response based on the recovery " +
			"message. ===")
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself briefly to the user, then invite " +
				"them to ask about the weather.",
		})
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
