// Command function-calling-advanced-functionschema is a voice agent whose LLM
// calls a tool whose schema is written out by hand. The LLM is OpenAI.
//
// A tool's schema is written out explicitly when it needs control a plain
// description of the arguments cannot give: here a strict enum on the format
// parameter. The handler reads the arguments the model supplied from the call
// (the schema, not the handler, defines the tool's shape), and is bundled on
// the tool, which the conversation lists: the handler is registered wherever
// the tool is advertised.
//
// Speech-to-text is OpenAI's realtime transcription and text-to-speech OpenAI.
// While the tool runs, the bot says it is checking.
//
//	OPENAI_API_KEY=… \
//	  go run ./examples/function-calling/function-calling-advanced-functionschema
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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// fetchCurrentWeather answers get_current_weather. It reads the arguments the
// model supplied from the call. Set the tool's CancelOnInterruption and
// TimeoutSecs to override the default call options.
func fetchCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Location string `json:"location"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	slog.Info("Fetching weather for " + args.Location)
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
}

// tools describes the function to the LLM with its schema written out, which
// gives precise control over the parameters (here, a format enum).
func tools() []frames.Tool {
	return []frames.Tool{{
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
		Handler: llm.FunctionCallHandler(fetchCurrentWeather),
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses " +
	"will be spoken aloud, so avoid emojis, bullet points, or other " +
	"formatting that can't be spoken. Always use the get_current_weather " +
	"function to answer questions about the current weather."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := realtime.NewSTT(realtime.STTConfig{APIKey: os.Getenv("OPENAI_API_KEY")})

	tts := chat.NewTTS(chat.TTSConfig{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Instructions: "Please speak clearly and at a moderate pace.",
		Voice:        "ballad",
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	model.OnFunctionCallsStarted(func(ctx context.Context, _ []frames.ToolCall) {
		_ = tts.QueueFrame(ctx, frames.NewTTSSpeakFrame("Let me check on that."), processor.Downstream)
	})

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
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
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
