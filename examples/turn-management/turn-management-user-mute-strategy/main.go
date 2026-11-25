// Command turn-management-user-mute-strategy mutes the user while the bot
// gives its first answer and while a tool call runs.
//
// The user aggregator takes two mute strategies: MuteUntilFirstBotComplete
// keeps the user's input from the bot until it has finished its first
// response, and FunctionCallUserMute keeps it away while a function call is
// running. The weather tool takes five seconds, to leave time to try talking
// over it. The aggregator raises EventUserMuteStarted and EventUserMuteStopped
// as the user is muted and unmuted.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-user-mute-strategy
//
// Then open http://localhost:7860, connect, and name a city.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	// A delay, to test interruption during function calls.
	slog.Info("Weather API call starting...")
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(5 * time.Second): // 5-second delay
	}
	slog.Info("Weather API call completed")
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

const systemInstruction = "You are a helpful assistant who can check the weather. Always check the weather " +
	"when a location is mentioned. Respond concisely and naturally. Your output will be spoken aloud, so " +
	"avoid special characters that can't easily be spoken, such as emojis or bullet points."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := deepgram.NewWebsocketTTS(deepgram.WebsocketTTSConfig{
		APIKey: os.Getenv("DEEPGRAM_API_KEY"),
		Voice:  "aura-2-helena-en",
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo,
		aggregators.WithMuteStrategies(
			turns.NewMuteUntilFirstBotComplete(),
			turns.NewFunctionCallUserMute(),
		),
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
	)

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		model,           // LLM
		tts,             // TTS
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

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation with a weather-related prompt.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Ask the user what city they'd like to know the weather for.",
		})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	event.OnSignal(agg.User().Events(), aggregators.EventUserMuteStarted, func(context.Context) {
		slog.Info("User mute started")
	})

	event.OnSignal(agg.User().Events(), aggregators.EventUserMuteStopped, func(context.Context) {
		slog.Info("User mute stopped")
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
