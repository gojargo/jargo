// Command features-service-switcher switches the STT, the LLM and the TTS
// between two providers each while the bot runs.
//
// Each switcher starts on its first service and switches only when asked,
// with a ManuallySwitchServiceFrame. Fifteen seconds after the client
// connects the STT moves to Deepgram, fifteen seconds later the LLM moves to
// Google, and fifteen seconds after that the TTS moves to Deepgram. The tools
// advertised in the context are registered on every LLM the switcher manages,
// active or not, so they keep working across a switch.
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… OPENAI_API_KEY=… GOOGLE_API_KEY=… \
//	  go run ./examples/features/features-service-switcher
//
// Then open http://localhost:7860, connect, and ask about the weather.
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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
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
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
}

// getRestaurantRecommendation gets a restaurant recommendation.
func getRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	return params.Result(ctx, `{"name": "The Golden Dragon"}`, nil)
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

const systemPrompt = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. Respond to what the user said in a creative, " +
	"helpful, and brief way."

// switchDelay is how long the bot waits before each switch.
const switchDelay = 15 * time.Second

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	sttCartesia := cartesia.NewSTT(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})
	sttDeepgram := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	// A nil strategy switches only when asked.
	sttSwitcher, err := pipeline.NewServiceSwitcher([]processor.Processor{sttCartesia, sttDeepgram}, nil)
	if err != nil {
		return err
	}

	ttsCartesia := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})
	ttsDeepgram := deepgram.NewWebsocketTTS(deepgram.WebsocketTTSConfig{
		APIKey: os.Getenv("DEEPGRAM_API_KEY"),
		Voice:  "aura-2-helena-en",
	})

	// A nil strategy switches only when asked.
	ttsSwitcher, err := pipeline.NewServiceSwitcher([]processor.Processor{ttsCartesia, ttsDeepgram}, nil)
	if err != nil {
		return err
	}

	llmOpenAI := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llmOpenAI.SetSystemInstruction(systemPrompt)
	llmGoogle := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
	llmGoogle.SetSystemInstruction(systemPrompt)

	// A nil strategy switches only when asked.
	llmSwitcher, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{llmOpenAI, llmGoogle}, nil)
	if err != nil {
		return err
	}

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	// The tools listed in the context are registered on every LLM the switcher
	// manages, active or not, so they keep working across service switches.
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(), // Transport user input
		sttSwitcher,
		agg.User(),      // User responses
		llmSwitcher,     // LLM
		ttsSwitcher,     // TTS
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
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			for _, service := range []processor.Processor{sttDeepgram, llmGoogle, ttsDeepgram} {
				select {
				case <-ctx.Done():
					return
				case <-time.After(switchDelay):
				}
				slog.Info("Switching to " + service.Name())
				worker.QueueFrames([]frames.Frame{frames.NewManuallySwitchServiceFrame(service)})
			}
		}()
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
