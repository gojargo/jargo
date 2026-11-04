// Command function-calling-openai-responses-async is a voice agent whose LLM
// calls tools: OpenAI's Responses API, with asynchronous tools.
//
// The tools are not canceled when the user interrupts: the conversation goes
// on while they run, and their results arrive when ready. One always outlasts
// its deadline and is canceled; the LLM may cancel another. Speech-to-text is
// Deepgram and text-to-speech Cartesia. While the tools run, the bot says it
// is checking.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/function-calling/function-calling-openai-responses-async
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
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/responses"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// sleep waits for d, or until the call is canceled.
func sleep(ctx context.Context, d time.Duration) error {
	select {
	case <-time.After(d):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// getCurrentWeather gets the current weather. It simulates a long-running API
// call, to show an asynchronous call that an interruption does not cancel.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	if err := sleep(ctx, 15*time.Second); err != nil {
		return err
	}
	slog.Debug("Returning get_current_weather result.")
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
}

// getStockPrice gets the current share price for a stock. It hangs: it sleeps
// far past the deadline it was registered with, so the call is always canceled
// before it can report anything. The result it would eventually have returned
// is distinctive on purpose: if the bot ever quotes a share price, a canceled
// handler's result reached the conversation.
func getStockPrice(ctx context.Context, params llm.FunctionCallParams) error {
	if err := sleep(ctx, 20*time.Second); err != nil {
		return err
	}
	slog.Debug("Returning get_stock_price result.")
	return params.Result(ctx, `{"price": "184.20", "currency": "USD"}`, nil)
}

// writeReport writes a long research report on a topic. It takes long enough
// to still be running when the LLM asks to stop it: listing what is running and
// then calling the cancel tool, with a spoken reply often in between, outlasts
// shorter work.
func writeReport(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Topic string `json:"topic"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	if err := sleep(ctx, 25*time.Second); err != nil {
		return err
	}
	slog.Debug("Returning write_report result.")
	result, err := json.Marshal(map[string]string{"report": "A 5000-word report on " + args.Topic + "."})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
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
			Handler:              llm.FunctionCallHandler(getCurrentWeather),
			CancelOnInterruption: new(false),
			TimeoutSecs:          new(30.0),
		},
		{
			Name:        "write_report",
			Description: "Write a long research report on a topic.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"topic": {
						"type": "string",
						"description": "What the report should cover."
					}
				},
				"required": ["topic"]
			}`),
			Handler:              llm.FunctionCallHandler(writeReport),
			CancelOnInterruption: new(false),
			CancellableByLLM:     new(true),
			TimeoutSecs:          new(120.0),
		},
		{
			Name:        "get_stock_price",
			Description: "Get the current share price for a stock.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"symbol": {
						"type": "string",
						"description": "The ticker symbol, e.g. \"NVDA\"."
					}
				},
				"required": ["symbol"]
			}`),
			Handler:              llm.FunctionCallHandler(getStockPrice),
			CancelOnInterruption: new(false),
			TimeoutSecs:          new(5.0),
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. Respond to what the user said in a creative, " +
	"helpful, and brief way."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := responses.NewLLM(responses.Config{
		APIKey:    os.Getenv("OPENAI_API_KEY"),
		Model:     "gpt-5.4",
		Reasoning: &responses.ReasoningConfig{Effort: "low"},
	})
	model.SetSystemInstruction(systemInstruction)

	model.OnFunctionCallsStarted(func(ctx context.Context, _ []frames.ToolCall) {
		// Avoid appending this filler message to the LLM context: it would alter
		// the conversation history and keep the service's previous-response
		// continuation from matching, forcing a full context resend.
		speak := frames.NewTTSSpeakFrame("Let me check on that.")
		speak.AppendToContext = false
		_ = tts.QueueFrame(ctx, speak, processor.Downstream)
	})

	model.OnFunctionCallsCanceled(func(_ context.Context, calls []frames.ToolCall) {
		for _, call := range calls {
			slog.Info("Function call canceled", "function", call.Name, "tool_call_id", call.ID)
		}
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
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
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
