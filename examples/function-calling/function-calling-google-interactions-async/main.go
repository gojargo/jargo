// Command function-calling-google-interactions-async is a voice agent whose
// LLM calls tools: Google Gemini on the Interactions API, with asynchronous
// tools and a tool that looks at the user's camera.
//
// Some tools are not canceled when the user interrupts: the conversation goes
// on while they run, and their results arrive when ready. One always outlasts
// its deadline and is canceled; the LLM may cancel another. The get_image tool
// asks the transport for an image from the user's camera, which goes into the
// conversation with the tool's result. The camera is captured only when the
// tool asks for an image. Speech-to-text is Deepgram and text-to-speech
// Cartesia. While the tools run, the bot says it is checking.
//
//	DEEPGRAM_API_KEY=… GOOGLE_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/function-calling/function-calling-google-interactions-async
//
// Then open http://localhost:7860, connect with your camera on, and ask about
// the weather or what it sees.
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
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/interactions"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
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

// getImage fetches the user image and pushes it to the LLM.
//
// It pushes a UserImageRequestFrame upstream to the transport. The transport
// then takes an image from the user's video and pushes a UserImageRawFrame
// downstream, which the assistant aggregator adds to the conversation. The
// call's result callback is invoked once the image is retrieved and processed.
func getImage(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		UserID   string `json:"user_id"`
		Question string `json:"question"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	slog.Debug(fmt.Sprintf("Requesting image with user_id=%s, question=%s", args.UserID, args.Question))

	// Request a user image frame and indicate that it should be added to the
	// context. Also associate it to the function call. Pass the result callback
	// so it can be invoked when the image is actually retrieved.
	request := frames.NewUserImageRequestFrame(args.UserID)
	request.Text = args.Question
	request.AppendToContext = true
	request.FunctionName = params.FunctionName
	request.ToolCallID = params.ToolCallID
	request.ResultCallback = params.Result
	return params.LLM.PushFrame(ctx, request, processor.Upstream)
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
			Name: "get_image",
			Description: "Fetch the user image and push it to the LLM.\n\n" +
				"When called, this function pushes a UserImageRequestFrame upstream to the\n" +
				"transport. As a result, the transport will request the user image and push a\n" +
				"UserImageRawFrame downstream which will be added to the context by the LLM\n" +
				"assistant aggregator. The result_callback will be invoked once the image is\n" +
				"retrieved and processed.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"user_id": {
						"type": "string",
						"description": "The ID of the user to grab the image from."
					},
					"question": {
						"type": "string",
						"description": "The question that the user is asking about the image."
					}
				},
				"required": ["user_id", "question"]
			}`),
			Handler: llm.FunctionCallHandler(getImage),
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

// videoParams are the parameters of a transport that receives video: audio
// both ways, and the camera captured only to answer image requests.
func videoParams() transport.Params {
	p := transport.DefaultParams()
	p.VideoInEnabled = true
	p.VideoInSources = map[string]transport.VideoInSourceParams{
		transport.CameraVideoSource: {OnRequestOnly: true},
	}
	return p
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: videoParams,
	WebRTC:  videoParams,
}

const systemPrompt = `You are a helpful assistant who converses with a user and answers questions. ` +
	`Respond concisely to general questions.

Your response will be turned into speech so use only simple words and punctuation.

You have access to five tools: get_current_weather, get_stock_price, get_restaurant_recommendation, ` +
	`write_report, and get_image.

You can respond to questions about the weather using the get_current_weather tool.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that ` +
	`indicate you should use the get_image tool are:
- What do you see?
- What's in the video?
- Can you describe the video?
- Tell me about what you see.
- Tell me something interesting about what you see.
- What's happening in the video?
`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := interactions.NewLLM(interactions.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
	model.SetSystemInstruction(systemPrompt)

	model.OnFunctionCallsStarted(func(ctx context.Context, _ []frames.ToolCall) {
		_ = tts.QueueFrame(ctx, frames.NewTTSSpeakFrame("Let me check on that."), processor.Downstream)
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
	// CancelOnInterruption false (set on the tool) makes a call asynchronous.
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

	event.On(t.Events(), transport.EventClientConnected, func(_ context.Context, client any) {
		slog.Info("Client connected", "client", client)

		clientID := t.GetClientID(client)

		// Kick off the conversation.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user. Use '" + clientID + "' as the user ID during function calls.",
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
