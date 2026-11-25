// Command function-calling-google is a voice agent whose LLM calls tools:
// Google Gemini, with a tool that looks at the user's camera.
//
// The get_image tool asks the transport for an image from the user's camera.
// The image goes into the conversation with the tool's result, so the LLM can
// answer questions about what it sees. The camera is captured only when the
// tool asks for an image. Speech-to-text is Deepgram and text-to-speech
// Cartesia. While the tools run, the bot says it is checking.
//
//	DEEPGRAM_API_KEY=… GOOGLE_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/function-calling/function-calling-google
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

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
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
	}
	_ = json.Unmarshal(params.Arguments, &args)
	return params.Result(ctx, "The weather in "+args.Location+" is currently 72 degrees and sunny.", nil)
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
			Handler: llm.FunctionCallHandler(getCurrentWeather),
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

You have access to three tools: get_current_weather, get_restaurant_recommendation, and get_image.

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

	model := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
	model.SetSystemInstruction(systemPrompt)

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
