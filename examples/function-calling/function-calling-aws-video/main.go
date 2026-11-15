// Command function-calling-aws-video is a voice agent whose LLM calls a tool
// that looks at the user's camera: Claude on AWS Bedrock.
//
// The fetch_user_image tool asks the transport for an image from the user's
// camera. The image goes into the conversation with the tool's result, so the
// LLM can describe what it sees. The camera is captured only when the tool asks
// for an image. Speech-to-text is Deepgram and text-to-speech Cartesia. While
// the tool runs, the bot says it is checking. The AWS credentials come from
// the environment, as the AWS SDK reads them.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… \
//	  go run ./examples/function-calling/function-calling-aws-video
//
// Then open http://localhost:7860, connect with your camera on, and ask what
// it sees.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/aws/bedrock"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// fetchUserImage fetches the user image and pushes it to the LLM.
//
// It pushes a UserImageRequestFrame upstream to the transport. The transport
// then takes an image from the user's video and pushes a UserImageRawFrame
// downstream, which the assistant aggregator adds to the conversation. The
// call's result callback is invoked once the image is retrieved and processed.
func fetchUserImage(ctx context.Context, params llm.FunctionCallParams) error {
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
			Name: "fetch_user_image",
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
			Handler: llm.FunctionCallHandler(fetchUserImage),
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. Respond to what the user said in a creative, " +
	"helpful, and brief way. You are able to describe images from the user camera."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	// AWS for vision analysis.
	model, err := bedrock.NewLLM(ctx, bedrock.Config{
		Model:       "us.anthropic.claude-sonnet-4-6",
		Region:      "us-west-2",
		Temperature: new(0.8),
	})
	if err != nil {
		return err
	}
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

	events.On(t.Events(), transport.EventClientConnected, func(_ context.Context, client any) {
		slog.Info("Client connected", "client", client)

		clientID := t.GetClientID(client)

		// Kick off the conversation.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user briefly; don't mention the camera. " +
				"Use '" + clientID + "' as the user ID during function calls.",
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
