// Command features-screen-share is a bot that can look at the user's camera or
// screen share when asked.
//
// The transport captures both video sources as the client connects, listed in
// VideoInSources. Each is captured with OnRequestOnly, so no frames flow until
// the bot asks for one: the LLM has a tool per source, fetch_camera_image and
// fetch_screen_share_image, and each pushes a UserImageRequestFrame for its
// source. The transport answers with the next frame from that source, which is
// added to the LLM context so the LLM can describe it.
//
// Share your screen from the client and ask what's on it, or turn on your
// camera and ask what it sees.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-screen-share
//
// Then open http://localhost:7860 and connect.
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
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/responses"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	ttsservice "github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// imageArgs are the arguments both image tools take.
type imageArgs struct {
	UserID   string `json:"user_id"`
	Question string `json:"question"`
}

// requestUserImage requests an image from the user's video and adds it to the
// LLM context.
//
// It pushes a UserImageRequestFrame upstream to the transport. The transport
// answers it with the next frame from the requested video source, as a
// UserImageRawFrame that the assistant aggregator adds to the context. The
// result callback is invoked once the image is retrieved and processed.
func requestUserImage(ctx context.Context, params llm.FunctionCallParams, videoSource string) error {
	var args imageArgs
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Debug("Requesting image", "video_source", videoSource, "user_id", args.UserID, "question", args.Question)

	request := frames.NewUserImageRequestFrame(args.UserID)
	request.Text = args.Question
	request.VideoSource = videoSource
	request.AppendToContext = true
	request.FunctionName = params.FunctionName
	request.ToolCallID = params.ToolCallID
	request.ResultCallback = params.Result
	return params.LLM.PushFrame(ctx, request, processor.Upstream)
}

// fetchCameraImage gets an image from the user's camera to answer a question
// about it.
func fetchCameraImage(ctx context.Context, params llm.FunctionCallParams) error {
	return requestUserImage(ctx, params, "camera")
}

// fetchScreenShareImage gets an image of the user's screen share to answer a
// question about it.
func fetchScreenShareImage(ctx context.Context, params llm.FunctionCallParams) error {
	return requestUserImage(ctx, params, transport.ScreenVideoSource)
}

// imageParameters is the schema both image tools share.
const imageParameters = `{
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
}`

// tools describes the functions to the LLM, each carrying its own handler.
func tools() []frames.Tool {
	return []frames.Tool{
		{
			Name:        "fetch_camera_image",
			Description: "Get an image from the user's camera to answer a question about it.",
			Parameters:  json.RawMessage(imageParameters),
			Handler:     llm.FunctionCallHandler(fetchCameraImage),
		},
		{
			Name:        "fetch_screen_share_image",
			Description: "Get an image of the user's screen share to answer a question about it.",
			Parameters:  json.RawMessage(imageParameters),
			Handler:     llm.FunctionCallHandler(fetchScreenShareImage),
		},
	}
}

// videoParams captures the camera and the screen share, each only on request.
func videoParams(p transport.Params) transport.Params {
	p.VideoInEnabled = true
	p.VideoInSources = map[string]transport.VideoInSourceParams{
		"camera":                    {OnRequestOnly: true},
		transport.ScreenVideoSource: {OnRequestOnly: true},
	}
	return p
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: func() transport.Params { return videoParams(transport.DefaultParams()) },
	WebRTC:  func() transport.Params { return videoParams(transport.DefaultParams()) },
}

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. Respond to what the user said in a creative, " +
	"helpful, and brief way. You are able to describe images from the user's camera and screen share."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := responses.NewLLM(responses.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	model.OnFunctionCallsStarted(func(ctx context.Context, _ []frames.ToolCall) {
		speak := frames.NewTTSSpeakFrame("Let me check on that.")
		speak.AppendToContext = false
		_ = tts.QueueFrame(ctx, speak, processor.Downstream)
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
		slog.Info("Client connected")
		clientID := t.GetClientID(client)
		// Kick off the conversation.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user. Use '" + clientID + "' as the user ID during function calls.",
		})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	events.On(tts.Events(), ttsservice.EventTTSRequest, func(_ context.Context, r ttsservice.TTSRequest) {
		slog.Debug("On TTS request", "context_id", r.ContextID, "text", r.Text)
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
