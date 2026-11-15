// Command vision-gemini-flash is a voice agent that describes an image:
// Google Gemini.
//
// When a client connects, the bot is given an image and a question about it,
// and answers aloud; the conversation then goes on by voice. The image and the
// question come from the session's body ("image_path" and "question"), or
// default to a picture of a cat and "Describe this image". Speech-to-text is
// Deepgram and text-to-speech Cartesia.
//
//	DEEPGRAM_API_KEY=… GOOGLE_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/vision/vision-gemini-flash
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. Respond to what the user said in a creative, " +
	"helpful, and brief way. You are also able to describe images."

// scriptDir is the directory this source file is in.
func scriptDir() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	return filepath.Dir(file)
}

// openImage reads an image file into its pixels, packed 8-bit red, green and
// blue, and its size.
func openImage(path string) (pixels []byte, width, height int, err error) {
	f, err := os.Open(path) //nolint:gosec // the path is the session's own input
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, 0, 0, err
	}
	b := img.Bounds()
	pixels = make([]byte, 0, b.Dx()*b.Dy()*3)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			pixels = append(pixels, byte(r>>8), byte(g>>8), byte(bl>>8))
		}
	}
	return pixels, b.Dx(), b.Dy(), nil
}

// imageMessage builds the opening message: the image the session's body names
// and its question, or the default ones when the body is empty.
func imageMessage(raw json.RawMessage) (frames.Message, error) {
	var body map[string]any
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &body); err != nil {
			return frames.Message{}, err
		}
	}
	if len(body) == 0 {
		body = map[string]any{
			"image_path": filepath.Join(scriptDir(), "..", "assets", "cat.jpg"),
			"question":   "Describe this image",
		}
	}

	imagePath, _ := body["image_path"].(string)
	question, _ := body["question"].(string)

	pixels, width, height, err := openImage(imagePath)
	if err != nil {
		return frames.Message{}, err
	}
	return frames.CreateImageMessage(frames.RoleUser, "RGB", width, height, pixels, question)
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")

		// Kick off the conversation.
		message, err := imageMessage(args.Base().Body)
		if err != nil {
			slog.Error("Loading the image", "err", err)
			return
		}
		convo.AddMessage(message)
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
