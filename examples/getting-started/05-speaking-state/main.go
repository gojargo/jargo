// Command 05-speaking-state is a voice agent that shows whether it is speaking
// as its video.
//
// The voice agent of 06-voice-agent, with a processor after the
// text-to-speech service that sends one image when the bot starts speaking and
// another when it stops, so the video follows the bot's speaking state.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… go run ./examples/getting-started/05-speaking-state
//
// Then open http://localhost:7860 and connect.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"image"
	_ "image/png"
	"log/slog"
	"os"
	"strings"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// The images shown while the bot speaks and while it waits.
var (
	//go:embed speaking.png
	speakingPNG []byte
	//go:embed waiting.png
	waitingPNG []byte
)

// loadImage decodes an image into its pixels, packed 8-bit red, green and blue,
// and the name of the format it was stored in.
func loadImage(data []byte) ([]byte, string, error) {
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, "", err
	}
	b := img.Bounds()
	pixels := make([]byte, 0, b.Dx()*b.Dy()*3)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			pixels = append(pixels, byte(r>>8), byte(g>>8), byte(bl>>8))
		}
	}
	return pixels, strings.ToUpper(format), nil
}

// imageSyncAggregator sends the speaking image when the bot starts speaking
// and the waiting image when it stops, and passes every frame on.
type imageSyncAggregator struct {
	*processor.Base
	speakingImage       []byte
	speakingImageFormat string
	waitingImage        []byte
	waitingImageFormat  string
}

func newImageSyncAggregator(speaking, waiting []byte) (*imageSyncAggregator, error) {
	a := &imageSyncAggregator{}
	var err error
	if a.speakingImage, a.speakingImageFormat, err = loadImage(speaking); err != nil {
		return nil, err
	}
	if a.waitingImage, a.waitingImageFormat, err = loadImage(waiting); err != nil {
		return nil, err
	}
	a.Base = processor.New("ImageSyncAggregator", a)
	return a, nil
}

// ProcessFrame pushes the image that matches the bot's speaking state when it
// changes, then the frame itself.
func (a *imageSyncAggregator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch f.(type) {
	case *frames.BotStartedSpeakingFrame:
		frame := frames.NewOutputImageRawFrame(a.speakingImage, 1024, 1024, a.speakingImageFormat)
		if err := a.PushFrame(ctx, frame, processor.Downstream); err != nil {
			return err
		}
	case *frames.BotStoppedSpeakingFrame:
		frame := frames.NewOutputImageRawFrame(a.waitingImage, 1024, 1024, a.waitingImageFormat)
		if err := a.PushFrame(ctx, frame, processor.Downstream); err != nil {
			return err
		}
	}

	return a.PushFrame(ctx, f, dir)
}

// videoParams are the parameters of a transport that sends video: audio both
// ways, and 1024 by 1024 video out.
func videoParams() transport.Params {
	p := transport.DefaultParams()
	p.VideoOutEnabled = true
	p.VideoOutWidth = 1024
	p.VideoOutHeight = 1024
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction("You are a helpful assistant in a voice conversation. Your responses will be " +
		"spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond " +
		"to what the user said in a creative, helpful, and brief way.")

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	imageSync, err := newImageSyncAggregator(speakingPNG, waitingPNG)
	if err != nil {
		return err
	}

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		tts,
		imageSync,
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

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
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
