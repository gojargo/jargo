// Command tts-deepgram-flux changes the settings of Deepgram Flux
// text-to-speech while the conversation runs: every ten seconds after the
// client connects, it changes the speed twice, then the expressivity, then the
// voice.
//
// Speech-to-text is Deepgram Flux and the LLM OpenAI.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/update-settings/tts/tts-deepgram-flux
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
	" to what the user said in a creative, helpful, and brief way."

// updateAfter is how long the bot waits before each settings change.
const updateAfter = 10 * time.Second

// sleep waits for updateAfter, and reports false when ctx ends first.
func sleep(ctx context.Context) bool {
	select {
	case <-time.After(updateAfter):
		return true
	case <-ctx.Done():
		return false
	}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewFluxSTT(deepgram.FluxConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := deepgram.NewFluxTTS(deepgram.FluxTTSConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

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
		llm,             // LLM
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
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			// Speed is the one setting Flux can change on the open connection, so
			// these updates keep the acoustic state that makes the voice consistent.
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Deepgram Flux TTS settings: speed=0.85")
			worker.QueueFrame(frames.NewTTSUpdateSettingsFrame(&deepgram.FluxTTSSettings{
				Speed: settings.Set(0.85),
			}))

			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Deepgram Flux TTS settings: speed=1.15")
			worker.QueueFrame(frames.NewTTSUpdateSettingsFrame(&deepgram.FluxTTSSettings{
				Speed: settings.Set(1.15),
			}))

			// Voice and expressivity are fixed for the life of a connection, so
			// these reconnect.
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Deepgram Flux TTS settings: expressivity=2")
			worker.QueueFrame(frames.NewTTSUpdateSettingsFrame(&deepgram.FluxTTSSettings{
				Expressivity: settings.Set(2),
			}))

			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Deepgram Flux TTS settings: voice=\"flux-cole-en\"")
			worker.QueueFrame(frames.NewTTSUpdateSettingsFrame(&deepgram.FluxTTSSettings{
				Voice: settings.Set("flux-cole-en"),
			}))
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
