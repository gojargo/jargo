// Command stt-deepgram-flux changes the settings of Deepgram Flux
// speech-to-text while the conversation runs: ten seconds after the client
// connects, it raises the end-of-turn threshold and adds key terms, and ten
// seconds later it narrows the language hints to Spanish alone.
//
// Text-to-speech is Cartesia and the LLM OpenAI.
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/update-settings/stt/stt-deepgram-flux
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/settings"
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

	// Start with the multilingual model and broad hints so Flux can auto-detect
	// across English and Spanish. A transcription's language reflects
	// whichever language Flux detected for each turn.
	stt := deepgram.NewFluxSTT(deepgram.FluxConfig{
		APIKey:        os.Getenv("DEEPGRAM_API_KEY"),
		Model:         "flux-general-multi",
		LanguageHints: []language.Language{language.English, language.Spanish},
	})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			// Update configure-able fields mid-stream (sent via Configure message,
			// no reconnect needed).
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Deepgram Flux STT settings: eot_threshold, keyterm")
			worker.QueueFrame(frames.NewSTTUpdateSettingsFrame(&deepgram.FluxSettings{
				EOTThreshold: settings.Set(0.8),
				Keyterm:      settings.Set([]string{"jargo", "Deepgram"}),
			}))

			// Detect-then-lock: narrow the hints to a single language mid-stream.
			// Sent as a Configure message, so no reconnect is needed.
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Deepgram Flux STT settings: language_hints=[es]")
			worker.QueueFrame(frames.NewSTTUpdateSettingsFrame(&deepgram.FluxSettings{
				LanguageHints: settings.Set([]string{string(language.Spanish)}),
			}))
		}()
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
