// Command features-processor-usable shows processor health, and what a bot
// does about it.
//
// Every processor says whether it can still do its job (Usable), raises
// processor.EventUsableChanged when that changes, and stamps each error it
// reports with a category saying what kind of failure it was. This bot watches
// each processor and says which one is in trouble and what to check, runs the
// TTS behind a failover ServiceSwitcher so losing Cartesia falls over to
// ElevenLabs, and ends once a processor can no longer do its job
// (pipeline.UnusableEnd).
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… ELEVENLABS_API_KEY=… \
//	OPENAI_API_KEY=… go run ./examples/features/features-processor-usable
//
// ELEVENLABS_VOICE_ID optionally picks the ElevenLabs voice. Then open
// http://localhost:7860 and connect.
package main

import (
	"context"
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
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	ttsCartesia := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	ttsElevenLabs := elevenlabs.NewTTS(elevenlabs.Config{
		APIKey:  os.Getenv("ELEVENLABS_API_KEY"),
		VoiceID: os.Getenv("ELEVENLABS_VOICE_ID"),
	})

	// The failover strategy moves to the next provider that can still work,
	// and only once the active one can't. An error Cartesia can carry on from,
	// a dropped websocket it reconnects, costs no switch.
	ttsSwitcher, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{ttsCartesia, ttsElevenLabs},
		pipeline.NewFailoverStrategy,
	)
	if err != nil {
		return err
	}

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
		ttsSwitcher,     // TTS, with a second provider to fall back on
		t.Output(),      // Transport bot output
		agg.Assistant(), // Assistant spoken responses
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout: args.Base().PipelineIdleTimeout,
		// End the bot once a processor can no longer do its job. For the TTS
		// that means both providers are gone, since the switcher answers for
		// them: losing one is something it recovers from on its own.
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	// Watch each processor, so you can tell which one is in trouble, and what
	// to look at when it is. The switcher is in the list on its own account:
	// losing one of its providers doesn't show here, but running out of them
	// does.
	whatToCheck := []struct {
		processor processor.Processor
		check     string
	}{
		{stt, "check DEEPGRAM_API_KEY and the model"},
		{ttsCartesia, "check CARTESIA_API_KEY and the voice id"},
		{ttsElevenLabs, "check ELEVENLABS_API_KEY and the voice id"},
		{ttsSwitcher, "both TTS providers are gone, so there is nothing left to speak with"},
		{llm, "check OPENAI_API_KEY and the model"},
	}
	for _, w := range whatToCheck {
		event.On(w.processor.Events(), processor.EventUsableChanged, func(_ context.Context, usable bool) {
			if !usable {
				slog.Error(w.processor.Name()+" can no longer do its job", "what_to_check", w.check)
			}
		})
	}

	event.On(ttsSwitcher.Strategy().Events(), pipeline.EventServiceSwitched,
		func(_ context.Context, service processor.Processor) {
			slog.Info("TTS failed over to " + service.Name() + "; the bot keeps talking")
		})

	// Report an error, saying whether the processor survived it. The frame's
	// Category says what kind of failure it was, which is what distinguishes
	// "wrong API key" from "this voice doesn't exist", while the source's
	// Usable says whether it is worth trying again.
	event.On(&worker.Registry, pipeline.EventPipelineError, func(_ context.Context, f *frames.ErrorFrame) {
		if f.Source != nil && !f.Source.Usable() {
			slog.Error(f.Source.Name()+" can no longer do its job", "category", f.Category, "error", f.Error)
			return
		}
		source := "<unknown>"
		if f.Source != nil {
			source = f.Source.Name()
		}
		slog.Warn(source+" hit a problem it can recover from", "error", f.Error)
	})

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		worker.Cancel(ctx, "client disconnected")
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

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
