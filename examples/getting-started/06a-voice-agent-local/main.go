// Command 06a-voice-agent-local is 06-voice-agent on the local microphone and
// speaker: no server, no browser. It needs a running PulseAudio or PipeWire
// server; wear headphones, or the bot will hear itself.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… go run ./examples/getting-started/06a-voice-agent-local
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/localaudio"
	"github.com/gojargo/jargo/workers"
	"github.com/joho/godotenv"
)

func run(ctx context.Context) error {
	t, err := localaudio.New(localaudio.Config{AppName: "06a-voice-agent-local", Params: transport.DefaultParams()})
	if err != nil {
		return err
	}
	defer t.Close()

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
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	r := workers.NewRunner(workers.RunnerConfig{})
	r.AddWorkers(ctx, worker)

	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
	worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

	return r.Run(ctx, workers.RunOptions{})
}

func main() {
	_ = godotenv.Overload()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	if err := run(context.Background()); err != nil {
		slog.Error("06a-voice-agent-local", "err", err)
		os.Exit(1)
	}
}
