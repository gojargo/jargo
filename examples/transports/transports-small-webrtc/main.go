// Command transports-small-webrtc is a voice agent that builds its WebRTC
// transport itself, from the peer connection the development runner answered,
// rather than through runner.CreateTransport.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… go run ./examples/transports/transports-small-webrtc
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
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

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

// errNotWebRTC is returned for a session that did not arrive over WebRTC.
var errNotWebRTC = errors.New("this bot runs over WebRTC only")

// bot builds the transport from the session's WebRTC connection.
func bot(ctx context.Context, args runner.Arguments) error {
	webrtc, ok := args.(*runner.WebRTCRunnerArguments)
	if !ok {
		return fmt.Errorf("%w, not %T", errNotWebRTC, args)
	}
	// Create a transport using the WebRTC connection.
	t := rtc.NewTransport(webrtc.WebRTCConnection, transport.DefaultParams())
	return runBot(ctx, t, args)
}

func main() {
	runner.Main(bot)
}
