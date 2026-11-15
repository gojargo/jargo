// Command voice-speechmatics-vad is a voice agent: Speechmatics speech-to-text
// and text-to-speech, with an OpenAI LLM, using the Speechmatics service's own
// VAD for turn detection.
//
// Diarization tells speakers apart: the first speaker is S1 and the others get
// the labels after it, and the speaker format wraps each result in its
// speaker's label for the LLM. With turn detection set to VAD the service closes
// turns itself, so the pipeline needs no VAD of its own. Set it to external to
// have the pipeline's VAD close turns instead.
//
//	SPEECHMATICS_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/voice/voice-speechmatics-vad
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/speechmatics"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = "You are a helpful British assistant called Sarah in a voice conversation. Your " +
	"responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. " +
	"Always include punctuation in your responses. Give very short replies - do not give longer replies unless " +
	"strictly necessary. Respond to what the user said in a concise, funny, creative and helpful way. Use " +
	"`<Sn/>` tags to identify different speakers - do not use tags in your replies."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := speechmatics.NewSTT(speechmatics.Config{
		APIKey:   os.Getenv("SPEECHMATICS_API_KEY"),
		Language: language.English,
		// VAD mode: the Speechmatics service runs its own VAD and closes turns itself.
		TurnDetectionMode: speechmatics.TurnDetectionVAD,
		EnableDiarization: new(true),
		SpeakerFormat:     "<{speaker}>{text}</{speaker}>",
	})

	tts := speechmatics.NewTTS(speechmatics.TTSConfig{
		APIKey: os.Getenv("SPEECHMATICS_API_KEY"),
		Voice:  "sarah",
	})

	llm := chat.NewLLM(chat.LLMConfig{
		APIKey:      os.Getenv("OPENAI_API_KEY"),
		Temperature: new(0.75),
	})
	llm.SetSystemInstruction(systemInstruction)

	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,
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
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say a short hello to the user."})
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
