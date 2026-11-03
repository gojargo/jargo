// Command transports-livekit is a voice agent that joins a LiveKit room,
// without the development runner.
//
// It greets the first participant to join, and takes text chat from the room
// as though it had been spoken: each data message interrupts the bot and goes
// to the LLM as a transcription.
//
// The room and the server come from the environment: LIVEKIT_URL,
// LIVEKIT_API_KEY, LIVEKIT_API_SECRET and LIVEKIT_ROOM_NAME. A user token for
// the room is logged, for joining it from a LiveKit client.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… go run ./examples/transports/transports-livekit
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/livekit"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/joho/godotenv"
)

func run(ctx context.Context) error {
	url, token, roomName, err := runner.ConfigureLiveKit()
	if err != nil {
		return err
	}
	slog.Info("joining LiveKit room", "room", roomName)
	conn, err := livekit.Connect(livekit.Config{URL: url, Token: token})
	if err != nil {
		return err
	}
	t := livekit.NewTransport(conn, transport.DefaultParams())

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction("You are a helpful assistant in a voice conversation. Your responses will be " +
		"spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond " +
		"to what the user said in a creative, helpful, and brief way.")

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

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

	// Register an event handler so we can play the audio when the participant
	// joins.
	events.On(t.Events(), livekit.EventFirstParticipantJoined, func(context.Context, string) {
		time.Sleep(time.Second)
		worker.QueueFrame(frames.NewTTSSpeakFrame(
			"Hello there! How are you doing today? Would you like to talk about the weather?"))
	})

	// Register an event handler to receive data from the participant via text
	// chat in the LiveKit room. It is used as a transcription: it interrupts the
	// bot, goes to the LLM, and the answer goes back to the participant as
	// audio.
	t.Events().Add(livekit.EventDataReceived, func(_ context.Context, _ any, args ...any) {
		data, _ := args[0].([]byte)
		participantID, _ := args[1].(string)
		slog.Info("Received data from participant", "participant", participantID, "data", string(data))
		var message struct {
			Timestamp string `json:"timestamp"`
			Message   string `json:"message"`
		}
		if err := json.Unmarshal(data, &message); err != nil {
			slog.Error("data is not a chat message", "err", err)
			return
		}
		worker.QueueFrames([]frames.Frame{
			frames.NewInterruptionFrame(),
			frames.NewUserStartedSpeakingFrame(),
			frames.NewTranscriptionFrame(message.Message, participantID, message.Timestamp),
			frames.NewUserStoppedSpeakingFrame(),
		})
	})

	return r.Run(ctx, workers.RunOptions{})
}

func main() {
	_ = godotenv.Overload()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	if err := run(context.Background()); err != nil {
		slog.Error("transports-livekit", "err", err)
		os.Exit(1)
	}
}
