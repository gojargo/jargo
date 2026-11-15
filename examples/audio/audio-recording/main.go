// Command audio-recording records the conversation between the user and the
// bot, saving both a merged recording and the separate tracks.
//
// An audiobuffer.Processor after the output transport records the whole
// conversation, starting as soon as the pipeline starts. When the recording
// stops it saves three WAV files in a recordings directory, named with a
// timestamp:
//
//	recordings/
//	    merged_20240315_123456.wav  (both participants)
//	    user_20240315_123456.wav    (the user only)
//	    bot_20240315_123456.wav     (the bot only)
//
// The merged recording comes through OnAudioData and the separate tracks
// through OnTrackAudioData.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/audio/audio-recording
//
// Then open http://localhost:7860, connect, talk, and disconnect.
package main

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/audiobuffer"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// recordingsDir is where the recordings are saved.
const recordingsDir = "recordings"

// saveAudioFile saves audio to a WAV file.
func saveAudioFile(pcm []byte, filename string, sampleRate, numChannels int) {
	if len(pcm) == 0 {
		return
	}
	if err := os.WriteFile(filename, audio.PCMToWAV(pcm, sampleRate, numChannels), 0o600); err != nil {
		slog.Error("Error saving audio", "file", filename, "err", err)
		return
	}
	slog.Info("Audio saved to " + filename)
}

// timestamp names a recording by when it was saved.
func timestamp() string { return time.Now().Format("20060102_150405") }

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

	// The audio buffer processor. Recording starts when the pipeline starts;
	// alternatively, push an AudioBufferStartRecordingFrame or call
	// StartRecording to start it on demand.
	audioBuffer := audiobuffer.New(audiobuffer.Config{
		AutoStart:          true,
		OnRecordingStarted: func() { slog.Info("Recording started") },
		OnRecordingStopped: func() { slog.Info("Recording stopped") },
		// The merged audio.
		OnAudioData: func(pcm []byte, sampleRate, numChannels int) {
			if err := os.MkdirAll(recordingsDir, 0o750); err != nil {
				slog.Error("Error creating the recordings directory", "err", err)
				return
			}
			filename := filepath.Join(recordingsDir, "merged_"+timestamp()+".wav")
			saveAudioFile(pcm, filename, sampleRate, numChannels)
		},
		// The separate tracks.
		OnTrackAudioData: func(user, bot []byte, sampleRate, _ int) {
			if err := os.MkdirAll(recordingsDir, 0o750); err != nil {
				slog.Error("Error creating the recordings directory", "err", err)
				return
			}
			ts := timestamp()
			saveAudioFile(user, filepath.Join(recordingsDir, "user_"+ts+".wav"), sampleRate, 1)
			saveAudioFile(bot, filepath.Join(recordingsDir, "bot_"+ts+".wav"), sampleRate, 1)
		},
	})

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		tts,
		t.Output(),
		audioBuffer, // The audio buffer, recording
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
