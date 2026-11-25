// Command turn-management-user-assistant-turns keeps a transcript of the
// conversation from the user and assistant turn events.
//
// The user aggregator raises EventUserTurnStopped and the assistant aggregator
// EventAssistantTurnStopped as each turn ends, each with what was said and
// when. A transcriptHandler logs every message with its timestamp and role,
// and can append it to a file too.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-user-assistant-turns
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"
	"sync"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
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

// transcriptHandler outputs the transcript as it is received, to the log and
// optionally to a file. Each message carries its timestamp and role.
type transcriptHandler struct {
	// outputFile is the file the transcript is appended to; empty logs only.
	outputFile string
	mu         sync.Mutex
}

// newTranscriptHandler builds a handler appending to outputFile, or logging
// only when it is empty.
func newTranscriptHandler(outputFile string) *transcriptHandler {
	if outputFile != "" {
		slog.Debug("TranscriptHandler initialized with output_file=" + outputFile)
	} else {
		slog.Debug("TranscriptHandler initialized with log output only")
	}
	return &transcriptHandler{outputFile: outputFile}
}

// saveMessage outputs one transcript message to the log and, optionally, to
// the file.
func (h *transcriptHandler) saveMessage(role, content, timestamp string) {
	line := "[" + timestamp + "] " + role + ": " + content

	// Always log the message.
	slog.Info("Transcript: " + line)

	// Optionally write it to the file.
	if h.outputFile == "" {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	f, err := os.OpenFile(h.outputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		slog.Error("Error saving transcript message to file", "err", err)
		return
	}
	defer func() { _ = f.Close() }()
	if _, err := f.WriteString(line + "\n\n"); err != nil {
		slog.Error("Error saving transcript message to file", "err", err)
	}
}

// onUserTranscript handles a new user message.
func (h *transcriptHandler) onUserTranscript(m aggregators.UserTurnStopped) {
	slog.Debug("Received user transcript update")
	h.saveMessage("user", m.Content, m.Timestamp)
}

// onAssistantTranscript handles a new assistant message.
func (h *transcriptHandler) onAssistantTranscript(m aggregators.AssistantTurnStopped) {
	slog.Debug("Received assistant transcript update")
	h.saveMessage("assistant", m.Content, m.Timestamp)
}

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

	// The transcript handler.
	transcript := newTranscriptHandler("") // Output to log only
	// transcript := newTranscriptHandler("transcript.txt") // Output to file and log

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
		// Start the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	event.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) { transcript.onUserTranscript(m) })

	event.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) { transcript.onAssistantTranscript(m) })

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
