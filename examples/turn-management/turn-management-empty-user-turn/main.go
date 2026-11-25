// Command turn-management-empty-user-turn recovers from a user turn that ends
// with no transcript.
//
// A user turn can start on voice activity alone and end with no transcript:
// the user coughed, spoke over background noise, or the STT simply missed it.
// When that turn interrupted the bot, the bot has stopped talking and the LLM
// has nothing new to answer, so the conversation stalls.
//
// By default, the user aggregator runs the LLM once for such a turn, with a
// developer message saying that the user spoke but wasn't understood. The bot
// then asks the user to repeat, or repeats what they may have missed. An empty
// turn while the bot was idle is left unanswered by default: the bot is
// already waiting for the user, and answering what may be noise would be
// intrusive. The user idle timeout still applies after it, so the bot checks
// in if the user then stays quiet. turns.EmptyUserTurnConfig customizes this
// behavior.
//
// To try it without a noisy room, say "pineapple": this bot drops any
// transcript containing that word, as if the STT had failed to recognize it.
// Ask for a long story, then say "pineapple" while the bot is talking.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-empty-user-turn
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// unrecognizedWord is the word whose transcripts are dropped.
const unrecognizedWord = "pineapple"

// unrecognizedSpeechSimulator drops the transcripts containing
// unrecognizedWord.
type unrecognizedSpeechSimulator struct {
	*processor.Base
}

func newUnrecognizedSpeechSimulator() *unrecognizedSpeechSimulator {
	p := &unrecognizedSpeechSimulator{}
	p.Base = processor.New("UnrecognizedSpeechSimulator", p)
	return p
}

// ProcessFrame drops the transcripts holding the word and pushes the rest on.
func (p *unrecognizedSpeechSimulator) ProcessFrame(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	var text string
	switch t := f.(type) {
	case *frames.TranscriptionFrame:
		text = t.Text
	case *frames.InterimTranscriptionFrame:
		text = t.Text
	}
	if strings.Contains(strings.ToLower(text), unrecognizedWord) {
		slog.Info("Dropping transcript to simulate unrecognized speech: " + text)
		return nil
	}
	return p.PushFrame(ctx, f, dir)
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

// logTranscript logs each user and assistant turn as it ends.
func logTranscript(agg *aggregators.Pair) {
	event.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("Transcript: " + timestamped(m.Timestamp, "user: "+m.Content))
		})
	event.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			slog.Info("Transcript: " + timestamped(m.Timestamp, "assistant: "+m.Content))
		})
}

// timestamped prefixes line with the timestamp, when there is one.
func timestamped(timestamp, line string) string {
	if timestamp == "" {
		return line
	}
	return "[" + timestamp + "] " + line
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	convo := frames.NewLLMContext("")

	// An empty turn is closed by the user turn stop timeout (5s by default),
	// so the recovery comes that long after the user stops talking.
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		aggregators.WithTurns(turns.Config{
			IdleTimeout: 8 * time.Second,
			// Optional: the default config is used when this is left out.
			EmptyUserTurn: &turns.EmptyUserTurnConfig{
				// Optional: customize the recovery.
				// InterruptedPrompt: new("Custom prompt..."),
				// IdlePrompt: "The user may have said something that wasn't understood...",
				// MaxConsecutiveRecoveries: new(2),
			},
		}),
	)

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,
		newUnrecognizedSpeechSimulator(), // Drop unrecognized speech, only for demo
		agg.User(),                       // User responses
		llm,                              // LLM
		tts,                              // TTS
		t.Output(),                       // Transport bot output
		agg.Assistant(),                  // Assistant spoken responses
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
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	event.OnSignal(agg.User().Events(), aggregators.EventUserTurnIdle, func(ctx context.Context) {
		slog.Info("User turn idle")
		// The user aggregator consumes LLMMessagesAppendFrame, so push from it
		// to reach the LLM service downstream.
		appendFrame := frames.NewLLMMessagesAppendFrame([]frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "The user has been quiet. Politely and briefly ask if they're still there.",
		}})
		appendFrame.RunLLM = true
		_ = agg.User().PushFrame(ctx, appendFrame, processor.Downstream)
	})

	logTranscript(agg)

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
