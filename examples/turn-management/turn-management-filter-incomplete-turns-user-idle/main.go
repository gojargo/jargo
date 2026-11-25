// Command turn-management-filter-incomplete-turns-user-idle combines LLM-based
// turn completion detection with user idle detection.
//
// Turn completion detection suppresses the bot's responses when the user was
// cut off mid-thought. The LLM outputs one of three markers:
//
//   - ● (complete): the user finished their thought, respond normally
//   - ◐ (incomplete short): the user was cut off, wait about 5s then prompt
//   - ○ (incomplete long): the user needs time to think, wait about 10s then
//     prompt
//
// Idle detection covers the other kind of silence: the user said nothing at
// all since the bot stopped speaking. After the idle timeout,
// EventUserTurnIdle fires and the bot appends a developer message asking the
// LLM to check in, with RunLLM to run inference right away. That response
// carries a marker like any other, and is spoken when it is ●.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-filter-incomplete-turns-user-idle
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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
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

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	convo := frames.NewLLMContext("")

	// FilterIncompleteUserTurnStrategies pairs the default detector chain with
	// an LLM turn completion stop strategy: the detectors trigger LLM
	// inference, but EventUserTurnStopped fires only when the LLM confirms ●.
	// The LLM marks each response with one of:
	// ● = complete (respond normally)
	// ◐ = incomplete short (wait 5s, then prompt)
	// ○ = incomplete long (wait 10s, then prompt)
	//
	// The idle timeout is measured from the moment the bot stops speaking, so
	// it should leave more room than the ◐/○ timeouts above: those cover a
	// user who is mid-turn, this one a user who never started.
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		aggregators.WithTurns(turns.Config{
			Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{
				// Optional: customize the turn completion behavior.
				// IncompleteShortTimeout: 5 * time.Second,
				// IncompleteLongTimeout:  10 * time.Second,
				// IncompleteShortPrompt:  "Custom prompt...",
				// IncompleteLongPrompt:   "Custom prompt...",
				// Instructions:           "Custom turn completion instructions...",
			}),
			IdleTimeout: 8 * time.Second,
		}),
	)

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,
		agg.User(),      // User responses
		model,           // LLM
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
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user, asking them a question that will require a " +
				"complete response. To start, say 'Let me start with a fun one. If you could travel " +
				"anywhere in the world right now, where would you go?'",
		})
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
