// Command voice-deepgram-flux-eager is a voice agent: Deepgram Flux
// speech-to-text and text-to-speech, with an OpenAI LLM, answering Flux's
// eager end of turn over a turn held open across segments.
//
// Flux's predicted end of turn is answered speculatively, and the LLM's
// turn-completion markers can hold the user turn open after Flux commits a
// segment. A turn can then collect several Flux segments, each answered
// speculatively in turn.
//
// To exercise that, say an unfinished sentence, wait for Flux to commit it (the
// bot stays quiet when the LLM marks it incomplete), then finish it:
//
//	"I'd like to book a table for"  ...pause...  "four people at seven"
//
// Run it with:
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/voice/voice-deepgram-flux-eager
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewFluxSTT(deepgram.FluxConfig{
		APIKey: os.Getenv("DEEPGRAM_API_KEY"),
		// Answer Flux's predicted end of turn ahead of the committed one. Flux
		// reports no prediction without an eager threshold; 0.5 is the one to
		// use when there is no reason to tune it.
		EagerEOTThreshold: new(0.5),
		MinConfidence:     new(0.3),
	})

	tts := deepgram.NewFluxTTS(deepgram.FluxTTSConfig{
		APIKey: os.Getenv("DEEPGRAM_API_KEY"),
		Voice:  "flux-cole-en",
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		// The LLM's turn-completion gate joins the eager strategies Flux
		// recommends, so an incomplete segment keeps the turn open.
		aggregators.WithTurns(turns.Config{
			Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{}),
			StrategiesParams: turns.UserTurnStrategiesParams{
				StartMergeMode: turns.MergeModeExtend,
				StopMergeMode:  turns.MergeModeExtend,
			},
		}),
	)

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
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

	events.On(stt.Events(), deepgram.EventFluxEagerEndOfTurn, func(_ context.Context, transcript string) {
		slog.Info("Eager end of turn: [" + transcript + "]")
	})

	events.OnSignal(stt.Events(), deepgram.EventFluxTurnResumed, func(context.Context) {
		slog.Info("Turn resumed: the eager end of turn was withdrawn")
	})

	events.On(stt.Events(), deepgram.EventFluxEndOfTurn, func(_ context.Context, transcript string) {
		slog.Info("Flux end of turn: [" + transcript + "]")
	})

	events.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("Transcript: user: " + m.Content)
		})

	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			slog.Info("Transcript: assistant: " + m.Content)
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
