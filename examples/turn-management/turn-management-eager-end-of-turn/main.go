// Command turn-management-eager-end-of-turn answers a predicted end of turn
// while the user turn is still open, so the gap before the service commits to
// it is spent generating a response instead of waiting for one.
//
// Deepgram Flux predicts an end of turn (EagerEndOfTurn) ahead of committing to
// one (EndOfTurn), and withdraws the prediction (TurnResumed) if the user turns
// out to be mid-sentence. Enabling eager end of turn (an eager threshold on the
// Flux config) answers the prediction and holds the response in the LLM
// service until the turn is confirmed, discarding it if the user resumes
// speaking or the committed transcript differs from the predicted one. Nothing
// unconfirmed is spoken or written to the context.
//
// The pipeline is an ordinary one: the threshold is the whole configuration.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-eager-end-of-turn
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

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

// logTranscript logs each user and assistant turn as it ends.
func logTranscript(agg *aggregators.Pair) {
	events.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("Transcript: user: " + m.Content)
		})
	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			slog.Info("Transcript: assistant: " + m.Content)
		})
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewFluxSTT(deepgram.FluxConfig{
		APIKey: os.Getenv("DEEPGRAM_API_KEY"),
		// Answer Flux's predicted end of turn. Flux reports no prediction
		// without an eager threshold, and 0.5 is the one eager end of turn
		// uses when nothing tunes it.
		EagerEOTThreshold: new(0.5),
	})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	// The STT service recommends the turn strategies that answer its
	// predictions, so nothing here has to name them.
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		tts,
		t.Output(),
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
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user, then ask them where they would " +
				"travel if they could go anywhere right now, and why.",
		})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	events.On(stt.Events(), deepgram.EventFluxEagerEndOfTurn, func(_ context.Context, transcript string) {
		slog.Info("Eager end of turn: " + transcript)
	})

	events.OnSignal(stt.Events(), deepgram.EventFluxTurnResumed, func(context.Context) {
		slog.Info("Turn resumed: the eager end of turn was withdrawn")
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
