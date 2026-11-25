// Command realtime-gemini-live-locally-driven-turns is a voice agent on Gemini
// Live, a speech-to-speech model, with turn detection driven locally.
//
// By default Gemini Live drives the conversation with its own server-side VAD
// (see realtime-gemini-live). That setup doesn't surface
// UserStartedSpeakingFrame and UserStoppedSpeakingFrame, so the processors
// that depend on them (RTVI client speech events, the turn tracking observer,
// turn recording in the audio buffer, the user idle controller, user mute
// strategies, the voicemail detector) don't activate.
//
// This variant turns Gemini Live's server-side VAD off and drives turn
// boundaries locally instead, with a Silero VAD in the user aggregator. Use it
// if you need those processors, or want a turn analyzer such as Smart Turn to
// decide when the user is done speaking.
//
// Locally decided turn boundaries are a heuristic and may not match the turns
// the server would decide, which are what really drive the conversation. The
// two can drift apart in subtle, hard-to-debug ways, especially around
// interruptions and overlapping speech. Prefer the server's turns (the
// realtime-gemini-live example) without a specific reason to drive them
// locally.
//
//	GOOGLE_API_KEY=… go run ./examples/realtime/realtime-gemini-live-locally-driven-turns
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/runner"
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	s2s := live.New(live.Config{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
		Voice:  "Aoede", // Puck, Charon, Kore, Fenrir, Aoede
		VAD:    &live.VADParams{Disabled: new(true)},
		// InferenceOnContextInitialization: new(false),
	})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleUser, Text: "Say hello. Then ask if I want to hear a joke."})

	// The stop window is longer than the 0.2s default on purpose: with the
	// server's detection off, padding the end of speech with a bit more silence
	// seems to do a bit better.
	vadParams := vad.DefaultParams()
	vadParams.StopSecs = 0.5
	silero, err := vad.NewSilero(vad.WithParams(vadParams))
	if err != nil {
		return err
	}
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		agg.User(),
		s2s,
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

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	// In realtime mode the user transcript isn't final when the user's turn
	// stops, so the turn-stopped event carries no text; the finalized text
	// arrives with the user turn message being added.
	event.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("User turn stopped at " + m.Timestamp)
		})
	// In realtime mode this is the canonical "user said X" event, decoupled
	// from the turn stopping.
	event.On(agg.User().Events(), aggregators.EventUserTurnMessageAdded,
		func(_ context.Context, m aggregators.UserTurnMessageAdded) {
			slog.Info("Transcript: " + stamp(m.Timestamp) + "user: " + m.Content)
		})
	event.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			slog.Info("Transcript: " + stamp(m.Timestamp) + "assistant: " + m.Content)
		})

	return r.Run(ctx, workers.RunOptions{})
}

// stamp formats a transcript line's timestamp, or nothing without one.
func stamp(timestamp string) string {
	if timestamp == "" {
		return ""
	}
	return fmt.Sprintf("[%s] ", timestamp)
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
