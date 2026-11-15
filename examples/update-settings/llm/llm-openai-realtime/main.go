// Command llm-openai-realtime changes the settings of an OpenAI Realtime
// speech-to-speech model while the conversation runs: ten seconds after the
// client connects, it switches the session to text output, and ten seconds
// later back to audio.
//
//	OPENAI_API_KEY=… \
//	  go run ./examples/update-settings/llm/llm-openai-realtime
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/openai/realtime"
	rtevents "github.com/gojargo/jargo/provider/openai/realtime/events"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// stamp formats a transcript line's timestamp, or nothing without one.
func stamp(timestamp string) string {
	if timestamp == "" {
		return ""
	}
	return fmt.Sprintf("[%s] ", timestamp)
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

// updateAfter is how long the bot waits before each settings change.
const updateAfter = 10 * time.Second

// sleep waits for updateAfter, and reports false when ctx ends first.
func sleep(ctx context.Context) bool {
	select {
	case <-time.After(updateAfter):
		return true
	case <-ctx.Done():
		return false
	}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	llm := realtime.New(realtime.Config{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Instructions: systemInstruction,
	})

	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),       // Transport user input
		agg.User(),      // User responses
		llm,             // LLM
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

	// The service emits user-turn frames from server VAD, so
	// EventUserTurnStopped fires at the turn boundary. In realtime mode the
	// stopped turn carries no content (the user transcript isn't finalized at
	// turn-stop time); subscribe to EventUserTurnMessageAdded if you need the
	// finalized user text.
	events.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, m aggregators.UserTurnStopped) {
			slog.Info("User turn stopped at " + m.Timestamp)
		})
	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, m aggregators.AssistantTurnStopped) {
			slog.Info("Transcript: " + stamp(m.Timestamp) + "assistant: " + m.Content)
		})

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating OpenAI Realtime LLM settings: output_modalities=['text']")
			worker.QueueFrame(frames.NewLLMUpdateSettingsFrame(&realtime.Settings{
				SessionProperties: settings.Set(rtevents.SessionProperties{
					OutputModalities: []rtevents.Modality{rtevents.ModalityText},
				}),
			}))

			if !sleep(ctx) {
				return
			}
			slog.Info("Updating OpenAI Realtime LLM settings: output_modalities=['audio']")
			worker.QueueFrame(frames.NewLLMUpdateSettingsFrame(&realtime.Settings{
				SessionProperties: settings.Set(rtevents.SessionProperties{
					OutputModalities: []rtevents.Modality{rtevents.ModalityAudio},
				}),
			}))
		}()
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
