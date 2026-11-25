// Command observability-observer watches the frames flowing through the
// pipeline with observers.
//
// A custom observer logs every interruption leaving the output transport and
// every bot started and stopped speaking frame, from processor to processor,
// with its timestamp:
//
//	[EVENT TYPE]: [source processor] → [destination processor] at [timestamp]s
//
// An LLMLog observer logs what the LLM is given and what it produces, and a
// DebugLog observer logs the TTS text leaving the output transport, the user
// started speaking frames leaving the input transport, and the end frame.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/observability/observability-observer
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
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
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

// customObserver logs interruptions and the bot speaking.
//
// It logs every InterruptionFrame leaving an output transport, and every
// BotStartedSpeakingFrame and BotStoppedSpeakingFrame, which shows how these
// frames flow from processor to processor through the pipeline.
type customObserver struct{}

// OnPushFrame implements processor.Observer.
func (customObserver) OnPushFrame(data processor.FramePushed) {
	arrow := "→"
	if data.Direction != processor.Downstream {
		arrow = "←"
	}
	route := fmt.Sprintf("%s %s %s at %.2fs", data.Source.Name(), arrow, data.Destination.Name(), data.Timestamp.Seconds())
	switch data.Frame.(type) {
	case *frames.InterruptionFrame:
		if _, ok := data.Source.(interface{ PlaysOutput() }); ok {
			slog.Info("INTERRUPTION START: " + route)
		}
	case *frames.BotStartedSpeakingFrame:
		slog.Info("BOT START SPEAKING: " + route)
	case *frames.BotStoppedSpeakingFrame:
		slog.Info("BOT STOP SPEAKING: " + route)
	}
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
		IdleTimeout: args.Base().PipelineIdleTimeout,
		Observers: []pipeline.Observer{
			customObserver{},
			observers.NewLLMLog(observers.LLMLogConfig{}),
			observers.NewDebugLog(observers.DebugLogConfig{
				Frames: []observers.DebugFrameFilter{
					{
						Frame:    frames.NewTTSTextFrame("", frames.TextTypeSentence),
						Match:    func(p processor.Processor) bool { return p == t.Output() },
						Endpoint: observers.SourceEndpoint,
					},
					{
						Frame:    frames.NewUserStartedSpeakingFrame(),
						Match:    func(p processor.Processor) bool { return p == t.Input() },
						Endpoint: observers.SourceEndpoint,
					},
					{Frame: frames.NewEndFrame()},
				},
			}),
		},
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
