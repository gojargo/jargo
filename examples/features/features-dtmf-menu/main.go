// Command features-dtmf-menu is a keypad-driven phone menu (IVR) bot.
//
// The caller interacts entirely with DTMF keypresses, no speech. A DTMF
// aggregator turns each key sequence into a "DTMF: ..." transcription that the
// LLM reacts to:
//
//   - press 1 for business hours
//   - press 2 for the office location
//   - press 3 to request a callback: the caller enters their callback number
//     and presses # to finalize it (the aggregator's terminator key, which
//     flushes the sequence immediately instead of waiting for the idle timeout)
//
// DTMF arrives as InputDTMFFrame from the transport (e.g. a telephony
// provider), so there's no STT in the pipeline. The bot speaks its responses
// with TTS, like a real phone menu would.
//
//	OPENAI_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/features/features-dtmf-menu -t twilio -x your_username.ngrok.io
//
// Then call the Twilio number and use the keypad.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/dtmf"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

const systemInstruction = `You are an automated phone menu for Acme Corp.

The caller interacts only with their phone keypad. Each keypress arrives as a ` +
	`message like "DTMF: 1" (the digits they pressed, possibly ending in #).

When the call starts, greet the caller and read this menu: press 1 for our ` +
	`business hours, press 2 for our location, press 3 to request a callback.

When the caller presses 1, tell them Acme Corp is open from 9 AM to 5 PM, Monday ` +
	`through Friday. When they press 2, tell them Acme Corp is located at 123 Main ` +
	`Street. When they press 3, ask them to enter their callback number followed by ` +
	`the pound key. For any other key, say that's not a valid option and read the ` +
	`menu again.

While collecting a callback number, the digits may arrive split across several ` +
	`messages if the caller pauses; the number is complete only when a message ends ` +
	`in #. Once you have the complete number, read it back digit by digit, tell the ` +
	`caller someone will call them back, and read the menu again.

Keep every response short. Your responses may be read aloud, so don't use emojis ` +
	`or formatting.`

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

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(), // Transport user input (incl. DTMF keypresses)
		dtmf.NewAggregator(dtmf.AggregatorConfig{}), // InputDTMFFrame -> "DTMF: ..." TranscriptionFrame
		agg.User(),      // User (keypad) turns
		llm,             // LLM
		tts,             // TTS
		t.Output(),      // Transport bot output
		agg.Assistant(), // Assistant responses
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
		// Kick off the call with the menu greeting.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Greet the caller and read the menu."})
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
