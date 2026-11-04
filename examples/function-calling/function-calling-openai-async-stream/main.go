// Command function-calling-openai-async-stream is a voice agent whose LLM
// calls tools: OpenAI, with a tool that reports intermediate results.
//
// The track_current_location tool simulates a GPS tracker reporting the
// device's position during a road trip from San Francisco to San Diego. It
// sends two intermediate updates (results marked not final) as the vehicle
// passes through cities along the way, then delivers the final destination.
// Because the tool is not canceled when the user interrupts, the LLM can keep
// talking while the trip is in progress; each position update reaches it as a
// developer message, so it can narrate the journey to the user. Speech-to-text
// is Deepgram and text-to-speech Cartesia.
//
//	DEEPGRAM_API_KEY=… OPENAI_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/function-calling/function-calling-openai-async-stream
//
// Then open http://localhost:7860, connect, and ask it to track your location.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// sleep waits for d, or until the call is canceled.
func sleep(ctx context.Context, d time.Duration) error {
	select {
	case <-time.After(d):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// update marks a result as an intermediate update rather than the final one.
func update() *frames.FunctionCallResultProperties {
	return &frames.FunctionCallResultProperties{IsFinal: new(false)}
}

// trackCurrentLocation simulates a GPS tracker reporting position during a road
// trip:
//
//  1. San Francisco (trip start): update
//  2. Los Angeles (passing through): update
//  3. San Diego (destination): final result
func trackCurrentLocation(ctx context.Context, params llm.FunctionCallParams) error {
	// First update: initial city estimate.
	if err := params.Result(ctx,
		`{"gps": {"lat": 37.7310, "lng": -122.4527}, "city": "San Francisco"}`, update()); err != nil {
		return err
	}

	// Second update: revised city estimate.
	if err := sleep(ctx, 10*time.Second); err != nil {
		return err
	}
	if err := params.Result(ctx,
		`{"gps": {"lat": 33.96003, "lng": -118.40639}, "city": "Los Angeles"}`, update()); err != nil {
		return err
	}

	// Final result: confirmed city.
	if err := sleep(ctx, 10*time.Second); err != nil {
		return err
	}
	return params.Result(ctx, `{"gps": {"lat": 32.743569, "lng": -117.20466}, "city": "San Diego"}`, nil)
}

// tools describes the functions to the LLM, each carrying its own handler.
func tools() []frames.Tool {
	return []frames.Tool{
		{
			Name: "track_current_location",
			Description: "Start tracking the user's current GPS location, reporting position updates " +
				"until the user reaches their destination.",
			Parameters:           json.RawMessage(`{"type": "object", "properties": {}}`),
			Handler:              llm.FunctionCallHandler(trackCurrentLocation),
			CancelOnInterruption: new(false),
			TimeoutSecs:          new(30.0),
		},
	}
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. You have access to a function that starts tracking" +
	" the user's location and provides regular updates on it. When you " +
	"receive the final location, tell the user the destination has been " +
	"reached."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	model.OnFunctionCallsStarted(func(ctx context.Context, _ []frames.ToolCall) {
		_ = tts.QueueFrame(ctx, frames.NewTTSSpeakFrame("Sure, tracking your location now."), processor.Downstream)
	})

	model.OnFunctionCallsCanceled(func(_ context.Context, calls []frames.ToolCall) {
		for _, call := range calls {
			slog.Info("Function call canceled", "function", call.Name, "tool_call_id", call.ID)
		}
	})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		model,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
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
