// Command turn-management-detect-user-waiting notices when the user waits
// quietly on background work, and has the bot say it is still working.
//
// The one tool is an asynchronous function call that takes 40 seconds. While
// it runs the conversation carries on, and the call counts as background work.
// The user aggregator raises EventUserWaiting after ten seconds of quiet with
// background work in progress, and the bot then tells the user it is still
// working on their request.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-detect-user-waiting
//
// Then open http://localhost:7860, connect, ask for a restaurant, and wait.
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
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
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

// WaitReassuranceLLMExtension (processor/turns) does this for you, with a
// prompt that also lets the LLM stay quiet when it has nothing worth saying.
// This example shows how to handle the wait yourself.

// handleWait has the bot tell the waiting user that it is still working.
func handleWait(ctx context.Context, aggregator processor.Processor) {
	message := frames.Message{
		Role: frames.RoleDeveloper,
		Text: "The user has been waiting quietly while you work on their request. " +
			"Briefly let them know you're still working on it.",
	}
	appendFrame := frames.NewLLMMessagesAppendFrame([]frames.Message{message})
	appendFrame.RunLLM = true
	_ = aggregator.PushFrame(ctx, appendFrame, processor.Downstream)
}

// sleep waits for d, or until ctx is done.
func sleep(ctx context.Context, d time.Duration) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(d):
		return nil
	}
}

// getRestaurantRecommendation gets a restaurant recommendation.
func getRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	// Simulate a slow lookup, several times longer than the user waiting
	// timeout.
	if err := sleep(ctx, 40*time.Second); err != nil {
		return err
	}
	return params.Result(ctx, `{"name": "The Golden Dragon"}`, nil)
}

// tools describes the functions to the LLM, each carrying its own handler.
//
// CancelOnInterruption false makes the tool an asynchronous function call: the
// conversation carries on while it runs, and it counts as background work. A
// user who stays quiet during background work is waiting, which is what
// EventUserWaiting reports. A function call without it is part of the bot's
// turn, and no waiting is reported while it runs.
func tools() []frames.Tool {
	return []frames.Tool{
		{
			Name:        "get_restaurant_recommendation",
			Description: "Get a restaurant recommendation.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. \"San Francisco, CA\"."
					}
				},
				"required": ["location"]
			}`),
			Handler:              llm.FunctionCallHandler(getRestaurantRecommendation),
			CancelOnInterruption: new(false),
		},
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

// userWaitingTimeout is how long the user may wait quietly on background work
// before the bot speaks up.
const userWaitingTimeout = 10 * time.Second

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
		_ = tts.QueueFrame(ctx, frames.NewTTSSpeakFrame("Let me check on that."), processor.Downstream)
	})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo,
		aggregators.WithTurns(turns.Config{WaitingTimeout: userWaitingTimeout}), // Detect user waiting after 10 seconds
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
	)

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,
		agg.User(), // User aggregator with built-in waiting detection
		model,      // LLM
		tts,        // TTS
		t.Output(), // Transport bot output
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

	events.OnSignal(agg.User().Events(), aggregators.EventUserWaiting, func(ctx context.Context) {
		slog.Info("User waiting")
		handleWait(ctx, agg.User())
	})

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
