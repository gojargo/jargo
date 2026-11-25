// Command turn-management-detect-user-idle notices when the user goes quiet
// and escalates, then ends the conversation.
//
// The user aggregator raises EventUserTurnIdle after five seconds without the
// user speaking once the bot has finished. The first time, the bot asks
// whether the user is still there; the second time, whether they would like to
// continue; the third time, it says goodbye and ends. The count resets when
// the user speaks again. Idle detection is turned off thirty seconds after the
// client connects and back on thirty seconds later, with a
// UserIdleTimeoutUpdateFrame. The tools are slow on purpose, slower than the
// idle timeout, to show that a running tool call does not count as idle.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/turn-management/turn-management-detect-user-idle
//
// Then open http://localhost:7860, connect, and stay quiet.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"sync"
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

// idleHandler manages the user idle retries.
type idleHandler struct {
	mu         sync.Mutex
	retryCount int
}

// reset resets the retry count when the user becomes active.
func (h *idleHandler) reset() {
	h.mu.Lock()
	h.retryCount = 0
	h.mu.Unlock()
}

// handleIdle handles a user idle event with escalating prompts.
func (h *idleHandler) handleIdle(ctx context.Context, aggregator processor.Processor) {
	h.mu.Lock()
	h.retryCount++
	retry := h.retryCount
	h.mu.Unlock()

	switch retry {
	case 1:
		// First attempt: add a gentle prompt to the conversation.
		message := frames.Message{
			Role: frames.RoleDeveloper,
			Text: "The user has been quiet. Politely and briefly ask if they're still there.",
		}
		appendFrame := frames.NewLLMMessagesAppendFrame([]frames.Message{message})
		appendFrame.RunLLM = true
		_ = aggregator.PushFrame(ctx, appendFrame, processor.Downstream)
	case 2:
		// Second attempt: a more direct prompt.
		message := frames.Message{
			Role: frames.RoleDeveloper,
			Text: "The user is still inactive. Ask if they'd like to continue our conversation.",
		}
		appendFrame := frames.NewLLMMessagesAppendFrame([]frames.Message{message})
		appendFrame.RunLLM = true
		_ = aggregator.PushFrame(ctx, appendFrame, processor.Downstream)
	default:
		// Third attempt: end the conversation.
		_ = aggregator.PushFrame(ctx,
			frames.NewTTSSpeakFrame("It seems like you're busy right now. Have a nice day!"), processor.Downstream)
		_ = aggregator.PushFrame(ctx, frames.NewEndWorkerFrame(), processor.Downstream)
	}
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

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	// Simulate a slow API call, waiting longer than the user idle timeout.
	if err := sleep(ctx, 3*time.Second); err != nil {
		return err
	}
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
}

// getRestaurantRecommendation gets a restaurant recommendation.
func getRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	if err := sleep(ctx, 6*time.Second); err != nil {
		return err
	}
	return params.Result(ctx, `{"name": "The Golden Dragon"}`, nil)
}

// tools describes the functions to the LLM, each carrying its own handler.
func tools() []frames.Tool {
	return []frames.Tool{
		{
			Name:        "get_current_weather",
			Description: "Get the current weather.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. \"San Francisco, CA\"."
					},
					"format": {
						"type": "string",
						"description": "The temperature unit to use. Must be either \"celsius\" or \"fahrenheit\". ` +
				`Infer this from the user's location."
					}
				},
				"required": ["location", "format"]
			}`),
			Handler: llm.FunctionCallHandler(getCurrentWeather),
		},
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
			Handler: llm.FunctionCallHandler(getRestaurantRecommendation),
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

// The idle timeout, and how long the bot waits before turning idle detection
// off and on again.
const (
	userIdleTimeout = 5 * time.Second
	toggleDelay     = 30 * time.Second
)

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
		aggregators.WithTurns(turns.Config{IdleTimeout: userIdleTimeout}), // Detect user idle after 5 seconds
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
	)

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,
		agg.User(), // User aggregator with built-in idle detection
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

	// Idle handling with retries.
	idle := &idleHandler{}

	event.OnSignal(agg.User().Events(), aggregators.EventUserTurnIdle, func(ctx context.Context) {
		slog.Info("User turn idle")
		idle.handleIdle(ctx, agg.User())
	})

	event.On(agg.User().Events(), aggregators.EventUserTurnStarted, func(context.Context, turns.StartStrategy) {
		idle.reset()
	})

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			if sleep(ctx, toggleDelay) != nil {
				return
			}
			slog.Info("Disabling idle detection")
			worker.QueueFrames([]frames.Frame{frames.NewUserIdleTimeoutUpdateFrame(0)})
			if sleep(ctx, toggleDelay) != nil {
				return
			}
			slog.Info("Enabling idle detection")
			worker.QueueFrames([]frames.Frame{frames.NewUserIdleTimeoutUpdateFrame(userIdleTimeout)})
		}()
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
