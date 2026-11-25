// Command context-summarization-dedicated-llm customizes context
// summarization in depth.
//
// It shows:
//
//   - a dedicated cheap, fast LLM for generating summaries (Gemini Flash)
//
//   - a custom summary message template (XML tags)
//
//   - a custom summarization prompt
//
//   - a summarization timeout
//
//   - EventSummaryApplied, for observability
//
//     DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… GOOGLE_API_KEY=… \
//     go run ./examples/context-summarization/context-summarization-dedicated-llm
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	slog.Info("Tool called: get_current_weather")
	// Simulate some processing.
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(time.Second):
	}
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
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
	}
}

// customSummarizationPrompt is a summarization prompt tailored to the
// application.
const customSummarizationPrompt = `Summarize this conversation, preserving:
- Key decisions and agreements
- Important facts and user preferences
- Any pending action items or unresolved questions

Be concise. Use clear, factual statements grouped by topic.
Omit greetings, small talk, and resolved tangents.`

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

const systemPrompt = `You are a helpful LLM in a voice call. Your goal is to demonstrate your
                    capabilities in a succinct way. Your output will be spoken aloud, so avoid
                    special characters that can't easily be spoken, such as emojis or bullet points.
                    Respond to what the user said in a creative and helpful way.
                    You have access to tools to get the current weather - use them when relevant.
                    When you see a <context_summary> block, it contains a compressed summary
                    of earlier conversation. Use it as reference but don't mention it to the user.
                    `

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	// The primary LLM, for the conversation (it could be any provider).
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemPrompt)

	// A dedicated cheap, fast LLM, for summarization only.
	summarizationLLM := gemini.NewLLM(gemini.Config{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
		Model:  "gemini-3.6-flash",
	})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())

	// The aggregators, with custom summarization.
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		aggregators.WithSummarization(frames.AutoSummarizationConfig{
			// Trigger thresholds (low values, to demonstrate quickly).
			MaxContextTokens:        new(1000),
			MaxUnsummarizedMessages: new(10),
			SummaryConfig: frames.SummaryConfig{
				// Summary generation.
				TargetContextTokens:     800,
				MinMessagesAfterSummary: new(2),
				SummarizationPrompt:     customSummarizationPrompt,
				// A custom summary format: XML tags, so the system prompt can
				// tell summaries from live conversation.
				SummaryMessageTemplate: "<context_summary>\n{summary}\n</context_summary>",
				// A dedicated cheap LLM for summarization instead of the
				// primary conversation model.
				LLM: summarizationLLM,
				// Cancel summarization if it takes longer than 60 seconds.
				SummarizationTimeout: 60 * time.Second,
			},
		}),
	)

	// Listen for summarization events.
	event.On(&agg.Assistant().Summarizer().Registry, aggregators.EventSummaryApplied,
		func(_ context.Context, e aggregators.SummaryApplied) {
			slog.Info(fmt.Sprintf("Context summarized: %d messages -> %d messages (%d summarized, %d preserved)",
				e.OriginalMessageCount, e.NewMessageCount, e.SummarizedMessageCount, e.PreservedMessageCount))
		})

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
