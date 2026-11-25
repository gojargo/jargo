// Command thinking-functions-google shows a model's thinking alongside function calling:
// Google Gemini, thinking at the "low" level with its thoughts included.
//
// The assistant aggregator raises EventAssistantThought with each completed
// thought, which this bot logs. The LLM can check a flight's status and book a
// taxi, which together make a request worth thinking about.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… GOOGLE_API_KEY=… \
//	  go run ./examples/thinking/thinking-functions-google
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// checkFlightStatus checks the status of a flight. It returns the status
// (e.g. "on time", "delayed") and the departure time.
func checkFlightStatus(ctx context.Context, params llm.FunctionCallParams) error {
	return params.Result(ctx, `{"status": "delayed", "departure_time": "14:30"}`, nil)
}

// bookTaxi books a taxi for a given time. It returns the status (e.g. "done").
func bookTaxi(ctx context.Context, params llm.FunctionCallParams) error {
	return params.Result(ctx, `{"status": "done"}`, nil)
}

// tools describes the functions to the LLM, each carrying its own handler.
func tools() []frames.Tool {
	return []frames.Tool{
		{
			Name: "check_flight_status",
			Description: "Check the status of a flight. Returns status (e.g., \"on time\", \"delayed\") " +
				"and departure time.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"flight_number": {
						"type": "string",
						"description": "The flight number, e.g. \"AA100\"."
					}
				},
				"required": ["flight_number"]
			}`),
			Handler: llm.FunctionCallHandler(checkFlightStatus),
		},
		{
			Name:        "book_taxi",
			Description: "Book a taxi for a given time. Returns status (e.g., \"done\").",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"time": {
						"type": "string",
						"description": "The time to book the taxi for, e.g. \"15:00\"."
					}
				},
				"required": ["time"]
			}`),
			Handler: llm.FunctionCallHandler(bookTaxi),
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := gemini.NewLLM(gemini.Config{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
		Thinking: &gemini.ThinkingConfig{
			// Want more reasoning? Raise the level.
			Level:           "low",
			IncludeThoughts: true,
		},
	})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

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
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Say hello briefly."})
		// Here is an example prompt conducive to demonstrating thinking and
		// function calling.
		// convo.AddMessage(frames.Message{
		// 	Role: frames.RoleUser,
		// 	Text: "Check the status of flight AA100 and, if it's delayed, book me a taxi 2 hours " +
		// 		"before its departure time.",
		// })
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	event.On(agg.Assistant().Events(), aggregators.EventAssistantThought,
		func(_ context.Context, m aggregators.AssistantThought) {
			slog.Info("Thought (timestamp: " + m.Timestamp + "): " + m.Content)
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
