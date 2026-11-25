// Command hello_world is a "Hello world" introduction to conversation flows.
//
// The flow has two nodes. The first asks the user their favorite color and
// offers one function, which records the answer and moves to the second; the
// second thanks the user and ends the conversation with the end_conversation
// post-action.
//
//	CARTESIA_API_KEY=… GOOGLE_API_KEY=… \
//	  go run ./examples/flows/hello_world
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
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

// createInitialNode creates the initial node of the flow.
//
// It defines the bot's role and task for the node, plus the function it can
// call. The function records the result and transitions to the next node.
func createInitialNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "initial",
		RoleMessage: "You are an inquisitive child. Use very simple language. Ask simple questions. " +
			"You must ALWAYS use one of the available functions to progress the conversation. " +
			"Your responses will be converted to audio. Avoid outputting special characters and emojis.",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Say 'Hello world' and ask what is the user's favorite color. Wait for the user to " +
				"answer; call record_favorite_color only with the color they tell you.",
		}},
		Functions: []flows.NodeFunction{{
			Name: "record_favorite_color",
			Description: "Record the color the user said is their favorite.\n\n" +
				"Here \"record\" means print to the console, but any logic could go here:\n" +
				"write to a database, make an API call, etc.",
			Properties: map[string]any{
				"color": map[string]any{"type": "string", "description": "The user's favorite color."},
			},
			Required: []string{"color"},
			Handler:  recordFavoriteColor,
		}},
	}
}

// recordFavoriteColor records the color the user said is their favorite.
//
// Here "record" means print to the console, but any logic could go here:
// write to a database, make an API call, etc.
func recordFavoriteColor(
	_ context.Context, args json.RawMessage, _ *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Color string `json:"color"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	fmt.Println("Your favorite color is: " + in.Color)
	result, err := json.Marshal(in.Color)
	if err != nil {
		return "", nil, err
	}
	return string(result), createEndNode(), nil
}

// createEndNode ends the conversation.
//
// The flow transitions to this node when the user has answered the question.
// It thanks the user and ends the conversation using the end_conversation
// post-action.
func createEndNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "create_end_node",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Thank the user for answering and end the conversation",
		}},
		PostActions: []flows.ActionConfig{{Type: "end_conversation"}},
	}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	stt := cartesia.NewSTT(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "32b3f3c5-7171-46aa-abe7-b598964aa793",
	})
	model := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		aggregators.WithTurns(turns.Config{
			Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{}),
		}),
	)

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
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

	// Initialize the flow manager.
	flowManager, err := flows.New(flows.Config{
		Enqueuer:    worker,
		Watcher:     worker,
		Aggregators: agg,
		LLM:         model,
	})
	if err != nil {
		return err
	}

	event.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		if err := flowManager.Initialize(ctx, createInitialNode()); err != nil {
			slog.Error("initialize flow", "err", err)
		}
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
