// Command mcp-stdio is a voice agent whose tools come from an MCP server run
// over stdio: the memory server, which lets the bot store and recall things
// about the user.
//
// The server is started with npx, so Node.js must be installed. Speech-to-text
// is Deepgram, the LLM Anthropic's Claude and text-to-speech Cartesia.
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/mcp/mcp-stdio
//
// Then open http://localhost:7860, connect, and ask the bot to remember
// something.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/mcp"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	LiveKit: transport.DefaultParams,
	WebRTC:  transport.DefaultParams,
}

const systemPrompt = `
    You are a helpful LLM in a voice call.
    Your goal is to demonstrate your capabilities in a succinct way.
    You have access to memory tools that let you store and recall information.
    Offer to remember things for the user, like their name, preferences, or anything they'd like.
    You can also recall things you've previously stored.
    Your output will be spoken aloud, so avoid special characters that can't easily be spoken, ` +
	`such as emojis or bullet points.
    Respond to what the user said in a creative and helpful way.
    Don't overexplain what you are doing.
    Just respond with short sentences when you are carrying out tool calls.
    `

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	llm := anthropic.NewLLM(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
	llm.SetSystemInstruction(systemPrompt)

	// https://github.com/modelcontextprotocol/servers/tree/main/src/memory
	//
	// The server inherits this process's environment. Optional: set
	// MEMORY_FILE_PATH (e.g. /tmp/jargo_memory.jsonl) to choose where it stores
	// its memory.
	server, err := mcp.Connect(ctx, mcp.Config{
		Command: []string{"npx", "-y", "@modelcontextprotocol/server-memory"},
	})
	if err != nil {
		return err
	}
	tools, err := server.Tools(ctx)
	if err != nil {
		_ = server.Close()
		return err
	}

	silero, err := vad.NewSilero()
	if err != nil {
		_ = server.Close()
		return err
	}
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleUser, Text: "Please introduce yourself."})
	convo.SetTools(tools)
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,
		agg.User(),      // User spoken responses
		llm,             // LLM
		tts,             // TTS
		t.Output(),      // Transport bot output
		agg.Assistant(), // Assistant spoken responses and tool context
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

	events.On(t.Events(), transport.EventClientConnected, func(_ context.Context, client any) {
		slog.Info("Client connected", "client", client)
		// Kick off the conversation.
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
