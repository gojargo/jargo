// Command mcp-multiple-mcp is a voice agent whose tools come from two MCP
// servers at once: the memory server, run over stdio, which lets the bot store
// and recall things about the user, and GitHub's, reached over streamable HTTP
// with a personal access token, which answers questions about the user's
// repositories and account.
//
// The memory server is started with npx, so Node.js must be installed.
// Speech-to-text is Deepgram, the LLM Anthropic's Claude and text-to-speech
// Cartesia.
//
//	GITHUB_PERSONAL_ACCESS_TOKEN=… DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/mcp/mcp-multiple-mcp
//
// Then open http://localhost:7860, connect, and ask the bot to remember
// something or about your repositories.
package main

import (
	"context"
	"log/slog"
	"os"
	"slices"

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
	"github.com/joho/godotenv"
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
    You have access to memory tools that let you store and recall information,
    and tools to answer questions about the user's GitHub repositories and account.
    Offer to remember things for the user, like their name, preferences, or anything they'd like.
    You can also recall things you've previously stored.
    You can also offer to answer users questions about their GitHub repositories and account.
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
	memoryMCP, err := mcp.Connect(ctx, mcp.Config{
		Command: []string{"npx", "-y", "@modelcontextprotocol/server-memory"},
		// Env: map[string]string{"MEMORY_FILE_PATH": "/tmp/jargo_memory.jsonl"}, // Optional: specify MEMORY_FILE_PATH
	})
	if err != nil {
		return err
	}
	// Github MCP docs: https://github.com/github/github-mcp-server
	// Enable Github Copilot on your GitHub account. Free tier is ok. (https://github.com/settings/copilot)
	// Generate a personal access token. It must be a Fine-grained token, classic tokens are not
	// supported. (https://github.com/settings/personal-access-tokens)
	// Set permissions you want to use (eg. "all repositories", "profile: read/write", etc)
	githubMCP, err := mcp.Connect(ctx, mcp.Config{
		HTTPURL: "https://api.githubcopilot.com/mcp/",
		Headers: map[string]string{"Authorization": "Bearer " + os.Getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
	})
	if err != nil {
		_ = memoryMCP.Close()
		return err
	}
	closeAll := func() {
		_ = memoryMCP.Close()
		_ = githubMCP.Close()
	}

	memoryTools, err := memoryMCP.Tools(ctx)
	if err != nil {
		closeAll()
		return err
	}
	githubTools, err := githubMCP.Tools(ctx)
	if err != nil {
		closeAll()
		return err
	}

	allTools := slices.Concat(memoryTools, githubTools)

	silero, err := vad.NewSilero()
	if err != nil {
		closeAll()
		return err
	}
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleUser, Text: "Please introduce yourself."})
	convo.SetTools(allTools)
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
	_ = godotenv.Overload()
	if os.Getenv("GITHUB_PERSONAL_ACCESS_TOKEN") == "" {
		slog.Error("Please set `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable.")
		os.Exit(1)
	}
	runner.Main(bot)
}
