// Command mcp-streamable-http is a voice agent whose tools come from an MCP
// server reached over streamable HTTP: GitHub's, authenticated with a personal
// access token sent in a header, which lets the bot answer questions about the
// user's GitHub repositories and account.
//
// Speech-to-text is Deepgram, the LLM Google's Gemini and text-to-speech
// Cartesia.
//
//	GITHUB_PERSONAL_ACCESS_TOKEN=… DEEPGRAM_API_KEY=… GOOGLE_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/mcp/mcp-streamable-http
//
// Then open http://localhost:7860, connect, and ask about your repositories.
package main

import (
	"context"
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
	"github.com/gojargo/jargo/service/mcp"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
	"github.com/joho/godotenv"
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

const systemPrompt = `You are a helpful LLM in a voice call.
Your goal is to answer questions about the user's GitHub repositories and account.
You have access to a number of tools provided by Github. Use any and all tools to help users.
Your output will be spoken aloud, so avoid special characters that can't easily be spoken, ` +
	`such as emojis or bullet points.
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

	llm := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
	llm.SetSystemInstruction(systemPrompt)

	// Github MCP docs: https://github.com/github/github-mcp-server
	// Enable Github Copilot on your GitHub account. Free tier is ok. (https://github.com/settings/copilot)
	// Generate a personal access token. It must be a Fine-grained token, classic tokens are not
	// supported. (https://github.com/settings/personal-access-tokens)
	// Set permissions you want to use (eg. "all repositories", "profile: read/write", etc)
	server, err := mcp.Connect(ctx, mcp.Config{
		HTTPURL: "https://api.githubcopilot.com/mcp/",
		Headers: map[string]string{"Authorization": "Bearer " + os.Getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
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

	event.On(t.Events(), transport.EventClientConnected, func(_ context.Context, client any) {
		slog.Info("Client connected", "client", client)
		// Kick off the conversation.
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
	_ = godotenv.Overload()
	if os.Getenv("GITHUB_PERSONAL_ACCESS_TOKEN") == "" {
		slog.Error("Please set GITHUB_PERSONAL_ACCESS_TOKEN environment variable for this example.")
		os.Exit(1)
	}
	runner.Main(bot)
}
