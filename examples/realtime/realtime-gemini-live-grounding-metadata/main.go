// Command realtime-gemini-live-grounding-metadata is a voice agent on Gemini
// Live, a speech-to-speech model, that answers from Google Search and logs the
// grounding metadata the model reports with each searched answer: the text the
// search supports, the rendered search suggestions, and the pages it drew on.
//
//	GOOGLE_API_KEY=… go run ./examples/realtime/realtime-gemini-live-grounding-metadata
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	googleframes "github.com/gojargo/jargo/provider/google/grounding"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/runner"
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

const systemInstruction = `
You are a helpful AI assistant that actively uses Google Search to provide up-to-date, accurate information.

IMPORTANT: For ANY question about current events, news, recent developments, real-time information, ` +
	`or anything that might have changed recently, you MUST use the google_search tool to get the latest information.

You should use Google Search for:
- Current news and events
- Recent developments in any field
- Today's weather, stock prices, or other real-time data
- Any question that starts with "what's happening", "latest", "recent", "current", "today", etc.
- When you're not certain about recent information

Always be proactive about using search when the user asks about anything that could benefit from real-time information.

Your output will be spoken aloud, so avoid special characters that can't easily be spoken, ` +
	`such as emojis or bullet points.

Respond to what the user said in a creative and helpful way, always using search for current information.
`

// tools offers Google Search and code execution, in Gemini's own format. No
// standard function declarations are needed.
func tools() frames.ToolsSchema {
	return frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeGemini: {
				map[string]any{"google_search": map[string]any{}},
				map[string]any{"code_execution": map[string]any{}},
			},
		},
	}
}

// groundingMetadataProcessor logs the grounding metadata Gemini Live reports.
type groundingMetadataProcessor struct {
	*processor.Base
	groundingCount int
}

func newGroundingMetadataProcessor() *groundingMetadataProcessor {
	g := &groundingMetadataProcessor{}
	g.Base = processor.New("GroundingMetadataProcessor", g)
	return g
}

// ProcessFrame logs each search response, and passes every frame on.
func (g *groundingMetadataProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := g.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if fr, ok := f.(*googleframes.LLMSearchResponseFrame); ok {
		g.groundingCount++
		logGrounding(g.groundingCount, fr)
	}
	// Always push the frame downstream.
	return g.PushFrame(ctx, f, dir)
}

// logGrounding logs one search response, the count-th received.
func logGrounding(count int, fr *googleframes.LLMSearchResponseFrame) {
	slog.Info(fmt.Sprintf("\n\nGROUNDING METADATA RECEIVED #%d\n", count))
	if fr.SearchResult != "" {
		slog.Info(fmt.Sprintf("Search Result Text: %s...", truncate(fr.SearchResult, 200)))
	}
	if fr.RenderedContent != "" {
		slog.Info("Rendered Content: " + fr.RenderedContent)
	}
	if len(fr.Origins) == 0 {
		return
	}
	slog.Info(fmt.Sprintf("Number of Origins: %d", len(fr.Origins)))
	for i, origin := range fr.Origins {
		slog.Info(fmt.Sprintf("  Origin %d: %s - %s", i+1, origin.SiteTitle, origin.SiteURI))
		if len(origin.Results) > 0 {
			slog.Info(fmt.Sprintf("    Results: %d items", len(origin.Results)))
		}
	}
}

// truncate returns the first n characters of s.
func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting Gemini Live Grounding Metadata Test Bot")

	s2s := live.New(live.Config{
		APIKey:       os.Getenv("GOOGLE_API_KEY"),
		Instructions: systemInstruction,
		Voice:        "Charon", // Aoede, Charon, Fenrir, Kore, Puck
		Tools:        tools(),
	})

	// A processor to capture grounding metadata.
	grounding := newGroundingMetadataProcessor()

	convo := frames.NewLLMContext("")
	// Gemini Live doesn't emit user-turn frames. Server-side VAD is on by
	// default; to surface turn frames (for RTVI speech events, turn observers,
	// and so on) give the user aggregator local turn detection with
	// aggregators.WithVAD. See realtime-gemini-live for the full discussion.
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),
		agg.User(),
		s2s,
		grounding, // The grounding processor
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself and let me know that you can help with current information " +
				"by searching the web. Ask me what current information I'd like to know about.",
		})
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
