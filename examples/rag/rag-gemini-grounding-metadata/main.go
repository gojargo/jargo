// Command rag-gemini-grounding-metadata answers with the latest news, grounded
// in Google Search, and logs the grounding metadata Gemini reports.
//
// The Gemini LLM is given Google Search as a tool, in Gemini's own format. An
// observer logs every LLMSearchResponseFrame entering or leaving the LLM: what
// the model searched for, and the sources its answer rests on.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… GOOGLE_API_KEY=… \
//	  go run ./examples/rag/rag-gemini-grounding-metadata
//
// Then open http://localhost:7860, connect, and ask for the news somewhere.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	googleframes "github.com/gojargo/jargo/provider/google/frames"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// tools offers Google Search, in Gemini's own format.
func tools() frames.ToolsSchema {
	return frames.ToolsSchema{
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeGemini: {
				map[string]any{"google_search": map[string]any{}},
			},
		},
	}
}

//nolint:lll // the prompt is the model's text, kept as written
const systemInstruction = `
You are an expert at providing the most recent news from any place. Your responses will be converted to audio, so avoid using special characters or overly complex formatting.

Always use the google search API to retrieve the latest news. You must also use it to check which day is today.

You can:
- Use the Google search API to check the current date.
- Provide the most recent and relevant news from any place by using the google search API.
- Answer any questions the user may have, ensuring your responses are accurate and concise.

Start each interaction by asking the user about which place they would like to know the information.
`

// llmService is what every LLM service implements.
type llmService interface{ LLMService() }

// llmSearchLoggerObserver logs the search responses entering or leaving an
// LLM service.
type llmSearchLoggerObserver struct{}

// OnPushFrame implements processor.Observer.
func (llmSearchLoggerObserver) OnPushFrame(data processor.FramePushed) {
	_, srcLLM := data.Source.(llmService)
	_, dstLLM := data.Destination.(llmService)
	if !srcLLM && !dstLLM {
		return
	}
	if fr, ok := data.Frame.(*googleframes.LLMSearchResponseFrame); ok {
		slog.Debug(fmt.Sprintf("→ %s LLM SEARCH RESPONSE FRAME: %s at %.2fs",
			data.Destination.Name(), fr, data.Timestamp.Seconds()))
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	llm := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetToolsSchema(tools())
	convo.AddMessage(frames.Message{
		Role: frames.RoleDeveloper,
		Text: "Start by greeting the user warmly, introducing yourself, and mentioning the current day. " +
			"Be friendly and engaging to set a positive tone for the interaction.",
	})
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		Observers:               []pipeline.Observer{llmSearchLoggerObserver{}},
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Start the conversation with no prompt, letting the LLM follow its
		// system instruction.
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
