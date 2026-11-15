// Command features-app-resources shows pipeline.WorkerConfig.AppResources.
//
// AppResources is whatever the application wants to share across a session:
// database handles, HTTP clients, feature flags, per-user state, observability
// clients, in-memory caches. The framework passes it through untouched and
// exposes it as the worker's AppResources, so any code with a handle on the
// worker can read or change it.
//
// Two of the read sites are shown here:
//
//   - Tool handlers read it from llm.FunctionCallParams.AppResources.
//   - Custom processors read it from Running().AppResources().
//
// Two small loggers stand in for the shared thing: a toolCallLogger written
// from the tool handlers, and a transcriptionLogger written from a custom
// processor in the pipeline. A real application might pass a database pool, a
// Redis client or an SDK client just as well. The mechanics are the same
// whatever goes in: build it once, hand it to the worker, read it at each
// site, inspect it after the session.
//
// The resources are bundled in a typed appResources struct and asserted back
// to it at each read site. The worker does not care what type it is given, but
// a typed container is safer than lookups by string key.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-app-resources
//
// Then open http://localhost:7860, connect, and ask about the weather.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/responses"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// toolCall is one recorded tool call.
type toolCall struct {
	Timestamp    string         `json:"timestamp"`
	FunctionName string         `json:"function_name"`
	Arguments    map[string]any `json:"arguments"`
}

// toolCallLogger is a stand-in shared resource: swap it for whatever the
// application actually needs.
type toolCallLogger struct {
	mu    sync.Mutex
	calls []toolCall
}

// logToolCall records a tool call.
func (l *toolCallLogger) logToolCall(functionName string, arguments json.RawMessage) {
	args := map[string]any{}
	_ = json.Unmarshal(arguments, &args)
	l.mu.Lock()
	l.calls = append(l.calls, toolCall{
		Timestamp:    time.Now().UTC().Format(time.RFC3339Nano),
		FunctionName: functionName,
		Arguments:    args,
	})
	l.mu.Unlock()
	slog.Info("[ToolCallLogger] tool called", "function", functionName, "arguments", args)
}

// dump returns every recorded tool call as JSON.
func (l *toolCallLogger) dump() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	out, err := json.MarshalIndent(l.calls, "", "  ")
	if err != nil {
		return err.Error()
	}
	return string(out)
}

// transcription is one recorded user transcription.
type transcription struct {
	Timestamp string `json:"timestamp"`
	Text      string `json:"text"`
}

// transcriptionLogger records final user transcriptions, written from a custom
// processor.
type transcriptionLogger struct {
	mu      sync.Mutex
	entries []transcription
}

// logTranscription records a transcription.
func (l *transcriptionLogger) logTranscription(text string) {
	l.mu.Lock()
	l.entries = append(l.entries, transcription{
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Text:      text,
	})
	l.mu.Unlock()
	slog.Info("[TranscriptionLogger] transcription", "text", text)
}

// dump returns every recorded transcription as JSON.
func (l *transcriptionLogger) dump() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	out, err := json.MarshalIndent(l.entries, "", "  ")
	if err != nil {
		return err.Error()
	}
	return string(out)
}

// appResources is the typed container for everything the application shares
// across this session. Add fields as the application grows (a database pool,
// an HTTP client). Read sites assert back to this type.
type appResources struct {
	toolCallLogger      *toolCallLogger
	transcriptionLogger *transcriptionLogger
}

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	if resources, ok := params.AppResources.(*appResources); ok {
		resources.toolCallLogger.logToolCall(params.FunctionName, params.Arguments)
	}
	return params.Result(ctx, `{"conditions": "nice", "temperature": "75"}`, nil)
}

// getRestaurantRecommendation gets a restaurant recommendation.
func getRestaurantRecommendation(ctx context.Context, params llm.FunctionCallParams) error {
	if resources, ok := params.AppResources.(*appResources); ok {
		resources.toolCallLogger.logToolCall(params.FunctionName, params.Arguments)
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

// transcriptionLoggingProcessor logs each final user transcription into the
// shared application resources.
//
// It is the second read site: any custom processor reaches the same resources
// every tool handler sees through Running().AppResources(). Running is nil
// until the worker sets the processor up, so the read is guarded.
type transcriptionLoggingProcessor struct {
	*processor.Base
}

func newTranscriptionLoggingProcessor() *transcriptionLoggingProcessor {
	p := &transcriptionLoggingProcessor{}
	p.Base = processor.New("TranscriptionLoggingProcessor", p)
	return p
}

// ProcessFrame forwards every frame, logging final user transcriptions on the
// way through.
func (p *transcriptionLoggingProcessor) ProcessFrame(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if t, ok := f.(*frames.TranscriptionFrame); ok {
		if running := p.Running(); running != nil {
			if resources, ok := running.AppResources().(*appResources); ok {
				resources.transcriptionLogger.logTranscription(t.Text)
			}
		}
	}
	return p.PushFrame(ctx, f, dir)
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will" +
	" be spoken aloud, so avoid emojis, bullet points, or other formatting " +
	"that can't be spoken. Respond to what the user said in a creative, " +
	"helpful, and brief way."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := responses.NewLLM(responses.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	events.On(model.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		slog.Error("LLM connection error", "err", msg)
	})

	model.OnFunctionCallsStarted(func(ctx context.Context, _ []frames.ToolCall) {
		speak := frames.NewTTSSpeakFrame("Let me check on that.")
		speak.AppendToContext = false
		_ = tts.QueueFrame(ctx, speak, processor.Downstream)
	})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		newTranscriptionLoggingProcessor(),
		agg.User(),
		model,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	calls := &toolCallLogger{}
	transcriptions := &transcriptionLogger{}
	resources := &appResources{
		toolCallLogger:      calls,
		transcriptionLogger: transcriptions,
	}

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		AppResources:            resources,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

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

	err = r.Run(ctx, workers.RunOptions{})

	slog.Info("Tool calls logged during session:\n" + calls.dump())
	slog.Info("Transcriptions logged during session:\n" + transcriptions.dump())
	return err
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
