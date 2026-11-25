// Command features-switch-languages switches between English and Spanish when
// the user asks.
//
// The TTS is a ParallelPipeline with a Cartesia voice per language, each behind
// a FunctionFilter that lets frames through only while its language is the
// current one. The LLM calls switch_language to change it.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-switch-languages
//
// Then open http://localhost:7860, connect, and ask the bot to speak Spanish.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"sync"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
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

// switchLanguage is the TTS: a voice per language, only the current one
// speaking.
type switchLanguage struct {
	*pipeline.ParallelPipeline

	mu              sync.Mutex
	currentLanguage string
}

func newSwitchLanguage() (*switchLanguage, error) {
	s := &switchLanguage{currentLanguage: "English"}

	englishTTS := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	spanishTTS := cartesia.NewTTS(cartesia.Config{
		APIKey:   os.Getenv("CARTESIA_API_KEY"),
		Language: language.Spanish,
		VoiceID:  "d4db5fb9-f44b-4bd1-85fa-192e0f0d75f9", // Spanish-speaking Lady
	})

	downstream := processor.Downstream
	parallel, err := pipeline.NewParallel(
		// English
		[]processor.Processor{processor.NewFunctionFilter("EnglishFilter", &downstream, s.englishFilter), englishTTS},
		// Spanish
		[]processor.Processor{processor.NewFunctionFilter("SpanishFilter", &downstream, s.spanishFilter), spanishTTS},
	)
	if err != nil {
		return nil, err
	}
	s.ParallelPipeline = parallel
	return s, nil
}

// language returns the language the bot speaks now.
func (s *switchLanguage) language() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentLanguage
}

// switchLanguage switches to another language when the user asks.
func (s *switchLanguage) switchLanguage(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Language string `json:"language"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	s.mu.Lock()
	s.currentLanguage = args.Language
	s.mu.Unlock()
	result, err := json.Marshal(map[string]string{
		"voice": "Your answers from now on should be in " + s.language() + ".",
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

func (s *switchLanguage) englishFilter(frames.Frame) bool { return s.language() == "English" }

func (s *switchLanguage) spanishFilter(frames.Frame) bool { return s.language() == "Spanish" }

// tools describes the functions to the LLM, each carrying its own handler.
func tools(tts *switchLanguage) []frames.Tool {
	return []frames.Tool{
		{
			Name:        "switch_language",
			Description: "Switch to another language when the user asks you to.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"language": {
						"type": "string",
						"description": "The language the user wants you to speak."
					}
				},
				"required": ["language"]
			}`),
			Handler: llm.FunctionCallHandler(tts.switchLanguage),
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
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the " +
	"user said in a creative, helpful, and brief way. You can speak the following languages: 'English' and 'Spanish'."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{
		APIKey:   os.Getenv("DEEPGRAM_API_KEY"),
		Language: "multi",
	})

	tts, err := newSwitchLanguage()
	if err != nil {
		return err
	}

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools(tts))
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		model,           // LLM
		tts,             // TTS (the bot speaks the chosen language)
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
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user and let them know the languages you speak. " +
				"Your initial responses should be in " + tts.language() + ".",
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
