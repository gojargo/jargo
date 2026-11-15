// Command features-switch-voices switches between three voices when the user
// asks.
//
// The TTS is a ParallelPipeline with a Cartesia service per voice, each behind
// a FunctionFilter that lets frames through only while its voice is the
// current one. The LLM calls switch_voice to change it.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-switch-voices
//
// Then open http://localhost:7860, connect, and ask the bot to change its voice.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"sync"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
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
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// switchVoices is the TTS: a service per voice, only the current one
// speaking.
type switchVoices struct {
	*pipeline.ParallelPipeline

	mu           sync.Mutex
	currentVoice string
}

func newSwitchVoices() (*switchVoices, error) {
	s := &switchVoices{currentVoice: "News Lady"}

	newsLady := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "bf991597-6c13-47e4-8411-91ec2de5c466", // Newslady
	})

	britishLady := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "71a7ad14-091c-4e8e-a314-022ece01c121", // British Reading Lady
	})

	barbershopMan := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "a0e99841-438c-4a64-b679-ae501e7d6091", // Barbershop Man
	})

	downstream := processor.Downstream
	parallel, err := pipeline.NewParallel(
		// News Lady voice
		[]processor.Processor{processor.NewFunctionFilter("NewsLadyFilter", &downstream, s.newsLadyFilter), newsLady},
		// British Reading Lady voice
		[]processor.Processor{
			processor.NewFunctionFilter("BritishLadyFilter", &downstream, s.britishLadyFilter), britishLady,
		},
		// Barbershop Man voice
		[]processor.Processor{
			processor.NewFunctionFilter("BarbershopManFilter", &downstream, s.barbershopManFilter), barbershopMan,
		},
	)
	if err != nil {
		return nil, err
	}
	s.ParallelPipeline = parallel
	return s, nil
}

// voice returns the voice the bot speaks with now.
func (s *switchVoices) voice() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentVoice
}

// switchVoice switches the voice, only when the user asks.
func (s *switchVoices) switchVoice(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Voice string `json:"voice"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	s.mu.Lock()
	s.currentVoice = args.Voice
	s.mu.Unlock()
	result, err := json.Marshal(map[string]string{
		"voice": "You are now using your " + s.voice() + " voice. Your responses should now be as if you were a " +
			s.voice() + ".",
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

func (s *switchVoices) newsLadyFilter(frames.Frame) bool { return s.voice() == "News Lady" }

func (s *switchVoices) britishLadyFilter(frames.Frame) bool { return s.voice() == "British Lady" }

func (s *switchVoices) barbershopManFilter(frames.Frame) bool { return s.voice() == "Barbershop Man" }

// tools describes the functions to the LLM, each carrying its own handler.
func tools(tts *switchVoices) []frames.Tool {
	return []frames.Tool{
		{
			Name:        "switch_voice",
			Description: "Switch your voice only when the user asks you to.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"voice": {
						"type": "string",
						"description": "The voice the user wants you to use."
					}
				},
				"required": ["voice"]
			}`),
			Handler: llm.FunctionCallHandler(tts.switchVoice),
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
	"user said in a creative and helpful way. You can do the following voices: 'News Lady', 'British Lady' and " +
	"'Barbershop Man'."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts, err := newSwitchVoices()
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
		t.Input(), // Transport user input
		stt,
		agg.User(),      // User responses
		model,           // LLM
		tts,             // TTS with switch voice functionality
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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user and let them know the voices you can do. " +
				"Your initial responses should be as if you were a " + tts.voice() + ".",
		})
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
