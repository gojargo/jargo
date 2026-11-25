// Command voice-concurrent-tts is a voice agent that pushes several
// TTSSpeakFrames at once, each synthesized in its own context.
//
// A TTS service that does not hold frames per turn opens a context for each
// TTSSpeakFrame as it arrives, so the three sentences below synthesize
// concurrently on one connection and still play in order. The bot speaks a
// burst on connect and another when the user asks for the list (the read_list
// tool).
//
// Pick the TTS with the runner body: {"tts": "cartesia"} (the default) or
// {"tts": "elevenlabs"}.
//
//	CARTESIA_API_KEY=… ELEVENLABS_API_KEY=… ELEVENLABS_VOICE_ID=… OPENAI_API_KEY=… \
//	  go run ./examples/voice/voice-concurrent-tts
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
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/openai/realtime"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

//nolint:gochecknoglobals // the sentences spoken on connect
var greeting = []string{
	"First, the striped zebra crossed the wide muddy river at dawn.",
	"Second, the old lighthouse keeper polished every brass lamp in the tower.",
	"Third, a quiet violinist tuned her strings before the evening concert.",
}

//nolint:gochecknoglobals // the sentences the read_list tool speaks
var list = []string{
	"One, a basket of ripe orange tangerines.",
	"Two, a heavy wool blanket for the cabin.",
	"Three, a box of matches and a lantern.",
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
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. When the user asks for " +
	"the list, call read_list and say nothing else."

// ttsName is the TTS the runner body picks, Cartesia by default.
func ttsName(body json.RawMessage) string {
	var b struct {
		TTS string `json:"tts"`
	}
	if len(body) > 0 {
		if err := json.Unmarshal(body, &b); err != nil {
			slog.Warn("Reading the runner body", "err", err)
		}
	}
	if b.TTS == "" {
		return "cartesia"
	}
	return b.TTS
}

func createTTS(name string) *tts.Base {
	if name == "elevenlabs" {
		return elevenlabs.NewRealtimeTTS(elevenlabs.RealtimeTTSConfig{
			APIKey:         os.Getenv("ELEVENLABS_API_KEY"),
			VoiceID:        os.Getenv("ELEVENLABS_VOICE_ID"),
			WordTimestamps: true,
		})
	}
	return cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	name := ttsName(args.Base().Body)
	slog.Info("Starting bot", "tts", name)

	stt := realtime.NewSTT(realtime.STTConfig{APIKey: os.Getenv("OPENAI_API_KEY")})

	speech := createTTS(name)

	speakAll := func(ctx context.Context, sentences []string) {
		// Queued together, so the TTS has all of them before the first one plays.
		for _, sentence := range sentences {
			_ = speech.QueueFrame(ctx, frames.NewTTSSpeakFrame(sentence), processor.Downstream)
		}
	}

	// readList reads the packing list aloud to the user.
	readList := func(ctx context.Context, params llm.FunctionCallParams) error {
		speakAll(ctx, list)
		return params.Result(ctx, `{"status": "read aloud"}`, &frames.FunctionCallResultProperties{RunLLM: new(false)})
	}

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{{
		Name:        "read_list",
		Description: "Read the packing list aloud to the user.",
		Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
		Handler:     llm.FunctionCallHandler(readList),
	}})
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		model,           // LLM
		speech,          // TTS
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

	event.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		speakAll(ctx, greeting)
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
