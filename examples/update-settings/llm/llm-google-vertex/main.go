// Command llm-google-vertex changes the settings of a Google Vertex AI LLM
// while the conversation runs: ten seconds after the client connects, it sets
// the temperature to 0.1.
//
// Speech-to-text is Deepgram and text-to-speech Cartesia.
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… GOOGLE_CLOUD_PROJECT_ID=… GOOGLE_VERTEX_TEST_CREDENTIALS=… \
//	  go run ./examples/update-settings/llm/llm-google-vertex
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/google/vertex"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// errNoCredentials is returned when no service-account key is given.
var errNoCredentials = errors.New("GOOGLE_VERTEX_TEST_CREDENTIALS is not set")

// credentials are the service-account key in GOOGLE_VERTEX_TEST_CREDENTIALS.
// For Application Default Credentials, set vertex.Credentials.TokenSource to
// the token source golang.org/x/oauth2/google gives instead.
func credentials() (vertex.Credentials, error) {
	key := os.Getenv("GOOGLE_VERTEX_TEST_CREDENTIALS")
	if key == "" {
		return vertex.Credentials{}, errNoCredentials
	}
	return vertex.Credentials{JSON: []byte(key)}, nil
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
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
	" to what the user said in a creative, helpful, and brief way."

// updateAfter is how long the bot waits before each settings change.
const updateAfter = 10 * time.Second

// sleep waits for updateAfter, and reports false when ctx ends first.
func sleep(ctx context.Context) bool {
	select {
	case <-time.After(updateAfter):
		return true
	case <-ctx.Done():
		return false
	}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	creds, err := credentials()
	if err != nil {
		return err
	}
	llm := vertex.NewLLM(vertex.LLMConfig{
		Credentials: creds,
		ProjectID:   os.Getenv("GOOGLE_CLOUD_PROJECT_ID"),
	})
	llm.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		llm,             // LLM
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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})

		go func() {
			if !sleep(ctx) {
				return
			}
			slog.Info("Updating Google Vertex LLM settings: temperature=0.1")
			worker.QueueFrame(frames.NewLLMUpdateSettingsFrame(&gemini.LLMSettings{
				Temperature: settings.Set(0.1),
			}))
		}()
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
