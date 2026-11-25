// Command openai-responses-frontend-predictive-delegation is an engineering
// assistant with a backend, with predictive delegation.
//
// It is openai-responses-frontend with predictive delegation turned on, an
// optimization that can speed up responses that need backend involvement. It
// uses a fast yes/no classifier to determine, after each user request, whether
// it needs to wait for the frontend's inference or can go ahead and trigger an
// early backend delegation. The cost is money, not time: the classifier runs
// in parallel with the frontend, but on every turn, and an early delegation
// abandons a frontend request already under way.
//
// The backend, its tools and both prompts are in the backend package. Try:
// "fix the flaky retry test in the HTTP client", then ask for something else
// while it works.
//
// The architecture:
//
//	Main worker (transport + STT + LLMWithBackend + TTS)
//	  ├── predictive delegation: Jev classifier, racing the frontend
//	  ├── frontend: fast LLM, delegate tool
//	  └── backend: BackendWorker (Claude + tools), attached for the session
//
// Run it with:
//
//	OPENAI_API_KEY=… ANTHROPIC_API_KEY=… DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	TYPESAFE_API_KEY=… \
//	  go run ./examples/multi-worker/llm-with-backend/openai-responses-frontend-predictive-delegation
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/classifier/jev"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/examples/multi-worker/llm-with-backend/backend"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/llmwithbackend"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/responses"
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", // Jacqueline
	})

	classifier, err := jev.New(jev.Config{APIKey: os.Getenv("TYPESAFE_API_KEY")})
	if err != nil {
		return err
	}
	predictive, err := llmwithbackend.NewPredictiveDelegation(llmwithbackend.PredictiveDelegationConfig{
		Classifier: classifier,
	})
	if err != nil {
		return err
	}

	frontend := responses.NewLLM(responses.Config{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Instructions: backend.FrontendInstructions,
	})
	// A brief word from the LLM while the backend works.
	if err = frontend.AddExtension(turns.NewWaitReassuranceLLMExtension(turns.WaitReassuranceConfig{})); err != nil {
		return err
	}

	be, err := backend.BuildBackend()
	if err != nil {
		return err
	}
	llm, err := llmwithbackend.New(llmwithbackend.Config{
		Frontend:             frontend,
		Backend:              be,
		PredictiveDelegation: predictive,
		// Every exchange with the backend (each request, output, tool-call phase,
		// error and idle) is sent to the client as an RTVI "server-message" whose
		// data has "type": "llm-with-backend", so the prebuilt UI's Events panel
		// shows the backend at work. Tracing is off by default; comment out the
		// connector to leave it off.
		Connector: llmwithbackend.NewBackendConnector(llmwithbackend.ConnectorConfig{ClientTrace: true}),
	})
	if err != nil {
		return err
	}

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	// The frontend's tools are installed by the service; the real tools live in
	// the backend.
	convo := frames.NewLLMContext("")
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
		Name: "frontend",
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		// The backend's calls are reported to the client with their names,
		// arguments and results, so a UI can show what the backend is doing. The
		// handoff itself is hidden, so those calls show at top level, as they do
		// for OpenAI Live's client delegation; remove the "delegate" entry to see
		// it as a call.
		RTVIObserverParams: &rtvi.ObserverParams{
			FunctionCallReportLevel: map[string]rtvi.FunctionCallReportLevel{
				"*":        rtvi.ReportFull,
				"delegate": rtvi.ReportDisabled,
			},
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Greet the user and ask how you can help."})
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
