// Command gemini-live-frontend is an engineering assistant with a backend:
// a speech-to-speech frontend delegating to a backend LLM.
//
// The frontend is Gemini Live, holding the spoken conversation with no
// tools of its own. Anything that needs tools or careful reasoning it hands to
// a backend running Claude, and relays what comes back as it comes.
// LLMWithBackend wires the two together: it installs the delegate tool on the
// frontend and runs the backend as a worker of its own. With a speech-to-speech
// frontend the model words the request itself, since its context can lag the
// audio.
//
// The backend, its tools and both prompts are in the backend package, shared
// with openai-responses-frontend, which puts a cascade pipeline in the
// frontend's place. Try: "fix the flaky retry test in the HTTP client", then
// ask for something else while it works.
//
// The architecture:
//
//	Main worker (transport + LLMWithBackend)
//	  ├── frontend: speech-to-speech model, delegate tool
//	  └── backend: BackendWorker (Claude + tools), attached for the session
//
// Run it with:
//
//	GOOGLE_API_KEY=… ANTHROPIC_API_KEY=… \
//	  go run ./examples/multi-worker/llm-with-backend/gemini-live-frontend
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/examples/multi-worker/llm-with-backend/backend"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/llmwithbackend"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/provider/google/live"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
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

	be, err := backend.BuildBackend()
	if err != nil {
		return err
	}
	llm, err := llmwithbackend.New(llmwithbackend.Config{
		Frontend: live.New(live.Config{
			APIKey:       os.Getenv("GOOGLE_API_KEY"),
			Instructions: backend.FrontendInstructions,
		}),
		Backend: be,
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

	// The frontend's tools are installed by the service; the real tools live in
	// the backend.
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Greet the user and ask how you can help."})
	// Gemini Live drives the conversation server-side and emits no turn frames
	// (UserStartedSpeakingFrame, UserStoppedSpeakingFrame). The local VAD adds
	// supplemental turn frames for processors that expect them, such as RTVI,
	// whose clients start a new user transcript entry on them. They are
	// approximate: they may not always align with the server's turns.
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		agg.User(),
		llm,
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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
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
