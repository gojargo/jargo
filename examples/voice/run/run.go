// Package run holds the shared scaffolding for the per-provider voice examples
// under examples/voice. Each example wires one provider stack explicitly in Go
// and calls Voice; this package owns the WebRTC transport, the
// STT -> LLM -> TTS pipeline, turn-taking and barge-in, and the HTTP server.
//
// jargo itself reads no environment variables: the library takes explicit
// Config structs. The examples follow the same rule — the provider and its
// settings are written out in Go; only the API key secret is read from the
// environment, the way you would source any secret.
package run

import (
	"cmp"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vadproc"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/gojargo/jargo/turns"
	"github.com/pion/webrtc/v4"
)

//go:embed static
var staticFiles embed.FS

// SampleRate is the audio sample rate the examples run at (Opus full-band).
const SampleRate = opus.SampleRate

const systemPrompt = "You are a friendly voice assistant. Keep your replies short, " +
	"warm and conversational — one or two sentences."

// Builder constructs one service. It is called once per connection so each
// caller gets a fresh STT/LLM/TTS — the STT and TTS services hold a per-session
// stream and must not be shared between connections.
type Builder = func() (processor.Processor, error)

// Builders is the per-connection STT/LLM/TTS stack for an example. Use Service
// (or the Default* builders) to fill it in.
type Builders struct {
	STT Builder
	LLM Builder
	TTS Builder
}

// Build validates cfg and, when valid, constructs the service with ctor. It
// keeps each provider's builder to a single line and makes a misconfigured
// provider fail fast with a clear error rather than at the first request.
func Build[C interface{ Validate() error }, S processor.Processor](cfg C, ctor func(C) S) (processor.Processor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return ctor(cfg), nil
}

// Voice serves the voice bot on :8080 (override with the ADDR env var), building
// a fresh STT/LLM/TTS stack from b for each browser connection. It blocks until
// the server exits. Open the URL it logs, click start, and allow the mic.
func Voice(b Builders) error {
	// Validate the configured providers up front so misconfiguration fails at
	// startup, not on the first call.
	for _, mk := range []Builder{b.STT, b.LLM, b.TTS} {
		if _, err := mk(); err != nil {
			return fmt.Errorf("provider configuration: %w", err)
		}
	}

	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(static)))
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) { handleOffer(w, r, b) })

	addr := cmp.Or(os.Getenv("ADDR"), ":8080")
	slog.Info("jargo voice example listening", "url", "http://localhost"+addr)
	return http.ListenAndServe(addr, mux)
}

func handleOffer(w http.ResponseWriter, r *http.Request, b Builders) {
	var offer webrtc.SessionDescription
	if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	conn, err := pionrtc.NewConnection()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	answer, err := conn.Answer(offer)
	if err != nil {
		_ = conn.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	go runBot(conn, b)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(answer); err != nil {
		slog.Error("write answer", "err", err)
	}
}

// runBot builds and runs the STT -> LLM -> TTS pipeline for one connection.
func runBot(conn *pionrtc.Connection, b Builders) {
	defer func() { _ = conn.Close() }()

	stt, err := b.STT()
	if err != nil {
		slog.Error("STT unavailable", "err", err)
		return
	}
	llm, err := b.LLM()
	if err != nil {
		slog.Error("LLM unavailable", "err", err)
		return
	}
	tts, err := b.TTS()
	if err != nil {
		slog.Error("TTS unavailable", "err", err)
		return
	}

	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	t := pionrtc.NewTransport(conn, params)

	convo := frames.NewLLMContext(systemPrompt)

	// Turn taking (Silero VAD + Smart Turn) needs the ONNX runtime; when it is
	// unavailable the bot still works, falling back to STT endpointing for
	// turn-taking and losing barge-in.
	vadProc, turnsProc := buildTurnStack()

	procs := []processor.Processor{t.Input()}
	if vadProc != nil {
		procs = append(procs, vadProc)
	}
	procs = append(procs, stt)
	var aggOpts []aggregators.Option
	if turnsProc != nil {
		procs = append(procs, turnsProc)
		aggOpts = append(aggOpts, aggregators.WithTurnTaking())
	}
	agg := aggregators.New(convo, aggOpts...)
	procs = append(procs, agg.User(), llm, tts, rtvi.NewProcessor(), t.Output(), agg.Assistant())

	task := pipeline.NewTask(pipeline.New(procs...), pipeline.TaskParams{
		AudioInSampleRate:  opus.SampleRate,
		AudioOutSampleRate: opus.SampleRate,
		// Emit per-turn metrics (TTFB, processing, tokens, characters) in-band so
		// the RTVI client sees live latency.
		EnableMetrics:      true,
		EnableUsageMetrics: true,
	})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-conn.Done()
		cancel()
	}()

	// Greet the caller so they hear the bot as soon as they connect.
	task.QueueFrame(frames.NewTextFrame("Hello! How can I help you today?"))

	slog.Info("voice pipeline started")
	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("voice pipeline ended", "err", err)
	}
	slog.Info("voice pipeline stopped")
}

// buildTurnStack constructs the turn-taking stack: a VAD processor (Silero) and
// a UserTurnProcessor whose stop strategy is Smart Turn v3. If the ONNX runtime
// or models cannot be loaded it logs a warning and returns nil, nil, so the bot
// runs without turn taking (STT endpointing drives turns, and barge-in is lost).
func buildTurnStack() (*vadproc.Processor, *turns.UserTurnProcessor) {
	vd, err := vad.NewSilero()
	if err != nil {
		slog.Warn("turn taking disabled: Silero VAD unavailable (set JARGO_ONNXRUNTIME_LIB)", "err", err)
		return nil, nil
	}
	tr, err := turn.NewSmartTurnV3()
	if err != nil {
		slog.Warn("turn taking disabled: Smart Turn unavailable", "err", err)
		_ = vd.Close()
		return nil, nil
	}
	slog.Info("turn taking enabled (Silero VAD + Smart Turn v3)")
	vp := vadproc.New(vadproc.Config{VAD: vd})
	tp := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: turns.DefaultStartStrategies(),
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr})},
		},
	})
	return vp, tp
}
