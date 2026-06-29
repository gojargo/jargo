// Command voicebot is the full-featured voice agent built on jargo: microphone
// audio comes in over WebRTC, Deepgram transcribes it, an Anthropic LLM reasons
// over it, ElevenLabs speaks the reply, and the audio goes back out over WebRTC.
// On top of the core STT -> LLM -> TTS pipeline it adds turn-taking and barge-in
// (Silero VAD + Smart Turn), optional long-term memory (mem0), and optional
// OpenTelemetry tracing and metrics. RTVI events (the handshake and live
// transcripts) flow over the data channel.
//
// The provider stack is fixed here so the example can focus on those advanced
// features. To see other STT/LLM/TTS providers wired explicitly — one provider
// per file — look at examples/voice (e.g. `go run ./examples/voice/cartesia`).
//
// Set DEEPGRAM_API_KEY, ANTHROPIC_API_KEY and ELEVENLABS_API_KEY, then run it,
// open http://localhost:8080, click start, and allow the mic. Long-term memory
// turns on when MEM0_HOST is set; tracing and metrics when
// OTEL_EXPORTER_OTLP_ENDPOINT is set.
//
// jargo itself reads no environment variables: the library takes explicit
// Config structs, and this app is responsible for sourcing and validating them.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"log/slog"
	"net/http"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vadproc"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/metrics"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/mem0"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/tracing"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/gojargo/jargo/turns"
	"github.com/pion/webrtc/v4"
	"github.com/spf13/viper"
)

//go:embed static
var staticFiles embed.FS

const systemPrompt = "You are a friendly voice assistant. Keep your replies short, " +
	"warm and conversational — one or two sentences."

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	v, err := loadConfig()
	if err != nil {
		return err
	}

	shutdown := setupTelemetry(v)
	defer shutdown()

	// Validate the provider stack up front so misconfiguration fails at startup,
	// not on the first call.
	if _, _, _, err = buildStack(v); err != nil {
		return fmt.Errorf("provider configuration: %w", err)
	}

	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return err
	}
	http.Handle("/", http.FileServer(http.FS(static)))
	http.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) { handleOffer(w, r, v) })

	addr := v.GetString("addr")
	slog.Info("jargo voicebot listening", "url", "http://localhost"+addr)
	return http.ListenAndServe(addr, nil)
}

// setupTelemetry installs OpenTelemetry tracing and metrics exporters when an
// OTLP endpoint is configured, returning a shutdown function (a no-op when
// telemetry is off).
func setupTelemetry(v *viper.Viper) func() {
	if v.GetString("OTEL_EXPORTER_OTLP_ENDPOINT") == "" {
		return func() {}
	}
	var shutdowns []func(context.Context) error
	if sd, err := tracing.Init(context.Background(), tracing.Config{ServiceName: "jargo-voicebot"}); err != nil {
		slog.Error("tracing init failed", "err", err)
	} else {
		shutdowns = append(shutdowns, sd)
		slog.Info("OpenTelemetry tracing enabled")
	}
	if sd, err := metrics.Init(context.Background(), metrics.Config{ServiceName: "jargo-voicebot"}); err != nil {
		slog.Error("metrics init failed", "err", err)
	} else {
		shutdowns = append(shutdowns, sd)
		slog.Info("OpenTelemetry metrics enabled")
	}
	return func() {
		for _, sd := range shutdowns {
			_ = sd(context.Background())
		}
	}
}

// loadConfig builds the Viper config: defaults, the environment (AutomaticEnv),
// and an optional voicebot.yaml in the working directory that the environment
// overrides.
func loadConfig() (*viper.Viper, error) {
	v := viper.New()
	v.AutomaticEnv()
	v.SetDefault("addr", ":8080")
	v.SetDefault("mem0_user", "voicebot-user")

	v.SetConfigName("voicebot")
	v.AddConfigPath(".")
	if err := v.ReadInConfig(); err != nil {
		var notFound viper.ConfigFileNotFoundError
		if !errors.As(err, &notFound) {
			return nil, err
		}
	}
	return v, nil
}

func handleOffer(w http.ResponseWriter, r *http.Request, v *viper.Viper) {
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

	go runBot(conn, v)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(answer); err != nil {
		slog.Error("write answer", "err", err)
	}
}

// runBot builds and runs the STT -> LLM -> TTS pipeline for one connection.
func runBot(conn *pionrtc.Connection, v *viper.Viper) {
	defer func() { _ = conn.Close() }()

	stt, llm, tts, err := buildStack(v)
	if err != nil {
		slog.Error("provider stack unavailable", "err", err)
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
	procs = append(procs, agg.User())
	// Optional long-term memory: when mem0_host is set, recall relevant memories
	// into the context before the LLM and store new turns after it.
	if mem := buildMemory(v); mem != nil {
		procs = append(procs, mem)
	}
	procs = append(procs,
		llm,
		tts,
		rtvi.NewProcessor(),
		t.Output(),
		agg.Assistant(),
	)

	task := pipeline.NewTask(pipeline.New(procs...), pipeline.TaskParams{
		AudioInSampleRate:  opus.SampleRate,
		AudioOutSampleRate: opus.SampleRate,
		// Emit per-turn metrics (TTFB, processing, tokens, characters) in-band so
		// the RTVI client sees live latency.
		EnableMetrics:      true,
		EnableUsageMetrics: true,
	})

	ctx, cancel := context.WithCancel(context.Background())
	// Root span for the session; the LLM and TTS spans nest under it.
	ctx, span := tracing.StartConversation(ctx, "")
	defer span.End()
	go func() {
		<-conn.Done()
		cancel()
	}()

	// Greet the caller so they hear the bot as soon as they connect.
	task.QueueFrame(frames.NewTextFrame("Hello! How can I help you today?"))

	slog.Info("voicebot pipeline started")
	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("voicebot pipeline ended", "err", err)
	}
	slog.Info("voicebot pipeline stopped")
}

// build validates cfg and, when valid, constructs the service with ctor. It
// ensures a misconfigured provider fails fast with a clear error rather than at
// the first request.
func build[C interface{ Validate() error }, S processor.Processor](cfg C, ctor func(C) S) (processor.Processor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return ctor(cfg), nil
}

// buildStack constructs the fixed provider stack — Deepgram (STT), Anthropic
// (LLM) and ElevenLabs (TTS) — validating each config so misconfiguration fails
// fast. See examples/voice for other providers wired one per file.
func buildStack(v *viper.Viper) (stt, llm, tts processor.Processor, err error) {
	sttCfg := deepgram.Config{APIKey: v.GetString("DEEPGRAM_API_KEY"), SampleRate: opus.SampleRate}
	stt, err = build(sttCfg, deepgram.NewSTT)
	if err != nil {
		return nil, nil, nil, err
	}
	llm, err = build(anthropic.Config{APIKey: v.GetString("ANTHROPIC_API_KEY")}, anthropic.NewLLM)
	if err != nil {
		return nil, nil, nil, err
	}
	tts, err = build(elevenlabs.Config{APIKey: v.GetString("ELEVENLABS_API_KEY")}, elevenlabs.NewTTS)
	if err != nil {
		return nil, nil, nil, err
	}
	return stt, llm, tts, nil
}

// buildMemory enables mem0 long-term memory when mem0_host is set, scoping
// memories with mem0_user. It returns nil when memory is off or misconfigured.
func buildMemory(v *viper.Viper) *mem0.Service {
	host := v.GetString("mem0_host")
	if host == "" {
		return nil
	}
	cfg := mem0.Config{
		Host:   host,
		APIKey: v.GetString("MEM0_API_KEY"),
		UserID: v.GetString("mem0_user"),
	}
	if err := cfg.Validate(); err != nil {
		slog.Error("mem0 disabled: invalid config", "err", err)
		return nil
	}
	slog.Info("long-term memory enabled (mem0)", "host", cfg.Host, "user", cfg.UserID)
	return mem0.NewMemory(cfg)
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
