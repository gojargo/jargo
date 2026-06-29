// Command voicebot is a full voice agent built on jargo: microphone audio comes
// in over WebRTC, an STT service transcribes it, an LLM reasons over it, a TTS
// service speaks the reply, and the audio goes back out over WebRTC. RTVI events
// (the handshake and live transcripts) flow over the data channel.
//
// Configuration is read with Viper from the environment (and an optional
// voicebot.yaml in the working directory; env overrides it). By default it uses
// Deepgram (STT), Anthropic (LLM) and ElevenLabs (TTS); set DEEPGRAM_API_KEY,
// ANTHROPIC_API_KEY and ELEVENLABS_API_KEY. Swap providers with STT, LLM and TTS
// (e.g. STT=assemblyai LLM=openai TTS=cartesia) and set that provider's own API
// key. Then run it, open http://localhost:8080, click start, allow the mic.
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
	"github.com/gojargo/jargo/provider/assemblyai"
	"github.com/gojargo/jargo/provider/azureopenai"
	"github.com/gojargo/jargo/provider/azurespeech"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/deepseek"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/fish"
	"github.com/gojargo/jargo/provider/gladia"
	"github.com/gojargo/jargo/provider/google"
	"github.com/gojargo/jargo/provider/groq"
	"github.com/gojargo/jargo/provider/hume"
	"github.com/gojargo/jargo/provider/lmnt"
	"github.com/gojargo/jargo/provider/mem0"
	"github.com/gojargo/jargo/provider/minimax"
	"github.com/gojargo/jargo/provider/mistral"
	"github.com/gojargo/jargo/provider/nebius"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/provider/qwen"
	"github.com/gojargo/jargo/provider/rime"
	"github.com/gojargo/jargo/provider/sambanova"
	"github.com/gojargo/jargo/provider/soniox"
	"github.com/gojargo/jargo/provider/speechmatics"
	"github.com/gojargo/jargo/provider/together"
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

// providerOpenAI is the config value selecting OpenAI in each category.
const providerOpenAI = "openai"

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

	// Validate the selected providers up front so misconfiguration fails at
	// startup, not on the first call.
	for _, sel := range []func(*viper.Viper) (processor.Processor, error){selectSTT, selectLLM, selectTTS} {
		if _, err = sel(v); err != nil {
			return fmt.Errorf("provider configuration: %w", err)
		}
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
	v.SetDefault("stt", "deepgram")
	v.SetDefault("llm", "anthropic")
	v.SetDefault("tts", "elevenlabs")
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

	stt, err := selectSTT(v)
	if err != nil {
		slog.Error("STT unavailable", "err", err)
		return
	}
	llm, err := selectLLM(v)
	if err != nil {
		slog.Error("LLM unavailable", "err", err)
		return
	}
	tts, err := selectTTS(v)
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
// keeps each provider case to a line and ensures a misconfigured provider fails
// fast with a clear error rather than at the first request.
func build[C interface{ Validate() error }, S processor.Processor](cfg C, ctor func(C) S) (processor.Processor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return ctor(cfg), nil
}

// selectSTT picks the STT service from the "stt" setting, defaulting to Deepgram.
// The openai and groq options are segmented (batch) and need turn taking enabled
// to delimit utterances.
func selectSTT(v *viper.Viper) (processor.Processor, error) {
	rate := opus.SampleRate
	switch v.GetString("stt") {
	case "assemblyai":
		return build(assemblyai.Config{APIKey: v.GetString("ASSEMBLYAI_API_KEY"), SampleRate: rate}, assemblyai.NewSTT)
	case "gladia":
		return build(gladia.Config{APIKey: v.GetString("GLADIA_API_KEY"), SampleRate: rate}, gladia.NewSTT)
	case "speechmatics":
		return build(speechmatics.Config{APIKey: v.GetString("SPEECHMATICS_API_KEY"), SampleRate: rate}, speechmatics.NewSTT)
	case "soniox":
		return build(soniox.Config{APIKey: v.GetString("SONIOX_API_KEY"), SampleRate: rate}, soniox.NewSTT)
	case "azure":
		return build(azureopenai.STTConfig{
			Endpoint:   v.GetString("AZURE_OPENAI_ENDPOINT"),
			Deployment: v.GetString("azure_stt_deployment"),
			STTConfig:  openai.STTConfig{APIKey: v.GetString("AZURE_OPENAI_API_KEY"), SampleRate: rate},
		}, azureopenai.NewSTT)
	case providerOpenAI:
		return build(openai.STTConfig{APIKey: v.GetString("OPENAI_API_KEY"), SampleRate: rate}, openai.NewSTT)
	case "groq":
		return build(openai.STTConfig{APIKey: v.GetString("GROQ_API_KEY"), SampleRate: rate}, groq.NewSTT)
	default:
		return build(deepgram.Config{APIKey: v.GetString("DEEPGRAM_API_KEY"), SampleRate: rate}, deepgram.NewSTT)
	}
}

// selectLLM picks the LLM service from the "llm" setting, defaulting to Anthropic.
func selectLLM(v *viper.Viper) (processor.Processor, error) {
	switch v.GetString("llm") {
	case providerOpenAI:
		return build(openai.LLMConfig{APIKey: v.GetString("OPENAI_API_KEY")}, openai.NewLLM)
	case "google":
		return build(google.Config{APIKey: v.GetString("GEMINI_API_KEY")}, google.NewLLM)
	case "groq":
		return build(openai.LLMConfig{APIKey: v.GetString("GROQ_API_KEY")}, groq.NewLLM)
	case "together":
		return build(openai.LLMConfig{APIKey: v.GetString("TOGETHER_API_KEY")}, together.NewLLM)
	case "deepseek":
		return build(openai.LLMConfig{APIKey: v.GetString("DEEPSEEK_API_KEY")}, deepseek.NewLLM)
	case "mistral":
		return build(openai.LLMConfig{APIKey: v.GetString("MISTRAL_API_KEY")}, mistral.NewLLM)
	case "nebius":
		return build(openai.LLMConfig{APIKey: v.GetString("NEBIUS_API_KEY")}, nebius.NewLLM)
	case "sambanova":
		return build(openai.LLMConfig{APIKey: v.GetString("SAMBANOVA_API_KEY")}, sambanova.NewLLM)
	case "qwen":
		return build(openai.LLMConfig{APIKey: v.GetString("DASHSCOPE_API_KEY")}, qwen.NewLLM)
	case "ollama":
		return build(openai.LLMConfig{}, ollama.NewLLM) // local; no key required
	default:
		return build(anthropic.Config{APIKey: v.GetString("ANTHROPIC_API_KEY")}, anthropic.NewLLM)
	}
}

// selectTTS picks the TTS service from the "tts" setting, defaulting to ElevenLabs.
func selectTTS(v *viper.Viper) (processor.Processor, error) {
	switch v.GetString("tts") {
	case "cartesia":
		return build(cartesia.Config{APIKey: v.GetString("CARTESIA_API_KEY")}, cartesia.NewTTS)
	case providerOpenAI:
		return build(openai.TTSConfig{APIKey: v.GetString("OPENAI_API_KEY")}, openai.NewTTS)
	case "deepgram":
		return build(deepgram.TTSConfig{APIKey: v.GetString("DEEPGRAM_API_KEY")}, deepgram.NewTTS)
	case "rime":
		return build(rime.Config{APIKey: v.GetString("RIME_API_KEY")}, rime.NewTTS)
	case "lmnt":
		return build(lmnt.Config{APIKey: v.GetString("LMNT_API_KEY")}, lmnt.NewTTS)
	case "azure":
		return build(azurespeech.TTSConfig{
			APIKey: v.GetString("AZURE_SPEECH_KEY"),
			Region: v.GetString("azure_region"),
			Voice:  v.GetString("azure_voice"),
		}, azurespeech.NewTTS)
	case "hume":
		return build(hume.Config{APIKey: v.GetString("HUME_API_KEY"), VoiceID: v.GetString("hume_voice")}, hume.NewTTS)
	case "fish":
		return build(fish.Config{APIKey: v.GetString("FISH_API_KEY"), ReferenceID: v.GetString("fish_voice")}, fish.NewTTS)
	case "minimax":
		return build(minimax.Config{
			APIKey:  v.GetString("MINIMAX_API_KEY"),
			GroupID: v.GetString("MINIMAX_GROUP_ID"),
			VoiceID: v.GetString("minimax_voice"),
		}, minimax.NewTTS)
	default:
		return build(elevenlabs.Config{APIKey: v.GetString("ELEVENLABS_API_KEY")}, elevenlabs.NewTTS)
	}
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
