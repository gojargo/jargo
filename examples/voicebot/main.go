// Command voicebot is a full voice agent built on jargo: microphone audio comes
// in over WebRTC, an STT service transcribes it, an LLM reasons over it, a TTS
// service speaks the reply, and the audio goes back out over WebRTC. RTVI events
// (the handshake and live transcripts) flow over the data channel.
//
// By default it uses Deepgram (STT), Anthropic (LLM) and ElevenLabs (TTS); set
// DEEPGRAM_API_KEY, ANTHROPIC_API_KEY and ELEVENLABS_API_KEY. Swap providers
// with the STT, LLM and TTS env vars (e.g. STT=assemblyai LLM=openai
// TTS=cartesia) and set that provider's own API key env var. Then run it, open
// http://localhost:8080, click start, and allow the microphone.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"log"
	"log/slog"
	"net/http"
	"os"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/assemblyai"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/deepseek"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/gladia"
	"github.com/gojargo/jargo/provider/google"
	"github.com/gojargo/jargo/provider/groq"
	"github.com/gojargo/jargo/provider/lmnt"
	"github.com/gojargo/jargo/provider/mem0"
	"github.com/gojargo/jargo/provider/ollama"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/gojargo/jargo/provider/rime"
	"github.com/gojargo/jargo/provider/together"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/gojargo/jargo/turntaking"
	"github.com/pion/webrtc/v4"
)

//go:embed static
var staticFiles embed.FS

const systemPrompt = "You are a friendly voice assistant. Keep your replies short, " +
	"warm and conversational — one or two sentences."

// providerOpenAI is the env-var value selecting OpenAI in each category.
const providerOpenAI = "openai"

func main() {
	const addr = ":8080"

	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}
	http.Handle("/", http.FileServer(http.FS(static)))
	http.HandleFunc("/offer", handleOffer)

	slog.Info("jargo voicebot listening", "url", "http://localhost"+addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func handleOffer(w http.ResponseWriter, r *http.Request) {
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

	go runBot(conn)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(answer); err != nil {
		slog.Error("write answer", "err", err)
	}
}

// runBot builds and runs the STT -> LLM -> TTS pipeline for one connection.
func runBot(conn *pionrtc.Connection) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	t := pionrtc.NewTransport(conn, params)

	stt := selectSTT()
	llm := selectLLM()
	tts := selectTTS()

	convo := frames.NewLLMContext(systemPrompt)

	// Turn taking (Silero VAD + Smart Turn) needs the ONNX runtime; when it is
	// unavailable the bot still works, falling back to STT endpointing for
	// turn-taking and losing barge-in.
	detector := buildTurnTaking()

	procs := []processor.Processor{t.Input()}
	var aggOpts []aggregators.Option
	if detector != nil {
		procs = append(procs, detector)
		aggOpts = append(aggOpts, aggregators.WithTurnTaking())
	}
	agg := aggregators.New(convo, aggOpts...)
	procs = append(procs, stt, agg.User())
	// Optional long-term memory: when MEM0_HOST is set, recall relevant memories
	// into the context before the LLM and store new turns after it.
	if mem := buildMemory(); mem != nil {
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
	})

	ctx, cancel := context.WithCancel(context.Background())
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
	_ = conn.Close()
	slog.Info("voicebot pipeline stopped")
}

// selectSTT picks the STT service from the STT env var, defaulting to Deepgram.
// The openai and groq options are segmented (batch) and need turn taking enabled
// to delimit utterances.
func selectSTT() processor.Processor {
	switch os.Getenv("STT") {
	case "assemblyai":
		return assemblyai.NewSTT(assemblyai.Config{SampleRate: opus.SampleRate})
	case "gladia":
		return gladia.NewSTT(gladia.Config{SampleRate: opus.SampleRate})
	case providerOpenAI:
		return openai.NewSTT(openai.STTConfig{SampleRate: opus.SampleRate})
	case "groq":
		return groq.NewSTT(openai.STTConfig{SampleRate: opus.SampleRate})
	default:
		return deepgram.NewSTT(deepgram.Config{SampleRate: opus.SampleRate})
	}
}

// selectLLM picks the LLM service from the LLM env var, defaulting to Anthropic.
func selectLLM() processor.Processor {
	switch os.Getenv("LLM") {
	case providerOpenAI:
		return openai.NewLLM(openai.LLMConfig{})
	case "google":
		return google.NewLLM(google.Config{})
	case "groq":
		return groq.NewLLM(openai.LLMConfig{})
	case "together":
		return together.NewLLM(openai.LLMConfig{})
	case "deepseek":
		return deepseek.NewLLM(openai.LLMConfig{})
	case "ollama":
		return ollama.NewLLM(openai.LLMConfig{})
	default:
		return anthropic.NewLLM(anthropic.Config{})
	}
}

// selectTTS picks the TTS service from the TTS env var, defaulting to ElevenLabs.
func selectTTS() processor.Processor {
	switch os.Getenv("TTS") {
	case "cartesia":
		return cartesia.NewTTS(cartesia.Config{})
	case providerOpenAI:
		return openai.NewTTS(openai.TTSConfig{})
	case "deepgram":
		return deepgram.NewTTS(deepgram.TTSConfig{})
	case "rime":
		return rime.NewTTS(rime.Config{})
	case "lmnt":
		return lmnt.NewTTS(lmnt.Config{})
	default:
		return elevenlabs.NewTTS(elevenlabs.Config{})
	}
}

// buildMemory enables mem0 long-term memory when MEM0_HOST is set, scoping
// memories with MEM0_USER (or a default). It returns nil when memory is off.
func buildMemory() *mem0.Service {
	host := os.Getenv("MEM0_HOST")
	if host == "" {
		return nil
	}
	userID := os.Getenv("MEM0_USER")
	if userID == "" {
		userID = "voicebot-user"
	}
	slog.Info("long-term memory enabled (mem0)", "host", host, "user", userID)
	return mem0.NewMemory(mem0.Config{
		Host:   host,
		APIKey: os.Getenv("MEM0_API_KEY"),
		UserID: userID,
	})
}

// buildTurnTaking constructs a turn-taking detector from Silero VAD and Smart
// Turn. If the ONNX runtime or models cannot be loaded it logs a warning and
// returns nil, so the bot runs without turn taking.
func buildTurnTaking() *turntaking.Detector {
	v, err := vad.NewSilero()
	if err != nil {
		slog.Warn("turn taking disabled: Silero VAD unavailable (set JARGO_ONNXRUNTIME_LIB)", "err", err)
		return nil
	}
	tr, err := turn.NewSmartTurnV3()
	if err != nil {
		slog.Warn("turn taking disabled: Smart Turn unavailable", "err", err)
		_ = v.Close()
		return nil
	}
	slog.Info("turn taking enabled (Silero VAD + Smart Turn v3)")
	return turntaking.New(turntaking.Config{VAD: v, Turn: tr})
}
