// Command voice-soniox is a headless jargo voice backend.
//
// It uses Soniox speech-to-text, with an Anthropic LLM and ElevenLabs TTS.
//
// jargo is a backend framework, so this is a server only — it exposes the
// WebRTC signaling endpoint POST /offer and no web UI. Point a browser client
// at it (the nextjs-voicebot example in gojargo/jargo-client-react, with
// NEXT_PUBLIC_JARGO_URL=http://localhost:8080).
//
//	SONIOX_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/voice/soniox
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
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
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/soniox"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/gojargo/jargo/turns"
	"github.com/pion/webrtc/v4"
)

const systemPrompt = "You are a friendly voice assistant. Keep your replies short, " +
	"warm and conversational — one or two sentences."

func main() {
	http.HandleFunc("/offer", withCORS(handleOffer))
	slog.Info("jargo voice backend listening", "url", "http://localhost:8080", "signaling", "POST /offer")
	log.Fatal(http.ListenAndServe(":8080", nil))
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
	defer func() { _ = conn.Close() }()

	// --- the provider stack: the only part that differs between examples ---
	stt := soniox.NewSTT(soniox.Config{APIKey: os.Getenv("SONIOX_API_KEY"), SampleRate: opus.SampleRate})
	llm := anthropic.NewLLM(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
	tts := elevenlabs.NewTTS(elevenlabs.Config{APIKey: os.Getenv("ELEVENLABS_API_KEY")})
	// ----------------------------------------------------------------------

	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	t := pionrtc.NewTransport(conn, params)

	convo := frames.NewLLMContext(systemPrompt)

	// Turn taking (Silero VAD + Smart Turn) needs the ONNX runtime; without it
	// the bot still works, falling back to STT endpointing and losing barge-in.
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

	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("pipeline ended", "err", err)
	}
}

// buildTurnStack builds the turn-taking stack (Silero VAD + Smart Turn v3). If
// the ONNX runtime or models cannot be loaded it logs a warning and returns
// nil, nil, so the bot runs without turn taking (and without barge-in).
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
	vp := vadproc.New(vadproc.Config{VAD: vd})
	tp := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: turns.DefaultStartStrategies(),
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr})},
		},
	})
	return vp, tp
}

// withCORS allows a browser client served from another origin (e.g. the Next.js
// dev server on :3000) to POST offers to this backend.
func withCORS(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.Header().Set("Access-Control-Max-Age", "86400")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h(w, r)
	}
}
