// Command twiliobot is a telephony voice agent built on jargo: a Twilio phone
// call streams audio over a WebSocket (μ-law 8 kHz), an STT service transcribes
// it, an LLM reasons over it, a TTS service speaks the reply, and the audio goes
// back to the caller. A user-idle watchdog re-engages a silent caller and hangs
// up after a few unanswered nudges.
//
// Point a Twilio number's Voice webhook at https://<host>/twiml (use ngrok in
// development). Twilio fetches the TwiML, which opens a bidirectional media
// stream to wss://<host>/ws. Set DEEPGRAM_API_KEY, ANTHROPIC_API_KEY and
// ELEVENLABS_API_KEY; set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to let the
// bot hang the call up itself.
//
// jargo reads no environment variables; this app sources and validates config.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vadproc"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/twilio"
	"github.com/gojargo/jargo/turns"
	"github.com/spf13/viper"
)

// phoneSampleRate is Twilio Media Streams' fixed μ-law rate. The whole pipeline
// runs at 8 kHz so no extra resampling is needed.
const phoneSampleRate = 8000

const systemPrompt = "You are a friendly voice assistant on a phone call. Keep " +
	"your replies short, warm and conversational — one or two sentences."

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	v := loadConfig()

	http.HandleFunc("/twiml", func(w http.ResponseWriter, r *http.Request) { handleTwiML(w, r, v) })
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) { handleStream(w, r, v) })

	addr := v.GetString("addr")
	slog.Info("jargo twiliobot listening", "addr", addr, "twiml", "POST /twiml", "stream", "GET /ws")
	return http.ListenAndServe(addr, nil)
}

func loadConfig() *viper.Viper {
	v := viper.New()
	v.AutomaticEnv()
	v.SetDefault("addr", ":8080")
	v.SetDefault("idle_timeout_secs", 10)
	v.SetDefault("idle_max_retries", 3)
	return v
}

// handleTwiML returns the TwiML that opens a bidirectional media stream back to
// this server. <Connect><Stream> (not <Start><Stream>) is required so the bot
// can send audio to the caller, not only receive it.
func handleTwiML(w http.ResponseWriter, r *http.Request, v *viper.Viper) {
	host := v.GetString("public_host")
	if host == "" {
		host = r.Host
	}
	twiml := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>`+
		`<Response><Connect><Stream url="wss://%s/ws"/></Connect></Response>`, host)
	w.Header().Set("Content-Type", "text/xml")
	//nolint:gosec // TwiML XML response; host is this server's own request Host
	if _, err := w.Write([]byte(twiml)); err != nil {
		slog.Error("write twiml", "err", err)
	}
}

func handleStream(w http.ResponseWriter, r *http.Request, v *viper.Viper) {
	ser := twilio.New(twilio.Config{
		AccountSID: v.GetString("TWILIO_ACCOUNT_SID"),
		AuthToken:  v.GetString("TWILIO_AUTH_TOKEN"),
		AutoHangUp: true,
	})

	params := transport.DefaultParams()
	params.AudioInSampleRate = phoneSampleRate
	params.AudioOutSampleRate = phoneSampleRate

	t, err := wsserver.Accept(w, r, ser, params)
	if err != nil {
		slog.Error("accept websocket", "err", err)
		return
	}
	runBot(t, v)
}

// runBot builds and runs the STT -> LLM -> TTS pipeline for one call.
func runBot(t *wsserver.Transport, v *viper.Viper) {
	stt := deepgram.NewSTT(deepgram.Config{APIKey: v.GetString("DEEPGRAM_API_KEY")})
	llm := anthropic.NewLLM(anthropic.Config{APIKey: v.GetString("ANTHROPIC_API_KEY")})
	tts := elevenlabs.NewTTS(elevenlabs.Config{
		APIKey:     v.GetString("ELEVENLABS_API_KEY"),
		SampleRate: phoneSampleRate,
	})

	convo := frames.NewLLMContext(systemPrompt)
	vadProc, turnsProc := newTurnStack(v)

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
	procs = append(procs,
		agg.User(),
		llm,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	task := pipeline.NewTask(pipeline.New(procs...), pipeline.TaskParams{
		AudioInSampleRate:  phoneSampleRate,
		AudioOutSampleRate: phoneSampleRate,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-t.Done() // the caller hung up or the socket closed
		cancel()
	}()

	task.QueueFrame(frames.NewTextFrame("Hello! Thanks for calling. How can I help you today?"))

	slog.Info("twiliobot pipeline started")
	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("twiliobot pipeline ended", "err", err)
	}
	slog.Info("twiliobot pipeline stopped")
}

// newTurnStack builds the turn-taking stack — a Silero VAD processor and a
// UserTurnProcessor (Smart Turn v3 stop) with an idle watchdog. The idle
// callback re-engages a quiet caller and, after idle_max_retries unanswered
// nudges, ends the call (which auto-hangs-up via the Twilio serializer); both
// the nudge and the EndFrame go downstream toward the TTS and output. When the
// ONNX runtime or models are unavailable it returns nil, nil and the bot falls
// back to STT endpointing without turn taking or idle.
func newTurnStack(v *viper.Viper) (*vadproc.Processor, *turns.UserTurnProcessor) {
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

	maxRetries := v.GetInt("idle_max_retries")
	timeoutSecs := v.GetInt("idle_timeout_secs")
	if timeoutSecs <= 0 {
		timeoutSecs = 10
	}
	var retries atomic.Int64

	vp := vadproc.New(vadproc.Config{VAD: vd})
	tp := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: turns.DefaultStartStrategies(),
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr})},
		},
		IdleTimeout: time.Duration(timeoutSecs) * time.Second,
		OnIdle: func(ctx context.Context, c *turns.UserIdleController) error {
			n := int(retries.Add(1))
			if maxRetries > 0 && n >= maxRetries {
				slog.Info("caller idle: ending call", "retry", n)
				return c.Push(ctx, frames.NewEndFrame(), processor.Downstream)
			}
			slog.Info("caller idle: nudging", "retry", n)
			return c.Push(ctx, frames.NewTTSSpeakFrame("Are you still there?"), processor.Downstream)
		},
	})
	return vp, tp
}
