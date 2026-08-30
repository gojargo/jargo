// Command outbound is an outbound telephony voice agent: it places a phone call
// over Twilio, talks to whoever answers, collects a few details and hangs up.
//
// The program starts a small HTTP server that exposes two endpoints — /twiml
// (the call's instructions) and /ws (the bidirectional media stream) — then asks
// Twilio's REST API to dial the target number. When the call connects, Twilio
// fetches /twiml, opens a μ-law 8 kHz media stream to wss://<public_host>/ws, and
// an STT -> LLM -> TTS pipeline runs the conversation. The assistant gathers the
// caller's full name, the reason for the call and a callback preference, records
// them through a tool call, says goodbye and ends the call; the process then
// exits.
//
// The server must be reachable from the public internet so Twilio can fetch
// /twiml and open the WebSocket. In development, expose it with a tunnel (for
// example `ngrok http 8080`) and set public_host to the tunnel host.
//
// Configuration comes from a YAML file and/or the environment; environment
// variables override the file. Point at the file with -config (see
// config.example.yaml for the full schema). The keys are:
//
//	public_host          publicly reachable host, no scheme (e.g. abc123.ngrok-free.app)
//	addr                 listen address (optional, default :8080)
//	twilio.account_sid   Twilio account SID
//	twilio.auth_token    Twilio auth token
//	twilio.from_number   caller ID, E.164 (a Twilio number you own)
//	twilio.to_number     number to call, E.164
//	deepgram.api_key     speech-to-text
//	anthropic.api_key    LLM
//	elevenlabs.api_key   text-to-speech
//
// Every key has an environment equivalent: uppercase it and turn dots into
// underscores, for example TWILIO_ACCOUNT_SID or PUBLIC_HOST.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/twilio"
	"github.com/spf13/viper"
)

// phoneSampleRate is Twilio Media Streams' fixed μ-law rate. The whole pipeline
// runs at 8 kHz so no extra resampling is needed.
const phoneSampleRate = 8000

// readHeaderTimeout bounds how long the demo server waits for request headers.
const readHeaderTimeout = 10 * time.Second

// systemPrompt drives the assistant: a short courtesy call that gathers three
// details and records them with the record_info tool.
const systemPrompt = "Tu es un assistant vocal courtois au téléphone. Tu appelles la personne " +
	"pour recueillir trois informations : son nom complet, le motif de son appel ou de sa demande, " +
	"et sa préférence de rappel (par email ou par téléphone, avec la coordonnée correspondante). " +
	"Pose une seule question à la fois, avec des phrases courtes et chaleureuses. Quand tu as " +
	"obtenu les trois informations, appelle immédiatement l'outil record_info, sans rien dire " +
	"d'autre. Une fois l'outil exécuté, remercie brièvement la personne et termine par une courte " +
	"formule de politesse."

// greeting is the first thing the assistant says when the call connects; the bot
// speaks first because it is the caller.
const greeting = "Bonjour, je suis un assistant vocal et je vous appelle pour recueillir quelques " +
	"informations. Pour commencer, pouvez-vous me donner votre nom complet ?"

//nolint:gochecknoglobals // sentinel error
var errMissingConfig = errors.New("missing required configuration")

//nolint:gochecknoglobals // sentinel error
var errTwilioAPI = errors.New("twilio calls api request failed")

func main() {
	cfgPath := flag.String("config", "", "path to a YAML config file (optional; environment variables also work)")
	flag.Parse()
	if err := run(*cfgPath); err != nil {
		log.Fatal(err)
	}
}

func run(cfgPath string) error {
	v, err := loadConfig(cfgPath)
	if err != nil {
		return err
	}
	if err = requireKeys(v,
		"public_host",
		"twilio.account_sid", "twilio.auth_token", "twilio.from_number", "twilio.to_number",
		"deepgram.api_key", "anthropic.api_key", "elevenlabs.api_key",
	); err != nil {
		return err
	}

	twiml := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>`+
		`<Response><Connect><Stream url="wss://%s/ws"/></Connect></Response>`, v.GetString("public_host"))

	var once sync.Once
	callDone := make(chan struct{})
	signalDone := func() { once.Do(func() { close(callDone) }) }

	mux := http.NewServeMux()
	mux.HandleFunc("/twiml", func(w http.ResponseWriter, _ *http.Request) {
		slog.Info("twiml requested by Twilio")
		w.Header().Set("Content-Type", "text/xml")
		if _, werr := io.WriteString(w, twiml); werr != nil {
			slog.Error("write twiml", "err", werr)
		}
	})
	mux.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		defer signalDone()
		handleStream(w, r, v)
	})

	var lc net.ListenConfig
	ln, err := lc.Listen(context.Background(), "tcp", v.GetString("addr"))
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: readHeaderTimeout}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("http server", "err", err)
			signalDone()
		}
	}()

	slog.Info("placing outbound call",
		"to", v.GetString("twilio.to_number"), "from", v.GetString("twilio.from_number"))
	if err := originateCall(context.Background(), v); err != nil {
		return err
	}

	<-callDone // the call's pipeline finished (or the server failed)
	return nil
}

// loadConfig builds the Viper instance. Environment variables override the file
// (a key's env name is the uppercased key with dots turned into underscores). A
// config path is optional: with none set, configuration comes from the
// environment alone.
func loadConfig(path string) (*viper.Viper, error) {
	v := viper.New()
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()
	v.SetDefault("addr", ":8080")
	if path == "" {
		return v, nil
	}
	v.SetConfigFile(path)
	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}
	return v, nil
}

// requireKeys returns an error naming every key that resolves to an empty value.
func requireKeys(v *viper.Viper, keys ...string) error {
	var missing []string
	for _, k := range keys {
		if v.GetString(k) == "" {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("%w (set them in the -config YAML or the environment): %s",
			errMissingConfig, strings.Join(missing, ", "))
	}
	return nil
}

// originateCall asks Twilio's REST API to dial the target number and point the
// call at this server's /twiml endpoint. It uses net/http directly so the
// example needs no Twilio SDK dependency.
func originateCall(ctx context.Context, v *viper.Viper) error {
	sid := v.GetString("twilio.account_sid")
	endpoint := fmt.Sprintf("https://api.twilio.com/2010-04-01/Accounts/%s/Calls.json", sid)

	form := url.Values{}
	form.Set("From", v.GetString("twilio.from_number"))
	form.Set("To", v.GetString("twilio.to_number"))
	form.Set("Url", fmt.Sprintf("https://%s/twiml", v.GetString("public_host")))

	//nolint:gosec // Twilio REST endpoint; the host is constant and the account SID comes from config.
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("build call request: %w", err)
	}
	req.SetBasicAuth(sid, v.GetString("twilio.auth_token"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("place call: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if err != nil {
		return fmt.Errorf("read call response: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("%w: %s: %s", errTwilioAPI, resp.Status, body)
	}

	var created struct {
		SID string `json:"sid"`
	}
	_ = json.Unmarshal(body, &created)
	slog.Info("outbound call created", "call_sid", created.SID)
	return nil
}

func handleStream(w http.ResponseWriter, r *http.Request, v *viper.Viper) {
	ser := twilio.New(twilio.Config{
		AccountSID: v.GetString("twilio.account_sid"),
		AuthToken:  v.GetString("twilio.auth_token"),
	})

	params := wsserver.DefaultParams()
	params.AudioInSampleRate = phoneSampleRate
	params.AudioOutSampleRate = phoneSampleRate

	t, err := wsserver.Accept(w, r, ser, params)
	if err != nil {
		slog.Error("accept websocket", "err", err)
		return
	}
	runBot(t, v)
}

// runBot builds and runs the STT -> LLM -> TTS pipeline for one call. The LLM
// carries a record_info tool; when it has gathered every field it calls the
// tool, which logs the result and arms the hang-up. Turn detection is left to
// the STT's endpointing, so no VAD/ONNX runtime is required to run this example.
func runBot(t *wsserver.Transport, v *viper.Viper) {
	stt := deepgram.NewSTT(deepgram.Config{
		APIKey:   v.GetString("deepgram.api_key"),
		Language: language.French,
	})
	llm := anthropic.NewLLM(anthropic.Config{APIKey: v.GetString("anthropic.api_key")})
	tts := elevenlabs.NewTTS(elevenlabs.Config{
		APIKey:     v.GetString("elevenlabs.api_key"),
		SampleRate: phoneSampleRate,
		Language:   language.French,
	})

	var collected atomic.Bool
	llm.RegisterFunction("record_info", recordInfo(&collected))

	convo := frames.NewLLMContext(systemPrompt)
	convo.SetTools([]frames.Tool{{
		Name: "record_info",
		Description: "Enregistre les informations recueillies. À appeler une fois le nom complet, " +
			"le motif et la préférence de rappel obtenus.",
		Parameters: recordInfoSchema(),
	}})

	agg := aggregators.New(convo)
	task := pipeline.NewWorker(pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		tts,
		newHangUpAfterGoodbye(&collected),
		t.Output(),
		agg.Assistant(),
	), pipeline.WorkerConfig{
		Params: pipeline.Params{
			AudioInSampleRate:  phoneSampleRate,
			AudioOutSampleRate: phoneSampleRate,
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-t.Done() // the callee hung up or the socket closed
		cancel()
	}()

	task.QueueFrame(frames.NewTextFrame(greeting))
	slog.Info("outbound voice pipeline started")
	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("pipeline ended", "err", err)
	}
	slog.Info("outbound voice pipeline stopped")
}

// recordInfoSchema is the JSON-Schema for the record_info tool's arguments.
func recordInfoSchema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"full_name": {"type": "string", "description": "Nom complet de la personne."},
			"reason": {"type": "string", "description": "Motif de l'appel ou de la demande."},
			"callback_method": {"type": "string", "enum": ["email", "phone"], "description": "Moyen de rappel préféré."},
			"callback_value": {"type": "string", "description": "Adresse email ou numéro de téléphone pour le rappel."}
		},
		"required": ["full_name", "reason", "callback_method", "callback_value"]
	}`)
}

// recordInfo returns the handler for the record_info tool. It logs the captured
// fields and flips collected so the hang-up processor ends the call once the
// goodbye has been spoken.
func recordInfo(collected *atomic.Bool) llm.FunctionCallHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) error {
		var info struct {
			FullName       string `json:"full_name"`
			Reason         string `json:"reason"`
			CallbackMethod string `json:"callback_method"`
			CallbackValue  string `json:"callback_value"`
		}
		if err := json.Unmarshal(params.Arguments, &info); err != nil {
			return fmt.Errorf("decode record_info args: %w", err)
		}
		slog.Info("collected contact info",
			"full_name", info.FullName,
			"reason", info.Reason,
			"callback_method", info.CallbackMethod,
			"callback_value", info.CallbackValue,
		)
		collected.Store(true)
		return params.Result(ctx, `{"status": "saved"}`, nil)
	}
}

// hangUpAfterGoodbye ends the call once the assistant has finished speaking after
// the information was recorded. BotStoppedSpeakingFrame travels upstream from the
// output transport; placed just before the output, this processor sees it, and
// when collected is set it pushes an EndFrame downstream — which the Twilio
// serializer turns into a REST hang-up. The single-shot guard keeps the goodbye
// from being cut off by an earlier BotStoppedSpeakingFrame.
type hangUpAfterGoodbye struct {
	*processor.Base
	collected *atomic.Bool
	ended     atomic.Bool
}

func newHangUpAfterGoodbye(collected *atomic.Bool) *hangUpAfterGoodbye {
	h := &hangUpAfterGoodbye{collected: collected}
	h.Base = processor.New("HangUpAfterGoodbye", h)
	return h
}

func (h *hangUpAfterGoodbye) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := h.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := h.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.BotStoppedSpeakingFrame); ok &&
		h.collected.Load() && h.ended.CompareAndSwap(false, true) {
		slog.Info("information collected; ending call after goodbye")
		return h.PushFrame(ctx, frames.NewEndFrame(), processor.Downstream)
	}
	return nil
}
