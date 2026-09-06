package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

// sttSession is what the fake transcription endpoint saw.
type sttSession struct {
	query  url.Values
	header http.Header
	sent   chan map[string]any
}

// sttServer starts a fake Sarvam transcription endpoint that replays scripted
// messages and records what the client sends it.
func sttServer(t *testing.T, reply []map[string]any) (endpoint string, got *sttSession) {
	t.Helper()
	got = &sttSession{sent: make(chan map[string]any, 8)}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.query = r.URL.Query()
		got.header = r.Header.Clone()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		for _, m := range reply {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) != nil {
				continue
			}
			select {
			case got.sent <- msg:
			default:
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// connectorFor builds a connector for model, with the defaults NewSTT applies.
func connectorFor(cfg STTConfig) *connector {
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.InputAudioCodec == "" {
		cfg.InputAudioCodec = defaultInputCodec
	}
	mc, ok := sttModelConfigs[cfg.Model]
	if !ok {
		mc = sttModelConfigs[defaultSTTModel]
	}
	return &connector{cfg: cfg, mc: mc}
}

// TestSTTConfigValidate pins the credential and the models the service accepts.
func TestSTTConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: STTConfig{APIKey: "k"}, Valid: true},
		{Name: "known model", Cfg: STTConfig{APIKey: "k", Model: "saaras:v3"}, Valid: true},
		{Name: "sunset model", Cfg: STTConfig{APIKey: "k", Model: "saaras:v2.5"}, Valid: false},
		{Name: "unknown model", Cfg: STTConfig{APIKey: "k", Model: "saaras:v9"}, Valid: false},
		{Name: "known mode", Cfg: STTConfig{APIKey: "k", Mode: "translit"}, Valid: true},
		{Name: "unknown mode", Cfg: STTConfig{APIKey: "k", Mode: "shout"}, Valid: false},
	})
}

// TestNewSTT checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewSTT(t *testing.T) {
	providertest.Service(t, "SarvamSTT", NewSTT(STTConfig{APIKey: "k"}))
	// A model outside the table falls back to the default's capabilities rather
	// than to a zero one that would silently drop every parameter.
	providertest.Service(t, "SarvamSTT", NewSTT(STTConfig{APIKey: "k", Model: "saaras:v9"}))
}

// TestSTTEndpointIsTheTranscriptionOne checks every supported model is dialed at
// the transcription endpoint. The translation endpoint served only the model
// Sarvam has sunset; translation is now a mode of the transcribing models.
func TestSTTEndpointIsTheTranscriptionOne(t *testing.T) {
	for _, model := range []string{"saaras:v3", "saaras:v4"} {
		if got := connectorFor(STTConfig{APIKey: "k", Model: model}).endpoint(); got != transcribeURL {
			t.Errorf("model %q dials %q, want the transcription endpoint", model, got)
		}
	}
}

// TestSTTQueryDefaults checks what an unconfigured session asks for. The flush
// signal is on because the pipeline's own VAD decides when the turn ends, and
// the server has to honor the flush that follows it.
func TestSTTQueryDefaults(t *testing.T) {
	q := connectorFor(STTConfig{APIKey: "k"}).query(16000)

	if q.Get("model") != defaultSTTModel {
		t.Errorf("model = %q, want %q", q.Get("model"), defaultSTTModel)
	}
	if q.Get("sample_rate") != "16000" {
		t.Errorf("sample_rate = %q, want the rate the transport runs at", q.Get("sample_rate"))
	}
	if q.Get("flush_signal") != "true" {
		t.Errorf("flush_signal = %q, want true so the pipeline's VAD can close the turn", q.Get("flush_signal"))
	}
	// Left unset so the server's own defaults apply.
	for _, k := range []string{"vad_signals", "high_vad_sensitivity", "prompt"} {
		if q.Has(k) {
			t.Errorf("%s was sent for an unset config: %q", k, q.Get(k))
		}
	}
	if q.Get("language_code") != "unknown" {
		t.Errorf("language_code = %q, want the model's own default", q.Get("language_code"))
	}
	if q.Get("mode") != "transcribe" {
		t.Errorf("mode = %q, want the model's own default", q.Get("mode"))
	}
}

// TestSTTQueryVADSignals checks the flush signal is dropped once the caller asks
// Sarvam to detect speech itself: the two ways of closing a turn are exclusive.
func TestSTTQueryVADSignals(t *testing.T) {
	on, off := true, false

	q := connectorFor(STTConfig{APIKey: "k", VADSignals: &on}).query(16000)
	if q.Has("flush_signal") {
		t.Errorf("flush_signal = %q, want it dropped when Sarvam detects speech itself", q.Get("flush_signal"))
	}
	if q.Get("vad_signals") != "true" {
		t.Errorf("vad_signals = %q, want true", q.Get("vad_signals"))
	}

	// Explicitly off is not the same as unset: it is sent, and the flush stays.
	q = connectorFor(STTConfig{APIKey: "k", VADSignals: &off, HighVADSensitivity: &on}).query(16000)
	if q.Get("vad_signals") != "false" {
		t.Errorf("vad_signals = %q, want an explicit false to be sent", q.Get("vad_signals"))
	}
	if q.Get("flush_signal") != "true" {
		t.Errorf("flush_signal = %q, want it kept", q.Get("flush_signal"))
	}
	if q.Get("high_vad_sensitivity") != "true" {
		t.Errorf("high_vad_sensitivity = %q, want true", q.Get("high_vad_sensitivity"))
	}
}

// TestSTTQueryModelCapabilities checks the parameters both supported models
// understand reach the connection: the fine-grained VAD controls, the mode and
// the language.
func TestSTTQueryModelCapabilities(t *testing.T) {
	threshold := 0.6
	frames := 3
	cfg := STTConfig{
		APIKey:                  "k",
		PositiveSpeechThreshold: &threshold,
		MinSpeechFrames:         &frames,
	}

	for _, model := range []string{"saaras:v3", "saaras:v4"} {
		q := connectorFor(withModel(cfg, model)).query(16000)
		if q.Get("positive_speech_threshold") != "0.6" {
			t.Errorf("%s positive_speech_threshold = %q, want 0.6", model, q.Get("positive_speech_threshold"))
		}
		if q.Get("min_speech_frames") != "3" {
			t.Errorf("%s min_speech_frames = %q, want 3", model, q.Get("min_speech_frames"))
		}
		if q.Get("language_code") != "unknown" {
			t.Errorf("%s language_code = %q, want the model's own default", model, q.Get("language_code"))
		}
		if q.Get("mode") != "transcribe" {
			t.Errorf("%s mode = %q, want the model's own default", model, q.Get("mode"))
		}
		// The prompt was only ever honored by the model Sarvam has sunset.
		if q.Has("prompt") {
			t.Errorf("%s sent a prompt: %q", model, q.Get("prompt"))
		}
	}
}

func withModel(cfg STTConfig, model string) STTConfig {
	cfg.Model = model
	return cfg
}

// TestSTTModeOverride checks a configured mode wins over the model's default.
func TestSTTModeOverride(t *testing.T) {
	c := connectorFor(STTConfig{APIKey: "k", Mode: "translit"})
	if got := c.mode(); got != "translit" {
		t.Errorf("mode() = %q, want the configured mode", got)
	}
	if got := c.query(16000).Get("mode"); got != "translit" {
		t.Errorf("mode = %q, want the configured mode", got)
	}
}

// TestSarvamSTTLanguageCode pins the regional codes. Sarvam names every language
// by its Indian region, and Odia is "od-IN" rather than the "or" its ISO code
// would suggest.
func TestSarvamSTTLanguageCode(t *testing.T) {
	cases := map[language.Language]string{
		language.Language("as"): "as-IN",
		language.Language("bn"): "bn-IN",
		language.English:        "en-IN",
		language.EnglishGB:      "en-IN",
		language.Language("gu"): "gu-IN",
		language.Language("hi"): "hi-IN",
		language.Language("kn"): "kn-IN",
		language.Language("ml"): "ml-IN",
		language.Language("mr"): "mr-IN",
		language.Language("or"): "od-IN",
		language.Language("pa"): "pa-IN",
		language.Language("ta"): "ta-IN",
		language.Language("te"): "te-IN",
		language.French:         "",
		language.Language(""):   "",
	}
	for in, want := range cases {
		if got := sarvamSTTLanguageCode(in); got != want {
			t.Errorf("sarvamSTTLanguageCode(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestSTTLanguageString checks a supported language wins over the model default
// and an unsupported one falls back to it.
func TestSTTLanguageString(t *testing.T) {
	if got := connectorFor(STTConfig{APIKey: "k", Language: language.Language("hi")}).languageString(); got != "hi-IN" {
		t.Errorf("languageString() = %q, want hi-IN", got)
	}
	if got := connectorFor(STTConfig{APIKey: "k", Language: language.French}).languageString(); got != "unknown" {
		t.Errorf("languageString() = %q, want the model default", got)
	}
}

// TestSTTEncoding checks the codec is labeled the way Sarvam expects, with the
// media type added only when the caller left it off.
func TestSTTEncoding(t *testing.T) {
	if got := connectorFor(STTConfig{APIKey: "k"}).encoding(); got != "audio/wav" {
		t.Errorf("encoding() = %q, want audio/wav", got)
	}
	if got := connectorFor(STTConfig{APIKey: "k", InputAudioCodec: "audio/pcm"}).encoding(); got != "audio/pcm" {
		t.Errorf("encoding() = %q, want the codec left as given", got)
	}
}

// openSTT opens a session against the fake endpoint. The service picks its own
// endpoint from the model rather than from configuration (as upstream does), so
// the connection is dialed here the way Connect dials it and handed to the same
// stream.
func openSTT(t *testing.T, c *connector, endpoint string) stt.Stream {
	t.Helper()
	header := http.Header{}
	header.Set("api-subscription-key", c.cfg.APIKey)

	ctx, cancel := context.WithCancel(t.Context())
	conn, err := wsutil.Dial(ctx, endpoint+"?"+c.query(16000).Encode(), header, readLimitSTT)
	if err != nil {
		cancel()
		t.Fatalf("dialing the session: %v", err)
	}
	s := &stream{conn: conn, ctx: ctx, encoding: c.encoding(), sampleRate: 16000}
	t.Cleanup(func() {
		cancel()
		_ = s.Close()
	})
	return s
}

// TestSTTSessionCarriesTheKeyAndParameters checks the session is authorized with
// Sarvam's subscription-key header and describes what it is about to send.
func TestSTTSessionCarriesTheKeyAndParameters(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	openSTT(t, connectorFor(STTConfig{APIKey: "test-key"}), endpoint)

	if h := got.header.Get("api-subscription-key"); h != "test-key" {
		t.Errorf("api-subscription-key = %q, want the configured key", h)
	}
	if got.query.Get("model") != defaultSTTModel {
		t.Errorf("model = %q, want %q", got.query.Get("model"), defaultSTTModel)
	}
	if got.query.Get("sample_rate") != "16000" {
		t.Errorf("sample_rate = %q, want the rate the session opened at", got.query.Get("sample_rate"))
	}
}

// TestSTTStreamSendsAudioAsJSON checks the PCM goes out base64 inside a JSON
// message, described by the encoding and rate it was captured at. Sarvam takes
// audio as text rather than as binary frames.
func TestSTTStreamSendsAudioAsJSON(t *testing.T) {
	endpoint, got := sttServer(t, nil)
	s := openSTT(t, connectorFor(STTConfig{APIKey: "k"}), endpoint)

	pcm := []byte{1, 2, 3, 4}
	if err := s.Send(pcm); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case msg := <-got.sent:
		audio, ok := msg["audio"].(map[string]any)
		if !ok {
			t.Fatalf("message = %v, want an audio message", msg)
		}
		if audio["data"] != base64.StdEncoding.EncodeToString(pcm) {
			t.Errorf("data = %v, want the base64 PCM", audio["data"])
		}
		if audio["encoding"] != "audio/wav" {
			t.Errorf("encoding = %v, want audio/wav", audio["encoding"])
		}
		if audio["sample_rate"] != float64(16000) {
			t.Errorf("sample_rate = %v, want the rate the session opened at", audio["sample_rate"])
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the endpoint never received the audio")
	}
}

// TestSTTStreamRecv checks a transcript is surfaced as a final, end-of-turn
// result: this service returns whole utterances, so the transcript that arrives
// is the turn. Everything else the session sends is skipped.
func TestSTTStreamRecv(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{"type": "events", "data": map[string]any{"signal_type": "START_SPEECH"}},
		{"type": "data", "data": map[string]any{"transcript": "   "}},
		{"type": "data", "data": map[string]any{"transcript": "  hello there  ", "language_code": "hi-IN"}},
	})
	s := openSTT(t, connectorFor(STTConfig{APIKey: "k"}), endpoint)

	res, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 {
		t.Fatalf("results = %+v, want the one transcript", res)
	}
	if res[0].Text != "hello there" {
		t.Errorf("text = %q, want it trimmed", res[0].Text)
	}
	if !res[0].Final || !res[0].EndOfTurn {
		t.Errorf("result = %+v, want a final that closes the turn", res[0])
	}
	if res[0].Language != "hi-IN" {
		t.Errorf("language = %q, want the recognized language", res[0].Language)
	}
}

// TestSTTStreamRecvServerError checks an error message is reported rather than
// leaving the pipeline waiting for a transcript that will not come.
func TestSTTStreamRecvServerError(t *testing.T) {
	endpoint, _ := sttServer(t, []map[string]any{
		{"type": "error", "data": map[string]any{"message": "unsupported sample rate"}},
	})
	s := openSTT(t, connectorFor(STTConfig{APIKey: "k"}), endpoint)

	_, err := s.Recv()
	if !errors.Is(err, errSTTServer) {
		t.Fatalf("Recv error = %v, want errSTTServer", err)
	}
	if !strings.Contains(err.Error(), "unsupported sample rate") {
		t.Errorf("error = %v, want it to carry the server's message", err)
	}
}
