package grok

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

// TestTTSConfigValidate pins which TTSConfig fields the provider requires and
// the ranges xAI accepts.
func TestTTSConfigValidate(t *testing.T) {
	slow, tooSlow := 0.7, 0.6
	level, tooHigh := 2, 3
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
		{Name: "speed in range", Cfg: TTSConfig{APIKey: "k", Speed: &slow}, Valid: true},
		{Name: "speed below range", Cfg: TTSConfig{APIKey: "k", Speed: &tooSlow}, Valid: false},
		{Name: "latency level in range", Cfg: TTSConfig{APIKey: "k", OptimizeStreamingLatency: &level}, Valid: true},
		{Name: "latency level above range", Cfg: TTSConfig{APIKey: "k", OptimizeStreamingLatency: &tooHigh}, Valid: false},
	})
}

// TestNewTTSServices checks each constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewTTSServices(t *testing.T) {
	providertest.Service(t, "XAITTS", NewTTS(TTSConfig{APIKey: "k"}))
	providertest.Service(t, "XAIHTTPTTS", NewHTTPTTS(TTSConfig{APIKey: "k"}))
}

// TestXAILanguage checks the regional codes xAI names explicitly, a regional
// variant xAI does not list resolving through its base language, and detection
// for the zero value.
func TestXAILanguage(t *testing.T) {
	cases := []struct {
		in   language.Language
		want string
	}{
		{language.Arabic, "ar-EG"},
		{language.ArabicSA, "ar-SA"},
		{language.ArabicAE, "ar-AE"},
		{language.ArabicMA, "ar-EG"},
		{language.Spanish, "es-ES"},
		{language.SpanishES, "es-ES"},
		{language.SpanishMX, "es-MX"},
		{language.SpanishUS, "es-ES"},
		{language.Portuguese, "pt-PT"},
		{language.PortuguesePT, "pt-PT"},
		{language.PortugueseBR, "pt-BR"},
		{language.English, "en"},
		{language.EnglishGB, "en"},
		{language.FrenchCA, "fr"},
		{language.Japanese, "ja"},
		{"", ttsLanguageAuto},
	}
	for _, c := range cases {
		if got := xaiLanguage(c.in); got != c.want {
			t.Errorf("xaiLanguage(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTTSSynthesizerDescription checks both transports report the rate they
// emit and the voice the synthesis is billed against.
func TestTTSSynthesizerDescription(t *testing.T) {
	cfg := TTSConfig{APIKey: "k", Voice: "atlas", SampleRate: 16000}
	ws := newTTSSynthesizer(cfg)
	batch := &httpTTSSynthesizer{cfg: applyTTSDefaults(cfg, defaultTTSHTTPURL)}

	if got := ws.SampleRate(); got != 16000 {
		t.Errorf("streaming SampleRate() = %d, want 16000", got)
	}
	if got := batch.SampleRate(); got != 16000 {
		t.Errorf("batch SampleRate() = %d, want 16000", got)
	}
	if got := ws.Metadata().VoiceID; got != "atlas" {
		t.Errorf("streaming Metadata().VoiceID = %q, want atlas", got)
	}
	if got := batch.Metadata().VoiceID; got != "atlas" {
		t.Errorf("batch Metadata().VoiceID = %q, want atlas", got)
	}
}

// TestTTSEndpoint checks the audio parameters baked into the session handshake.
func TestTTSEndpoint(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		q := parseQuery(t, newTTSSynthesizer(TTSConfig{APIKey: "k"}).buildURL())

		want := map[string]string{
			"voice":           defaultVoice,
			"language":        ttsLanguageAuto,
			"codec":           ttsCodec,
			"sample_rate":     "24000",
			"with_timestamps": "true",
		}
		for key, val := range want {
			if got := q.Get(key); got != val {
				t.Errorf("%s = %q, want %q", key, got, val)
			}
		}
		for _, key := range []string{"speed", "optimize_streaming_latency", "text_normalization"} {
			if q.Has(key) {
				t.Errorf("%s = %q, want it omitted when unset", key, q.Get(key))
			}
		}
	})

	t.Run("optional settings", func(t *testing.T) {
		speed, level, normalize, timestamps := 1.25, 1, false, false
		q := parseQuery(t, newTTSSynthesizer(TTSConfig{
			APIKey:                   "k",
			Voice:                    "atlas",
			Language:                 language.PortugueseBR,
			SampleRate:               16000,
			Speed:                    &speed,
			OptimizeStreamingLatency: &level,
			TextNormalization:        &normalize,
			WordTimestamps:           &timestamps,
		}).buildURL())

		want := map[string]string{
			"voice":                      "atlas",
			"language":                   "pt-BR",
			"sample_rate":                "16000",
			"speed":                      "1.25",
			"optimize_streaming_latency": "1",
			"text_normalization":         "false",
			"with_timestamps":            "false",
		}
		for key, val := range want {
			if got := q.Get(key); got != val {
				t.Errorf("%s = %q, want %q", key, got, val)
			}
		}
	})
}

// TestHTTPTTSSynthesize checks the batch service's request shape and that it
// streams the response body downstream.
func TestHTTPTTSSynthesize(t *testing.T) {
	type request struct {
		Text     string `json:"text"`
		VoiceID  string `json:"voice_id"`
		Language string `json:"language"`
		Format   struct {
			Codec      string `json:"codec"`
			SampleRate int    `json:"sample_rate"`
		} `json:"output_format"`
		Speed *float64 `json:"speed"`
	}

	var got request
	var auth, contentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		contentType = r.Header.Get("Content-Type")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		_, _ = w.Write([]byte{9, 8, 7, 6})
	}))
	defer srv.Close()

	speed := 1.1
	s := &httpTTSSynthesizer{
		cfg: applyTTSDefaults(TTSConfig{
			APIKey:   "test-key",
			URL:      srv.URL,
			Language: language.English,
			Speed:    &speed,
		}, defaultTTSHTTPURL),
		http: srv.Client(),
	}

	var pcm []byte
	if err := runPCM(s, context.Background(), "hello", func(chunk []byte) error {
		pcm = append(pcm, chunk...)
		return nil
	}); err != nil {
		t.Fatalf("Synthesize: %v", err)
	}

	if string(pcm) != "\x09\x08\x07\x06" {
		t.Errorf("emitted %q, want the response body", pcm)
	}
	if auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", auth)
	}
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", contentType)
	}
	if got.Text != "hello" || got.VoiceID != defaultVoice || got.Language != "en" {
		t.Errorf("request = %+v, want the text, default voice and language", got)
	}
	if got.Format.Codec != ttsCodec || got.Format.SampleRate != defaultTTSSampleRate {
		t.Errorf("output_format = %+v, want raw PCM at the default rate", got.Format)
	}
	if got.Speed == nil || *got.Speed != speed {
		t.Errorf("speed = %v, want %v", got.Speed, speed)
	}
}

// TestHTTPTTSErrorStatus checks a non-200 response is reported rather than
// emitted as audio.
func TestHTTPTTSErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("bad key"))
	}))
	defer srv.Close()

	s := &httpTTSSynthesizer{
		cfg:  applyTTSDefaults(TTSConfig{APIKey: "k", URL: srv.URL}, defaultTTSHTTPURL),
		http: srv.Client(),
	}
	err := runPCM(s, context.Background(), "hello", func([]byte) error { return nil })
	if err == nil {
		t.Fatal("Synthesize() = nil error, want the status reported")
	}
	if !strings.Contains(err.Error(), "bad key") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// TestNewTTSWordTimestampPath checks the word-aligned path is taken by default
// and can be turned off.
func TestNewTTSWordTimestampPath(t *testing.T) {
	off := false
	cases := []struct {
		name string
		cfg  TTSConfig
		want bool
	}{
		{"on by default", TTSConfig{APIKey: "k"}, true},
		{"explicitly off", TTSConfig{APIKey: "k", WordTimestamps: &off}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.cfg.wordTimestamps(); got != c.want {
				t.Errorf("wordTimestamps() = %v, want %v", got, c.want)
			}
			_, timed := newTTSSynthesizer(c.cfg).driven().(tts.WordTimestamps)
			if timed != c.want {
				t.Errorf("the driven synthesizer reports word timings: %v, want %v", timed, c.want)
			}
		})
	}
}

// runPCM drives a synthesizer the way the base does, handing back the raw audio
// it yields.
func runPCM(s tts.Synthesizer, ctx context.Context, text string, emit func(pcm []byte) error) error {
	return s.RunTTS(ctx, text, "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			return emit(af.Audio)
		}
		return nil
	})
}
