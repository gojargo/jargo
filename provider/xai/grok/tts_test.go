package grok

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
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

// TestXAILanguage checks the regional codes xAI names explicitly, the base-code
// fallback for everything else, and detection for the zero value.
func TestXAILanguage(t *testing.T) {
	cases := []struct {
		in   language.Language
		want string
	}{
		{language.Arabic, "ar-EG"},
		{language.Spanish, "es-ES"},
		{language.SpanishES, "es-ES"},
		{language.SpanishMX, "es-MX"},
		{language.SpanishUS, "es"},
		{language.Portuguese, "pt-PT"},
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
	cfg := applyTTSDefaults(TTSConfig{APIKey: "k", Voice: "atlas", SampleRate: 16000}, defaultTTSWSURL)
	ws := &ttsSynthesizer{cfg: cfg}
	batch := &httpTTSSynthesizer{cfg: cfg}

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
		s := &ttsSynthesizer{cfg: applyTTSDefaults(TTSConfig{APIKey: "k"}, defaultTTSWSURL)}
		q := parseQuery(t, s.endpoint(true))

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
		speed, level, normalize := 1.25, 1, false
		s := &ttsSynthesizer{cfg: applyTTSDefaults(TTSConfig{
			APIKey:                   "k",
			Voice:                    "atlas",
			Language:                 language.PortugueseBR,
			SampleRate:               16000,
			Speed:                    &speed,
			OptimizeStreamingLatency: &level,
			TextNormalization:        &normalize,
		}, defaultTTSWSURL)}
		q := parseQuery(t, s.endpoint(false))

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

// ttsServer starts a fake streaming endpoint that records the client's messages
// and replies with events.
func ttsServer(t *testing.T, events []map[string]any) (endpoint string, sent <-chan string, auth func() string) {
	t.Helper()
	messages := make(chan string, 8)
	authCh := make(chan string, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case authCh <- r.Header.Get("Authorization"):
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The client sends the text delta then closes the utterance; reply only
		// once both have arrived, as the real service does.
		for range 2 {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			messages <- string(data)
		}
		for _, ev := range events {
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		// Keep reading so the library can answer the client's close handshake;
		// blocking here instead would stall every test on its close timeout.
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	return wsURL(srv.URL), messages, func() string {
		select {
		case v := <-authCh:
			return v
		default:
			return ""
		}
	}
}

// audioDelta builds an audio.delta event carrying pcm and optional timings.
func audioDelta(pcm []byte, chars []string, times [][]float64) map[string]any {
	ev := map[string]any{"type": "audio.delta"}
	if len(pcm) > 0 {
		ev["delta"] = base64.StdEncoding.EncodeToString(pcm)
	}
	if chars != nil {
		ev["audio_timestamps"] = map[string]any{"graph_chars": chars, "graph_times": times}
	}
	return ev
}

// TestTTSSynthesize checks the request the streaming service sends and the
// audio it emits.
func TestTTSSynthesize(t *testing.T) {
	endpoint, sent, auth := ttsServer(t, []map[string]any{
		audioDelta([]byte{1, 2, 3, 4}, nil, nil),
		audioDelta([]byte{5, 6}, nil, nil),
		{"type": "audio.done"},
	})

	s := &ttsSynthesizer{cfg: applyTTSDefaults(TTSConfig{APIKey: "test-key", URL: endpoint}, defaultTTSWSURL)}
	var got []byte
	err := s.Synthesize(context.Background(), "hello there", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if string(got) != "\x01\x02\x03\x04\x05\x06" {
		t.Errorf("emitted %q, want the concatenated PCM chunks", got)
	}
	if a := auth(); a != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", a)
	}

	delta := <-sent
	if !strings.Contains(delta, `"type":"text.delta"`) || !strings.Contains(delta, "hello there") {
		t.Errorf("first message = %q, want the text delta", delta)
	}
	if done := <-sent; !strings.Contains(done, `"type":"text.done"`) {
		t.Errorf("second message = %q, want the utterance to be closed", done)
	}
}

// TestTTSSynthesizeTimed checks per-character timings are assembled into words,
// including a word split across two payloads and a trailing word with no
// terminating space.
func TestTTSSynthesizeTimed(t *testing.T) {
	endpoint, _, _ := ttsServer(t, []map[string]any{
		// "hi the" arrives first, leaving "the" carried over mid-word.
		audioDelta([]byte{1, 2}, []string{"h", "i", " ", "t", "h", "e"}, [][]float64{
			{0.0, 0.1}, {0.1, 0.2}, {0.2, 0.3}, {0.3, 0.4}, {0.4, 0.5}, {0.5, 0.6},
		}),
		// "re, pal" completes "there," then leaves "pal" unterminated.
		audioDelta(nil, []string{"r", "e", ",", " ", "p", "a", "l"}, [][]float64{
			{0.6, 0.7}, {0.7, 0.8}, {0.8, 0.9}, {0.9, 1.0}, {1.0, 1.1}, {1.1, 1.2}, {1.2, 1.3},
		}),
		{"type": "audio.done"},
	})

	s := &timedTTSSynthesizer{ttsSynthesizer: &ttsSynthesizer{
		cfg: applyTTSDefaults(TTSConfig{APIKey: "k", URL: endpoint}, defaultTTSWSURL),
	}}

	type timed struct {
		text   string
		offset float64
	}
	var words []timed
	err := s.SynthesizeTimed(context.Background(), "hi there, pal",
		func([]byte) error { return nil },
		func(text string, offset float64) error {
			words = append(words, timed{text, offset})
			return nil
		})
	if err != nil {
		t.Fatalf("SynthesizeTimed: %v", err)
	}

	want := []timed{{"hi", 0.0}, {"there,", 0.3}, {"pal", 1.0}}
	if len(words) != len(want) {
		t.Fatalf("words = %+v, want %+v", words, want)
	}
	for i, w := range want {
		if words[i].text != w.text || words[i].offset != w.offset {
			t.Errorf("word %d = %+v, want %+v", i, words[i], w)
		}
	}
}

// TestTTSTimingLengthMismatch checks a malformed timing payload fails loudly
// rather than silently misaligning every word after it.
func TestTTSTimingLengthMismatch(t *testing.T) {
	endpoint, _, _ := ttsServer(t, []map[string]any{
		audioDelta(nil, []string{"h", "i"}, [][]float64{{0.0, 0.1}}),
	})

	s := &timedTTSSynthesizer{ttsSynthesizer: &ttsSynthesizer{
		cfg: applyTTSDefaults(TTSConfig{APIKey: "k", URL: endpoint}, defaultTTSWSURL),
	}}
	err := s.SynthesizeTimed(context.Background(), "hi",
		func([]byte) error { return nil },
		func(string, float64) error { return nil })
	if err == nil {
		t.Fatal("SynthesizeTimed() = nil error, want the length mismatch reported")
	}
	if !strings.Contains(err.Error(), "timing length mismatch") {
		t.Errorf("error = %v, want it to name the mismatch", err)
	}
}

// TestTTSServerError surfaces a server-reported failure to the caller.
func TestTTSServerError(t *testing.T) {
	endpoint, _, _ := ttsServer(t, []map[string]any{
		{"type": "error", "message": "unknown voice"},
	})

	s := &ttsSynthesizer{cfg: applyTTSDefaults(TTSConfig{APIKey: "k", URL: endpoint}, defaultTTSWSURL)}
	err := s.Synthesize(context.Background(), "hello", func([]byte) error { return nil })
	if err == nil {
		t.Fatal("Synthesize() = nil error, want the server error surfaced")
	}
	if !strings.Contains(err.Error(), "unknown voice") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}
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
	if err := s.Synthesize(context.Background(), "hello", func(chunk []byte) error {
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
	err := s.Synthesize(context.Background(), "hello", func([]byte) error { return nil })
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
		})
	}
}
