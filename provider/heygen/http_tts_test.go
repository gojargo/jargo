package heygen

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

// TestHTTPTTSConfigValidate pins which fields the service requires.
func TestHTTPTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: HTTPTTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: HTTPTTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewHTTPTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewHTTPTTS(t *testing.T) {
	providertest.Service(t, "HeyGenHTTPTTS", NewHTTPTTS(HTTPTTSConfig{APIKey: "k"}))
}

// httpRequest is one request the stand-in HeyGen was sent.
type httpRequest struct {
	header http.Header
	body   map[string]any
}

// httpServer stands in for HeyGen's streaming speech endpoint. It answers with
// audio, split mid-sample to check that the frames stay sample-aligned, or
// with an error envelope when status is not 200.
type httpServer struct {
	mu       sync.Mutex
	requests []httpRequest
	url      string
}

func newHTTPServer(t *testing.T, audio []byte, status int, message string) *httpServer {
	t.Helper()
	s := &httpServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != httpTTSPath {
			http.NotFound(w, r)
			return
		}
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(data, &body)
		s.mu.Lock()
		s.requests = append(s.requests, httpRequest{header: r.Header.Clone(), body: body})
		s.mu.Unlock()
		if status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
				"code": "rate_limit_exceeded", "message": message, "request_id": "req-9",
			}})
			return
		}
		w.Header().Set("Content-Type", "audio/pcm")
		w.Header().Set("X-Request-Id", "req-123")
		split := min(2047, len(audio))
		_, _ = w.Write(audio[:split])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(10 * time.Millisecond)
		_, _ = w.Write(audio[split:])
	}))
	t.Cleanup(srv.Close)
	s.url = srv.URL
	return s
}

func (s *httpServer) all() []httpRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]httpRequest(nil), s.requests...)
}

// httpTTS builds the HTTP service against the stand-in.
func httpTTS(s *httpServer, cfg HTTPTTSConfig) *tts.Base {
	cfg.APIKey = "test-key"
	cfg.BaseURL = s.url + "/"
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 24000
	}
	return NewHTTPTTS(cfg).Base
}

// The service sends the documented request and emits sample-aligned PCM frames.
func TestHTTPTTSStreamsPCM(t *testing.T) {
	pcm := bytes.Repeat([]byte{0, 1, 2, 3}, 1024)
	server := newHTTPServer(t, pcm, http.StatusOK, "")

	down, _ := runTest(t, httpTTS(server, HTTPTTSConfig{}), speak("Hello from HeyGen."))

	if count[*frames.TTSStartedFrame](down) == 0 || count[*frames.TTSStoppedFrame](down) == 0 {
		t.Fatal("the synthesis was not bracketed by started and stopped frames")
	}
	audio := audioFrames(down)
	if len(audio) == 0 {
		t.Fatal("no audio")
	}
	var joined []byte
	for _, f := range audio {
		if f.SampleRate != 24000 || f.NumChannels != 1 || len(f.Audio)%2 != 0 {
			t.Fatalf("frame at %d Hz, %d channels, %d bytes", f.SampleRate, f.NumChannels, len(f.Audio))
		}
		joined = append(joined, f.Audio...)
	}
	if !bytes.Equal(joined, pcm) {
		t.Fatalf("emitted %d bytes, not the %d HeyGen sent", len(joined), len(pcm))
	}

	requests := server.all()
	if len(requests) != 1 {
		t.Fatalf("sent %d requests, want 1", len(requests))
	}
	if got := requests[0].header.Get("X-Api-Key"); got != "test-key" {
		t.Fatalf("x-api-key = %q", got)
	}
	want := map[string]any{
		"text":          "Hello from HeyGen.",
		"voice_id":      defaultVoiceID,
		"model":         "heygen-voice-1-turbo",
		"output_format": map[string]any{"encoding": "pcm", "sample_rate_hertz": float64(24000)},
	}
	if !reflect.DeepEqual(requests[0].body, want) {
		t.Fatalf("body = %v\nwant %v", requests[0].body, want)
	}
}

// Voice, model and language settings reach the request body.
func TestHTTPTTSSettingsInBody(t *testing.T) {
	server := newHTTPServer(t, make([]byte, 4096), http.StatusOK, "")

	runTest(t, httpTTS(server, HTTPTTSConfig{
		SampleRate: 16000, VoiceID: "my-voice", Model: "heygen-voice-1-turbo-free", Language: language.EnglishUS,
	}), speak("Hi."))

	body := server.all()[0].body
	format, _ := body["output_format"].(map[string]any)
	if body["voice_id"] != "my-voice" || body["model"] != "heygen-voice-1-turbo-free" || body["language"] != "en" ||
		format["sample_rate_hertz"] != float64(16000) {
		t.Fatalf("body = %v", body)
	}
}

// A rate HeyGen doesn't offer requests 24 kHz, warns once and resamples.
func TestHTTPTTSUnsupportedSampleRateIsResampled(t *testing.T) {
	server := newHTTPServer(t, make([]byte, 48000), http.StatusOK, "")
	logs := captureWarnings(t)

	down, _ := runTest(t, httpTTS(server, HTTPTTSConfig{SampleRate: 32000}), speak("Hi."), speak("Again."))

	for _, r := range server.all() {
		if format, _ := r.body["output_format"].(map[string]any); format["sample_rate_hertz"] != float64(24000) {
			t.Fatalf("requested %v, want 24000", format["sample_rate_hertz"])
		}
	}
	if len(server.all()) != 2 {
		t.Fatalf("sent %d requests, want 2", len(server.all()))
	}
	if n := strings.Count(logs(), "32000 Hz"); n != 1 {
		t.Fatalf("warned %d times about 32000 Hz, want once:\n%s", n, logs())
	}
	audio := audioFrames(down)
	if len(audio) == 0 {
		t.Fatal("no audio")
	}
	for _, f := range audio {
		if f.SampleRate != 32000 {
			t.Fatalf("a frame at %d Hz, want 32000", f.SampleRate)
		}
	}
}

// A non-200 response produces an ErrorFrame with HeyGen's code, message and
// request ID.
func TestHTTPTTSError(t *testing.T) {
	server := newHTTPServer(t, nil, http.StatusTooManyRequests, "Request limit reached.")

	_, up := runTest(t, httpTTS(server, HTTPTTSConfig{}), speak("Hi."))

	errs := errorFrames(up)
	if len(errs) == 0 {
		t.Fatal("no error reported")
	}
	msg := errs[0].Error
	if !strings.Contains(msg, "429") || !strings.Contains(msg, "rate_limit_exceeded: Request limit reached.") ||
		!strings.Contains(msg, "req-9") {
		t.Fatalf("error = %q", msg)
	}
}

// A language HeyGen doesn't speak is left out, so HeyGen detects the language.
func TestHTTPTTSUnsupportedLanguageIsNotSent(t *testing.T) {
	server := newHTTPServer(t, make([]byte, 4096), http.StatusOK, "")

	runTest(t, httpTTS(server, HTTPTTSConfig{Language: language.Norwegian}), speak("Hei."))

	if body := server.all()[0].body; body["language"] != nil {
		t.Fatalf("sent language %v, want none", body["language"])
	}
}
