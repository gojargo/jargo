package neuphonic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/utils/events"
)

// The audio the fake SSE endpoint streams, one chunk per event.
//
//nolint:gochecknoglobals // test fixtures, read-only
var (
	sseChunk1 = bytes.Repeat([]byte{0x01, 0x02}, 256)
	sseChunk2 = bytes.Repeat([]byte{0x03, 0x04}, 256)
)

// sseRequest is what the fake SSE endpoint was asked.
type sseRequest struct {
	path    string
	header  http.Header
	payload map[string]any
}

// sseServer is a fake Neuphonic SSE endpoint answering every request with
// status and body, and recording what it was asked.
type sseServer struct {
	status int
	body   string

	mu       sync.Mutex
	requests []sseRequest
}

func newSSEServer(t *testing.T, status int, body string) (*sseServer, string) {
	t.Helper()
	f := &sseServer{status: status, body: body}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	return f, srv.URL
}

func (f *sseServer) serve(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	payload := map[string]any{}
	_ = json.Unmarshal(raw, &payload)
	f.mu.Lock()
	f.requests = append(f.requests, sseRequest{path: r.URL.Path, header: r.Header.Clone(), payload: payload})
	f.mu.Unlock()
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(f.status)
	_, _ = io.WriteString(w, f.body)
}

func (f *sseServer) lastRequest(t *testing.T) sseRequest {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		t.Fatal("the endpoint was never asked")
	}
	return f.requests[len(f.requests)-1]
}

// audioEvent is an SSE event carrying audio.
func audioEvent(pcm []byte) string {
	return fmt.Sprintf("data: {\"status_code\": 200, \"data\": {\"audio\": %q, \"text\": \"hi\"}}\n\n",
		base64.StdEncoding.EncodeToString(pcm))
}

// synthesizeHTTP runs one synthesis and returns what was yielded.
func synthesizeHTTP(t *testing.T, s *httpSynthesizer, text string) []frames.Frame {
	t.Helper()
	var out []frames.Frame
	if err := s.RunTTS(t.Context(), text, "c1", func(f frames.Frame) error {
		out = append(out, f)
		return nil
	}); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return out
}

// pcmOf joins the audio of the yielded frames.
func pcmOf(t *testing.T, fs []frames.Frame, rate int) []byte {
	t.Helper()
	var pcm []byte
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			if a.SampleRate != rate || a.NumChannels != 1 {
				t.Errorf("audio at %d Hz on %d channels, want %d Hz mono", a.SampleRate, a.NumChannels, rate)
			}
			pcm = append(pcm, a.Audio...)
		}
	}
	return pcm
}

// errorsOf is the messages of the yielded error frames.
func errorsOf(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, ef.Error)
		}
	}
	return out
}

// The request goes to the SSE speak path for the language, authorized with the
// API key, and the audio of each event is yielded in order. Events carrying no
// audio and the closing [DONE] are skipped.
func TestHTTPTTSStreamsTheAudioOfEachEvent(t *testing.T) {
	body := audioEvent(sseChunk1) +
		"event: message\n\n" +
		"data: {\"status_code\": 200, \"data\": {\"text\": \"no audio\"}}\n\n" +
		audioEvent(sseChunk2) +
		"data: [DONE]\n\n"
	server, base := newSSEServer(t, http.StatusOK, body)
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: base + "/"})

	out := synthesizeHTTP(t, s, "Hello there.")

	if errs := errorsOf(out); len(errs) > 0 {
		t.Errorf("errors yielded: %v", errs)
	}
	if pcm := pcmOf(t, out, 22050); !bytes.Equal(pcm, slices.Concat(sseChunk1, sseChunk2)) {
		t.Errorf("yielded %d bytes of audio, want both chunks in order", len(pcm))
	}
	req := server.lastRequest(t)
	if req.path != "/sse/speak/en" {
		t.Errorf("path = %q, want the English SSE speak path", req.path)
	}
	if got := req.header.Get("X-API-KEY"); got != "test-key" {
		t.Errorf("X-API-KEY = %q", got)
	}
	if got := req.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q", got)
	}
	want := map[string]any{
		"text":          "Hello there.",
		"lang_code":     "en",
		"encoding":      "pcm_linear",
		"sampling_rate": 22050.0,
		"speed":         1.0,
	}
	if !mapsEqual(req.payload, want) {
		t.Errorf("payload = %v, want %v", req.payload, want)
	}
}

// The voice, the temperature and the IPA key are sent when set, and the
// language picks the path.
func TestHTTPTTSOptionalFields(t *testing.T) {
	server, base := newSSEServer(t, http.StatusOK, audioEvent(sseChunk1))
	_, s := newHTTPTTS(HTTPTTSConfig{
		APIKey:      "test-key",
		URL:         base,
		VoiceID:     "voice-1",
		Language:    language.Spanish,
		SampleRate:  16000,
		Encoding:    "pcm_mulaw",
		Speed:       1.2,
		Temperature: new(0.4),
		IPAKey:      "es-castilian",
	})

	out := synthesizeHTTP(t, s, "Hola.")

	if pcm := pcmOf(t, out, 16000); !bytes.Equal(pcm, sseChunk1) {
		t.Errorf("yielded %d bytes of audio, want the chunk", len(pcm))
	}
	req := server.lastRequest(t)
	if req.path != "/sse/speak/es" {
		t.Errorf("path = %q, want the Spanish SSE speak path", req.path)
	}
	want := map[string]any{
		"text":          "Hola.",
		"lang_code":     "es",
		"encoding":      "pcm_mulaw",
		"sampling_rate": 16000.0,
		"speed":         1.2,
		"voice_id":      "voice-1",
		"temperature":   0.4,
		"ipa_key":       "es-castilian",
	}
	if !mapsEqual(req.payload, want) {
		t.Errorf("payload = %v, want %v", req.payload, want)
	}
}

// A settings change applies from the next request.
func TestHTTPTTSSettingsApplyToTheNextRequest(t *testing.T) {
	server, base := newSSEServer(t, http.StatusOK, audioEvent(sseChunk1))
	b, s := newHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: base})

	if _, err := settings.Apply(s.live, &TTSSettings{
		Voice: settings.Set("voice-2"),
		Speed: settings.Set(0.8),
	}); err != nil {
		t.Fatalf("applying the settings: %v", err)
	}
	synthesizeHTTP(t, s, "Hello.")

	req := server.lastRequest(t)
	if req.payload["voice_id"] != "voice-2" || req.payload["speed"] != 0.8 {
		t.Errorf("payload = %v, want the new voice and speed", req.payload)
	}
	if b.CurrentSettings() != s.live {
		t.Error("the service does not expose the synthesizer's settings")
	}
}

// An event reporting errors is yielded as an error, and the events after it
// are still read.
func TestHTTPTTSErrorEventIsReportedAndTheStreamGoesOn(t *testing.T) {
	body := "data: {\"status_code\": 400, \"errors\": [\"voice not found\"]}\n\n" + audioEvent(sseChunk2)
	_, base := newSSEServer(t, http.StatusOK, body)
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: base})

	out := synthesizeHTTP(t, s, "Hello.")

	errs := errorsOf(out)
	if len(errs) != 1 || errs[0] != "Unknown error occurred: Neuphonic API error 400: [voice not found]" {
		t.Errorf("errors = %q, want the event's errors", errs)
	}
	if pcm := pcmOf(t, out, 22050); !bytes.Equal(pcm, sseChunk2) {
		t.Errorf("yielded %d bytes of audio, want the chunk after the error", len(pcm))
	}
}

// A non-200 answer is yielded as an error naming the status and the body, and
// nothing is read as audio.
func TestHTTPTTSStatusErrorIsYielded(t *testing.T) {
	_, base := newSSEServer(t, http.StatusUnauthorized, "bad key")
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: base})

	out := synthesizeHTTP(t, s, "Hello.")

	if errs := errorsOf(out); !slices.Equal(errs, []string{"Neuphonic API error: HTTP 401 - bad key"}) {
		t.Errorf("errors = %q, want the status and the body", errs)
	}
	if len(out) != 1 {
		t.Errorf("yielded %v, want only the error", out)
	}
}

// A request that fails outright is yielded as an error.
func TestHTTPTTSRequestFailureIsYielded(t *testing.T) {
	_, base := newSSEServer(t, http.StatusOK, "")
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: base, Client: &http.Client{
		Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, errRefused }),
	}})

	out := synthesizeHTTP(t, s, "Hello.")

	errs := errorsOf(out)
	if len(errs) != 1 || !strings.HasPrefix(errs[0], "Unknown error occurred: ") || !strings.Contains(errs[0], "refused") {
		t.Errorf("errors = %q, want the failure", errs)
	}
}

// errRefused is the failure the fake transport answers with.
//
//nolint:gochecknoglobals // sentinel error
var errRefused = errors.New("refused")

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// A message that is not a data event carrying JSON is skipped.
func TestParseSSEMessageSkipsWhatIsNotData(t *testing.T) {
	for _, message := range []string{
		"", "   ", "event: message", "data", "data:", "data: ", "data: [DONE]", "data: {not json",
	} {
		if _, ok, err := parseSSEMessage(message); ok || err != nil {
			t.Errorf("parseSSEMessage(%q) = %v, %v, want it skipped", message, ok, err)
		}
	}
	m, ok, err := parseSSEMessage(`data: {"data": {"audio": "AAE="}}`)
	if !ok || err != nil || m.Data == nil || m.Data.Audio == nil || *m.Data.Audio != "AAE=" {
		t.Errorf("parseSSEMessage = %+v, %v, %v, want the audio", m, ok, err)
	}
	if _, _, err := parseSSEMessage(`data: {"errors": "boom"}`); !errors.Is(err, errAPI) ||
		err.Error() != "Neuphonic API error unknown: boom" {
		t.Errorf("parseSSEMessage error = %v, want the errors with an unknown status", err)
	}
}

// In a pipeline, a failure yielded in the stream is pushed upstream as an
// error rather than sent on downstream.
func TestHTTPTTSErrorReachesThePipelineAsAnError(t *testing.T) {
	_, base := newSSEServer(t, http.StatusUnauthorized, "bad key")
	svc := NewHTTPTTS(HTTPTTSConfig{APIKey: "test-key", URL: base})
	var mu sync.Mutex
	var down, up []frames.Frame
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	time.Sleep(200 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()

	var upErrors []string
	for _, f := range up {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			upErrors = append(upErrors, ef.Error)
		}
	}
	if len(upErrors) == 0 || upErrors[0] != "Neuphonic API error: HTTP 401 - bad key" {
		t.Errorf("errors upstream = %q, want the failure first", upErrors)
	}
	for _, f := range down {
		if _, ok := f.(*frames.ErrorFrame); ok {
			t.Errorf("an error went downstream: %v", f)
		}
	}
}

// mapsEqual reports whether two decoded JSON objects hold the same values.
func mapsEqual(a, b map[string]any) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; !ok || w != v {
			return false
		}
	}
	return true
}
