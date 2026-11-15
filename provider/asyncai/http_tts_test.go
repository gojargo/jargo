package asyncai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
)

// httpRequest is what the fake endpoint was asked.
type httpRequest struct {
	path    string
	header  http.Header
	payload map[string]any
}

// newHTTPServer is a fake Async endpoint answering every request with status
// and body, and recording what it was asked.
func newHTTPServer(t *testing.T, status int, body []byte) (*[]httpRequest, string) {
	t.Helper()
	var requests []httpRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		payload := map[string]any{}
		_ = json.Unmarshal(raw, &payload)
		requests = append(requests, httpRequest{path: r.URL.Path, header: r.Header.Clone(), payload: payload})
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return &requests, srv.URL
}

func runHTTP(t *testing.T, s *httpSynthesizer) []frames.Frame {
	t.Helper()
	var out []frames.Frame
	if err := s.RunTTS(t.Context(), "Hello there.", "c1", func(f frames.Frame) error {
		out = append(out, f)
		return nil
	}); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return out
}

// The request goes to the streaming path with the version and the key, carries
// the voice, the model, the output format and the language, and the whole of
// the audio is yielded as one frame.
func TestHTTPTTSRequestsAndYieldsTheAudio(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x01, 0x02}, 300)
	requests, url := newHTTPServer(t, http.StatusOK, pcm)
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "k", Voice: "v1", URL: url, Language: language.French})

	out := runHTTP(t, s)

	req := (*requests)[0]
	if req.path != "/text_to_speech/streaming" {
		t.Errorf("path = %q, want /text_to_speech/streaming", req.path)
	}
	if req.header.Get("x-api-key") != "k" || req.header.Get("version") != "v1" {
		t.Errorf("headers = %v, want the key and the version", req.header)
	}
	want := map[string]any{
		"model_id":   "async_flash_v1.5",
		"transcript": "Hello there.",
		"voice":      map[string]any{"mode": "id", "id": "v1"},
		"output_format": map[string]any{
			"container": "raw", "encoding": "pcm_s16le", "sample_rate": float64(24000),
		},
		"language": "fr",
	}
	if !reflect.DeepEqual(req.payload, want) {
		t.Errorf("payload = %v, want %v", req.payload, want)
	}
	if len(out) != 1 {
		t.Fatalf("yielded %d frames, want one", len(out))
	}
	if audio, ok := out[0].(*frames.TTSAudioRawFrame); !ok || !bytes.Equal(audio.Audio, pcm) || audio.SampleRate != 24000 {
		t.Errorf("yielded %v, want the audio at 24 kHz", out[0])
	}
}

// No language configured is sent as none.
func TestHTTPTTSWithoutALanguage(t *testing.T) {
	requests, url := newHTTPServer(t, http.StatusOK, []byte{0, 0})
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "k", Voice: "v1", URL: url})
	runHTTP(t, s)
	if v, present := (*requests)[0].payload["language"]; !present || v != nil {
		t.Errorf("language = %v (%t), want null", v, present)
	}
}

// A refused request yields the API's error, then the failure it ends the
// synthesis with.
func TestHTTPTTSReportsARefusedRequest(t *testing.T) {
	_, url := newHTTPServer(t, http.StatusUnauthorized, []byte("bad key"))
	_, s := newHTTPTTS(HTTPTTSConfig{APIKey: "k", Voice: "v1", URL: url})

	out := runHTTP(t, s)

	var errs []string
	for _, f := range out {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			errs = append(errs, ef.Error)
		}
	}
	want := []string{
		"Async API error: bad key",
		"Unknown error occurred: Async API returned status 401: bad key",
	}
	if !reflect.DeepEqual(errs, want) {
		t.Errorf("errors = %q, want %q", errs, want)
	}
}
