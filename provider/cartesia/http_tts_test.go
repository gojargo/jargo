package cartesia

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
)

// httpTTSRequest is what the fake synthesis endpoint saw.
type httpTTSRequest struct {
	header http.Header
	path   string
	body   map[string]any
}

// httpTTSServer answers every synthesis with pcm, and records what it was asked.
func httpTTSServer(t *testing.T, status int, pcm []byte) (base string, seen func() *httpTTSRequest) {
	t.Helper()
	got := make(chan *httpTTSRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req := &httpTTSRequest{header: r.Header.Clone(), path: r.URL.Path, body: map[string]any{}}
		if err := json.NewDecoder(r.Body).Decode(&req.body); err != nil {
			t.Errorf("decoding the synthesis request: %v", err)
		}
		select {
		case got <- req:
		default:
		}
		w.WriteHeader(status)
		_, _ = w.Write(pcm)
	}))
	t.Cleanup(srv.Close)
	return srv.URL, func() *httpTTSRequest {
		select {
		case r := <-got:
			return r
		default:
			return nil
		}
	}
}

// collectHTTP runs one synthesis and returns the PCM it produced.
func collectHTTP(t *testing.T, s *httpSynthesizer, text string) []byte {
	t.Helper()
	var pcm []byte
	err := s.RunTTS(context.Background(), text, "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			pcm = append(pcm, af.Audio...)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return pcm
}

// httpTTSConfig is a config pointed at base with every default filled in the way
// NewHTTPTTS fills them, so the request under test is the real one.
func httpTTSConfig(base string) HTTPTTSConfig {
	return HTTPTTSConfig{
		APIKey:     "test-key",
		VoiceID:    testVoiceID,
		BaseURL:    base,
		Version:    defaultVersion,
		Model:      defaultModel,
		SampleRate: defaultSampleRate,
		Encoding:   defaultEncoding,
		Container:  defaultContainer,
		Client:     &http.Client{},
	}
}

// TestHTTPTTSConfigValidate pins what the service cannot be built without.
func TestHTTPTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing key", Cfg: HTTPTTSConfig{VoiceID: testVoiceID}, Valid: false},
		{Name: "missing voice", Cfg: HTTPTTSConfig{APIKey: "k"}, Valid: false},
		{Name: "key and voice", Cfg: HTTPTTSConfig{APIKey: "k", VoiceID: testVoiceID}, Valid: true},
	})
}

// TestNewHTTPTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewHTTPTTS(t *testing.T) {
	providertest.Service(t, "CartesiaHTTPTTS",
		NewHTTPTTS(HTTPTTSConfig{APIKey: "k", VoiceID: testVoiceID}))
}

// TestHTTPTTSRequestShape pins the request the service makes.
func TestHTTPTTSRequestShape(t *testing.T) {
	want := []byte{0x11, 0x22, 0x33, 0x44}
	base, seen := httpTTSServer(t, http.StatusOK, want)

	s := &httpSynthesizer{cfg: httpTTSConfig(base)}
	if got := collectHTTP(t, s, "hello there"); !slices.Equal(got, want) {
		t.Errorf("PCM = % x, want % x", got, want)
	}

	got := seen()
	if got == nil {
		t.Fatal("the endpoint saw no synthesis request")
	}
	if got.path != httpTTSPath {
		t.Errorf("path = %q, want %q", got.path, httpTTSPath)
	}
	if h := got.header.Get("X-API-Key"); h != "test-key" {
		t.Errorf("X-API-Key = %q, want the configured key", h)
	}
	if h := got.header.Get("Cartesia-Version"); h != defaultVersion {
		t.Errorf("Cartesia-Version = %q, want the pinned %q", h, defaultVersion)
	}
	if got.body["transcript"] != "hello there" {
		t.Errorf("transcript = %v, want the text to speak", got.body["transcript"])
	}
	if got.body["model_id"] != defaultModel {
		t.Errorf("model_id = %v, want %q", got.body["model_id"], defaultModel)
	}
	voice, _ := got.body["voice"].(map[string]any)
	if voice["mode"] != "id" || voice["id"] != testVoiceID {
		t.Errorf("voice = %v, want the configured voice in id mode", voice)
	}
	format, _ := got.body["output_format"].(map[string]any)
	if format["container"] != defaultContainer || format["encoding"] != defaultEncoding {
		t.Errorf("output_format = %v, want the raw PCM container", format)
	}
	if format["sample_rate"] != float64(defaultSampleRate) {
		t.Errorf("output_format.sample_rate = %v, want %d", format["sample_rate"], defaultSampleRate)
	}
	// Nothing optional was configured, so nothing optional is sent.
	for _, f := range []string{"language", "generation_config", "pronunciation_dict_id"} {
		if _, present := got.body[f]; present {
			t.Errorf("%s was sent for an unset config: %v", f, got.body[f])
		}
	}
}

// TestHTTPTTSOptionalFields checks the language, generation config and
// pronunciation dictionary are sent when they are set.
func TestHTTPTTSOptionalFields(t *testing.T) {
	base, seen := httpTTSServer(t, http.StatusOK, []byte{1, 2})
	speed := 1.2
	cfg := httpTTSConfig(base)
	cfg.Language = language.French
	cfg.GenerationConfig = &GenerationConfig{Speed: &speed, Emotion: "excited"}
	cfg.PronunciationDictID = "dict-1"
	collectHTTP(t, &httpSynthesizer{cfg: cfg}, "hi")

	got := seen()
	if got.body["language"] != "fr" {
		t.Errorf("language = %v, want the base code fr", got.body["language"])
	}
	if got.body["pronunciation_dict_id"] != "dict-1" {
		t.Errorf("pronunciation_dict_id = %v", got.body["pronunciation_dict_id"])
	}
	gen, _ := got.body["generation_config"].(map[string]any)
	if gen["speed"] != 1.2 || gen["emotion"] != "excited" {
		t.Errorf("generation_config = %v, want the configured speed and emotion", gen)
	}
}

// TestHTTPTTSReportsAFailure checks a rejected request surfaces with the status
// and what the service said about it.
func TestHTTPTTSReportsAFailure(t *testing.T) {
	base, _ := httpTTSServer(t, http.StatusUnauthorized, []byte("invalid api key"))
	s := &httpSynthesizer{cfg: httpTTSConfig(base)}

	err := s.RunTTS(context.Background(), "hi", "", func(frames.Frame) error { return nil })
	if err == nil {
		t.Fatal("RunTTS() = nil error for a rejected request")
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error = %v, want it to carry what the service said", err)
	}
}
