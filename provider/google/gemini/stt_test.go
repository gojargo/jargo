package gemini

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
)

// roundTripper answers every request from a canned reply, recording what it was
// asked. The Google endpoints are fixed constants rather than configuration, so
// intercepting the transport is the only way to see the request they build.
type roundTripper struct {
	status int
	body   string

	url    string
	header http.Header
	sent   map[string]any
}

func (rt *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.url = req.URL.String()
	rt.header = req.Header.Clone()
	if req.Body != nil {
		raw, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		rt.sent = map[string]any{}
		if err := json.Unmarshal(raw, &rt.sent); err != nil {
			return nil, err
		}
	}
	status := rt.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(rt.body)),
		Header:     http.Header{},
		Request:    req,
	}, nil
}

// TestTranscribeRequestShape checks the segment is addressed at the recognize
// endpoint with the key in the query, and described as the LINEAR16 audio it is.
func TestTranscribeRequestShape(t *testing.T) {
	rt := &roundTripper{body: `{"results":[{"alternatives":[{"transcript":"hello world"}]}]}`}
	tr := &sttTranscriber{cfg: STTConfig{APIKey: "test-key"}, http: &http.Client{Transport: rt}}

	pcm := bytes.Repeat([]byte{0x01, 0x02}, 160)
	text, err := tr.Transcribe(t.Context(), pcm, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "hello world" {
		t.Errorf("transcript = %q, want %q", text, "hello world")
	}

	if rt.url != sttEndpoint+"?key=test-key" {
		t.Errorf("URL = %q, want the recognize endpoint carrying the key", rt.url)
	}
	if got := rt.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}

	cfg, ok := rt.sent["config"].(map[string]any)
	if !ok {
		t.Fatalf("config = %v, want the recognition config", rt.sent["config"])
	}
	if cfg["encoding"] != "LINEAR16" {
		t.Errorf("encoding = %v, want LINEAR16", cfg["encoding"])
	}
	if cfg["sampleRateHertz"] != float64(16000) {
		t.Errorf("sampleRateHertz = %v, want the rate the segment arrived at", cfg["sampleRateHertz"])
	}
	if cfg["languageCode"] != defaultLangCode {
		t.Errorf("languageCode = %v, want the default %q", cfg["languageCode"], defaultLangCode)
	}

	audio, ok := rt.sent["audio"].(map[string]any)
	if !ok {
		t.Fatalf("audio = %v, want the audio block", rt.sent["audio"])
	}
	content, _ := audio["content"].(string)
	decoded, err := base64.StdEncoding.DecodeString(content)
	if err != nil {
		t.Fatalf("audio content is not base64: %v", err)
	}
	if !bytes.Equal(decoded, pcm) {
		t.Error("the encoded audio is not the PCM the segment carried")
	}
}

// TestTranscribeJoinsResults checks a response split across results is joined
// into one transcript, taking the first alternative of each and skipping the
// empty ones.
func TestTranscribeJoinsResults(t *testing.T) {
	rt := &roundTripper{body: `{"results":[
		{"alternatives":[{"transcript":"hello"},{"transcript":"hallo"}]},
		{"alternatives":[{"transcript":""}]},
		{"alternatives":[]},
		{"alternatives":[{"transcript":"world"}]}
	]}`}
	tr := &sttTranscriber{cfg: STTConfig{APIKey: "k"}, http: &http.Client{Transport: rt}}

	text, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "hello world" {
		t.Errorf("transcript = %q, want the first alternatives joined", text)
	}
}

// TestTranscribeEmptyResults checks a response with nothing recognized yields an
// empty transcript rather than an error.
func TestTranscribeEmptyResults(t *testing.T) {
	rt := &roundTripper{body: `{"results":[]}`}
	tr := &sttTranscriber{cfg: STTConfig{APIKey: "k"}, http: &http.Client{Transport: rt}}

	text, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "" {
		t.Errorf("transcript = %q, want empty", text)
	}
}

// TestTranscribeStatusError checks a non-200 is reported with the body attached.
func TestTranscribeStatusError(t *testing.T) {
	rt := &roundTripper{status: http.StatusBadRequest, body: "bad encoding"}
	tr := &sttTranscriber{cfg: STTConfig{APIKey: "k"}, http: &http.Client{Transport: rt}}

	_, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000)
	if !errors.Is(err, errStatus) {
		t.Fatalf("Transcribe error = %v, want errStatus", err)
	}
	if !strings.Contains(err.Error(), "bad encoding") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// TestLanguageToGoogleSTT pins the recognition language codes. Each base
// language resolves to the region Google recognizes it in by default, an
// unmodeled language passes its own code through, and the zero value falls back
// to US English.
func TestLanguageToGoogleSTT(t *testing.T) {
	cases := map[language.Language]string{
		language.English:      "en-US",
		language.EnglishGB:    "en-US",
		language.French:       "fr-FR",
		language.FrenchCA:     "fr-FR",
		language.Spanish:      "es-ES",
		language.German:       "de-DE",
		language.Italian:      "it-IT",
		language.Dutch:        "nl-NL",
		language.Portuguese:   "pt-PT",
		language.PortugueseBR: "pt-PT",
		language.Polish:       "pl-PL",
		language.Russian:      "ru-RU",
		language.Japanese:     "ja-JP",
		language.Korean:       "ko-KR",
		language.Chinese:      "cmn-Hans-CN",
		language.ChineseTW:    "cmn-Hans-CN",
		// Not modeled above, so the configured code is passed through as-is.
		language.Language("sv-SE"): "sv-SE",
		// The zero value names no language at all.
		language.Language(""): defaultLangCode,
	}
	for in, want := range cases {
		if got := languageToGoogleSTT(in); got != want {
			t.Errorf("languageToGoogleSTT(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestTranscribeSendsConfiguredLanguage checks the configured language reaches
// the request as the mapped code.
func TestTranscribeSendsConfiguredLanguage(t *testing.T) {
	rt := &roundTripper{body: `{"results":[]}`}
	tr := &sttTranscriber{
		cfg:  STTConfig{APIKey: "k", Language: language.Japanese},
		http: &http.Client{Transport: rt},
	}
	if _, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	cfg, _ := rt.sent["config"].(map[string]any)
	if cfg["languageCode"] != "ja-JP" {
		t.Errorf("languageCode = %v, want ja-JP", cfg["languageCode"])
	}
}
