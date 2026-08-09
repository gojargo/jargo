package elevenlabs

import (
	"bytes"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
)

// uploadedSTT is the transcription request as the endpoint received it. The
// fields are collected as a list per name, because keyterms repeats.
type uploadedSTT struct {
	path     string
	header   http.Header
	fields   map[string][]string
	filename string
	file     []byte
}

// field reports the single value sent for name, or "" when it was omitted.
func (u *uploadedSTT) field(name string) string {
	if v := u.fields[name]; len(v) > 0 {
		return v[0]
	}
	return ""
}

// newSTTServer stands in for /v1/speech-to-text, parsing the multipart body and
// replying with transcript.
func newSTTServer(t *testing.T, transcript string) (*httptest.Server, *uploadedSTT) {
	t.Helper()
	got := &uploadedSTT{fields: map[string][]string{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.header = r.Header.Clone()

		_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil {
			t.Errorf("parsing Content-Type: %v", err)
			return
		}
		mr := multipart.NewReader(r.Body, params["boundary"])
		for {
			part, err := mr.NextPart()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Errorf("reading multipart body: %v", err)
				return
			}
			body, err := io.ReadAll(part)
			if err != nil {
				t.Errorf("reading part %q: %v", part.FormName(), err)
				return
			}
			if part.FormName() == "file" {
				got.filename = part.FileName()
				got.file = body
				continue
			}
			got.fields[part.FormName()] = append(got.fields[part.FormName()], string(body))
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":"` + transcript + `"}`))
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

// transcribeWith runs one segment through a transcriber pointed at srv.
func transcribeWith(t *testing.T, srv *httptest.Server, cfg STTConfig, pcm []byte) string {
	t.Helper()
	cfg.BaseURL = srv.URL
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	tr := &sttTranscriber{cfg: cfg, http: &http.Client{}}
	text, err := tr.Transcribe(t.Context(), pcm, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	return text
}

// TestSTTMetadata checks the service reports the model it transcribes with and
// the latency downstream budgets against.
func TestSTTMetadata(t *testing.T) {
	tr := &sttTranscriber{cfg: STTConfig{APIKey: "k", Model: defaultSTTModel}}
	meta := tr.Metadata()
	if meta.Model != defaultSTTModel {
		t.Errorf("Metadata().Model = %q, want %q", meta.Model, defaultSTTModel)
	}
	if meta.TTFSP99 != sttTTFSP99 {
		t.Errorf("Metadata().TTFSP99 = %v, want %v", meta.TTFSP99, sttTTFSP99)
	}
}

// TestTranscribeUploadsWAV checks the segment is uploaded as a WAV file, keyed
// by the header ElevenLabs authorizes with rather than a Bearer token.
func TestTranscribeUploadsWAV(t *testing.T) {
	srv, got := newSTTServer(t, "hello world")
	pcm := bytes.Repeat([]byte{0x01, 0x02}, 160)

	if text := transcribeWith(t, srv, STTConfig{APIKey: "test-key"}, pcm); text != "hello world" {
		t.Errorf("transcript = %q, want %q", text, "hello world")
	}

	if got.path != "/v1/speech-to-text" {
		t.Errorf("path = %q, want /v1/speech-to-text", got.path)
	}
	if h := got.header.Get("xi-api-key"); h != "test-key" {
		t.Errorf("xi-api-key = %q, want the configured key", h)
	}
	if got.filename != "audio.wav" {
		t.Errorf("filename = %q, want audio.wav", got.filename)
	}
	if !bytes.HasPrefix(got.file, []byte("RIFF")) {
		t.Fatalf("uploaded file is not a WAV: % x", got.file[:16])
	}
	if !bytes.HasSuffix(got.file, pcm) {
		t.Error("uploaded WAV does not end with the PCM it was given")
	}
	if got.field("model_id") != defaultSTTModel {
		t.Errorf("model_id = %q, want %q", got.field("model_id"), defaultSTTModel)
	}
}

// TestTranscribeOmitsUnsetFields checks the optional fields stay off the request
// when the caller left them unset, so the API applies its own behavior. An unset
// language in particular means auto-detect.
func TestTranscribeOmitsUnsetFields(t *testing.T) {
	srv, got := newSTTServer(t, "")
	transcribeWith(t, srv, STTConfig{APIKey: "k"}, []byte{0, 0})

	for _, f := range []string{"language_code", "tag_audio_events", "keyterms"} {
		if _, present := got.fields[f]; present {
			t.Errorf("%s was sent for an unset config: %v", f, got.fields[f])
		}
	}
}

// TestTranscribeOptionalFields checks each optional field crosses the wire when
// set, with the language mapped to ElevenLabs' three-letter code, the boolean
// rendered lowercase, and one field per keyterm.
func TestTranscribeOptionalFields(t *testing.T) {
	srv, got := newSTTServer(t, "")
	tag := false
	transcribeWith(t, srv, STTConfig{
		APIKey:         "k",
		Language:       language.French,
		TagAudioEvents: &tag,
		Keyterms:       []string{"jargo", "pipeline"},
	}, []byte{0, 0})

	if got.field("language_code") != "fra" {
		t.Errorf("language_code = %q, want fra", got.field("language_code"))
	}
	if got.field("tag_audio_events") != "false" {
		t.Errorf("tag_audio_events = %q, want a lowercase false", got.field("tag_audio_events"))
	}
	if len(got.fields["keyterms"]) != 2 ||
		got.fields["keyterms"][0] != "jargo" || got.fields["keyterms"][1] != "pipeline" {
		t.Errorf("keyterms = %v, want one field per term", got.fields["keyterms"])
	}
}

// TestTranscribeStatusError checks a non-200 is reported with the body attached.
func TestTranscribeStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("invalid api key"))
	}))
	defer srv.Close()

	tr := &sttTranscriber{cfg: STTConfig{APIKey: "k", BaseURL: srv.URL}, http: &http.Client{}}
	_, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000)
	if !errors.Is(err, errSTTStatus) {
		t.Fatalf("Transcribe error = %v, want errSTTStatus", err)
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// TestElevenlabsSTTLanguage pins the recognition codes. Transcription takes
// ISO-639-3, unlike synthesis which takes the two-letter base code, and a
// language ElevenLabs does not transcribe maps to nothing so the model detects
// it instead.
func TestElevenlabsSTTLanguage(t *testing.T) {
	cases := map[language.Language]string{
		language.English:      "eng",
		language.EnglishGB:    "eng",
		language.French:       "fra",
		language.FrenchCA:     "fra",
		language.Spanish:      "spa",
		language.German:       "deu",
		language.Portuguese:   "por",
		language.PortugueseBR: "por",
		language.Russian:      "rus",
		language.Japanese:     "jpn",
		language.Korean:       "kor",
		language.Chinese:      "zho",
		// Filipino is its own code in both directions.
		language.Language(langFil): langFil,
		// Not transcribed, so nothing is sent and the model auto-detects.
		language.Language("cy"): "",
		language.Language(""):   "",
	}
	for in, want := range cases {
		if got := elevenlabsSTTLanguage(in); got != want {
			t.Errorf("elevenlabsSTTLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}
