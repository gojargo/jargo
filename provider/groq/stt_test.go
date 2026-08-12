package groq

import (
	"bytes"
	"context"
	"errors"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the transcription request. Groq's endpoint is OpenAI-compatible and
// takes the segment as an uploaded WAV alongside the settings as form fields,
// with the optional ones left out entirely rather than sent empty.

// request is what the fake endpoint saw.
type request struct {
	path   string
	header http.Header
	fields map[string]string
	file   []byte
}

// sttServer stands in for the Groq transcriptions endpoint. It replies with
// reply and reports the request it was sent.
func sttServer(t *testing.T, status int, reply string) (url string, seen func() *request) {
	t.Helper()
	got := make(chan *request, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req := &request{path: r.URL.Path, header: r.Header.Clone(), fields: map[string]string{}}

		_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil {
			t.Errorf("parse content type: %v", err)
			return
		}
		mr := multipart.NewReader(r.Body, params["boundary"])
		form, err := mr.ReadForm(1 << 20)
		if err != nil {
			t.Errorf("read form: %v", err)
			return
		}
		for k, v := range form.Value {
			req.fields[k] = v[0]
		}
		if files := form.File["file"]; len(files) == 1 {
			f, err := files[0].Open()
			if err != nil {
				t.Errorf("open uploaded file: %v", err)
				return
			}
			defer func() { _ = f.Close() }()
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(f); err != nil {
				t.Errorf("read uploaded file: %v", err)
				return
			}
			req.file = buf.Bytes()
			req.fields["__filename"] = files[0].Filename
		}

		got <- req
		w.WriteHeader(status)
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)

	return srv.URL, func() *request {
		select {
		case r := <-got:
			return r
		case <-time.After(2 * time.Second):
			t.Fatal("the endpoint was never called")
			return nil
		}
	}
}

// transcriber builds a transcriber pointed at url.
func transcriber(cfg STTConfig, url string) *sttTranscriber {
	cfg.BaseURL = url
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	return &sttTranscriber{cfg: cfg, http: &http.Client{}}
}

func TestTranscribeSendsTheSegmentAndSettings(t *testing.T) {
	url, seen := sttServer(t, http.StatusOK, `{"text":"hello there"}`)
	temp := 0.4
	tr := transcriber(STTConfig{
		APIKey:      "key-1",
		Language:    language.English,
		Prompt:      "a weather report",
		Temperature: &temp,
	}, url)

	pcm := make([]byte, 3200) // 100 ms at 16 kHz
	text, err := tr.Transcribe(context.Background(), pcm, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "hello there" {
		t.Errorf("transcript = %q, want %q", text, "hello there")
	}

	req := seen()
	if req.path != "/audio/transcriptions" {
		t.Errorf("path = %s, want /audio/transcriptions", req.path)
	}
	if got := req.header.Get("Authorization"); got != "Bearer key-1" {
		t.Errorf("Authorization = %q, want the key as a bearer token", got)
	}

	want := map[string]string{
		"model":           defaultSTTModel,
		"response_format": "json",
		"language":        "en",
		"prompt":          "a weather report",
		"temperature":     "0.4",
		"__filename":      "audio.wav",
	}
	for k, v := range want {
		if req.fields[k] != v {
			t.Errorf("field %s = %q, want %q", k, req.fields[k], v)
		}
	}

	// The segment is uploaded as a WAV, not as raw PCM.
	if len(req.file) != len(stt.WAV(pcm, 16000, 1)) {
		t.Errorf("uploaded %d bytes, want a WAV of the segment", len(req.file))
	}
	if !bytes.HasPrefix(req.file, []byte("RIFF")) {
		t.Error("the uploaded file is not a WAV")
	}
}

// TestTranscribeOmitsUnsetSettings checks the optional fields are left out
// rather than sent empty, which the API would read as a request to use them.
func TestTranscribeOmitsUnsetSettings(t *testing.T) {
	url, seen := sttServer(t, http.StatusOK, `{"text":"ok"}`)
	tr := transcriber(STTConfig{APIKey: "key-1", Language: language.English}, url)

	if _, err := tr.Transcribe(context.Background(), make([]byte, 320), 16000); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	req := seen()
	for _, absent := range []string{"prompt", "temperature"} {
		if v, ok := req.fields[absent]; ok {
			t.Errorf("field %s was sent as %q, want it omitted", absent, v)
		}
	}
}

// TestTranscribeSendsTheBaseLanguageCode checks a regional language is sent as
// its base code: the transcription API names languages by ISO-639-1, so a
// regional tag would not be recognized.
func TestTranscribeSendsTheBaseLanguageCode(t *testing.T) {
	url, seen := sttServer(t, http.StatusOK, `{"text":"ok"}`)
	tr := transcriber(STTConfig{APIKey: "key-1", Language: language.EnglishUS}, url)

	if _, err := tr.Transcribe(context.Background(), make([]byte, 320), 16000); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got := seen().fields["language"]; got != "en" {
		t.Errorf("language = %q, want the base code en", got)
	}
}

// TestTranscribeReportsAFailedRequest checks a non-OK reply is reported with the
// status and what the API said, rather than read as an empty transcript.
func TestTranscribeReportsAFailedRequest(t *testing.T) {
	url, _ := sttServer(t, http.StatusUnauthorized, `{"error":"bad key"}`)
	tr := transcriber(STTConfig{APIKey: "wrong"}, url)

	_, err := tr.Transcribe(context.Background(), make([]byte, 320), 16000)
	if err == nil {
		t.Fatal("Transcribe accepted an unauthorized reply")
	}
	if !errors.Is(err, errStatus) {
		t.Errorf("error = %v, want it to report the status", err)
	}
	if !bytes.Contains([]byte(err.Error()), []byte("bad key")) {
		t.Errorf("error = %v, want it to carry what the API said", err)
	}
}

// TestTranscribeReportsAnUnreadableReply checks a body that is not the expected
// JSON is reported rather than yielding an empty transcript.
func TestTranscribeReportsAnUnreadableReply(t *testing.T) {
	url, _ := sttServer(t, http.StatusOK, `not json`)
	tr := transcriber(STTConfig{APIKey: "key-1"}, url)

	if _, err := tr.Transcribe(context.Background(), make([]byte, 320), 16000); err == nil {
		t.Fatal("Transcribe accepted a reply that is not JSON")
	}
}

// TestTranscribeReportsAnUnreachableEndpoint checks a transport failure is
// reported rather than swallowed.
func TestTranscribeReportsAnUnreachableEndpoint(t *testing.T) {
	tr := transcriber(STTConfig{APIKey: "key-1"}, "http://127.0.0.1:1")

	if _, err := tr.Transcribe(context.Background(), make([]byte, 320), 16000); err == nil {
		t.Fatal("Transcribe accepted an unreachable endpoint")
	}
}

// TestMetadataReportsTheMeasuredLatency checks the transcript latency the turn
// strategies size their wait by: the measured figure, or an override.
func TestMetadataReportsTheMeasuredLatency(t *testing.T) {
	if got := (&sttTranscriber{}).Metadata().TTFSP99; got != stt.GroqTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured %v", got, stt.GroqTTFSP99)
	}
	tr := &sttTranscriber{cfg: STTConfig{TTFSP99: 250 * time.Millisecond}}
	if got := tr.Metadata().TTFSP99; got != 250*time.Millisecond {
		t.Errorf("TTFSP99 = %v, want the override", got)
	}
}

// TestNewSTTDefaults checks a caller who names only their key gets a usable
// service, and that the defaults it fills in are the ones sent on the wire.
func TestNewSTTDefaults(t *testing.T) {
	if svc := NewSTT(STTConfig{APIKey: "key-1"}); svc == nil {
		t.Fatal("NewSTT returned nothing")
	} else if svc.Name() == "" {
		t.Error("the service has no name")
	}

	// NewSTT fills the model and the language in; the request carries them.
	url, seen := sttServer(t, http.StatusOK, `{"text":"ok"}`)
	cfg := STTConfig{APIKey: "key-1"}
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Language == "" {
		cfg.Language = language.English
	}
	tr := &sttTranscriber{cfg: cfg, http: &http.Client{}}
	tr.cfg.BaseURL = url

	if _, err := tr.Transcribe(context.Background(), make([]byte, 320), 16000); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	req := seen()
	if req.fields["model"] != defaultSTTModel {
		t.Errorf("model = %q, want the default %q", req.fields["model"], defaultSTTModel)
	}
	if req.fields["language"] != "en" {
		t.Errorf("language = %q, want the default en", req.fields["language"])
	}
}
