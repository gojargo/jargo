package chat

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
)

// uploaded is the transcription request as the endpoint received it: the form
// fields, and the file part with its declared filename.
type uploaded struct {
	path     string
	header   http.Header
	fields   map[string]string
	filename string
	file     []byte
}

// newSTTServer stands in for /audio/transcriptions, parsing the multipart body
// and replying with transcript.
func newSTTServer(t *testing.T, transcript string) (*httptest.Server, *uploaded) {
	t.Helper()
	got := &uploaded{fields: map[string]string{}}
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
			got.fields[part.FormName()] = string(body)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text":` + quote(transcript) + `}`))
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

// transcribe runs one segment through a transcriber pointed at srv.
func transcribe(t *testing.T, srv *httptest.Server, cfg STTConfig, pcm []byte, rate int) string {
	t.Helper()
	cfg.BaseURL = srv.URL
	tr := &transcriber{cfg: cfg, http: httpClientOr(cfg.HTTPClient), shaper: defaultSTTShaper{}}
	text, err := tr.Transcribe(t.Context(), pcm, rate)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	return text
}

// TestSTTConfigValidate pins the credential the transcription API requires.
func TestSTTConfigValidate(t *testing.T) {
	if err := (STTConfig{}).Validate(); err == nil {
		t.Error("Validate() on a keyless config = nil, want an error")
	}
	if err := (STTConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}

// TestSTTSegmentTranscribesBufferedSpeech drives the service the pipeline does:
// a turn delimited upstream is buffered, uploaded in one request and reported as
// a transcription. It is also where the constructor's defaults become visible,
// since only the service it builds knows which model it settled on.
func TestSTTSegmentTranscribesBufferedSpeech(t *testing.T) {
	srv, got := newSTTServer(t, "buffered words")
	svc := NewSTT(STTConfig{APIKey: "k", BaseURL: srv.URL, SampleRate: 16000})

	transcripts := make(chan string, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.TranscriptionFrame); ok {
				select {
				case transcripts <- fr.Text:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	pcm := bytes.Repeat([]byte{0x01, 0x02}, 160)
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInputAudioRawFrame(pcm, 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	select {
	case text := <-transcripts:
		if text != "buffered words" {
			t.Errorf("transcription = %q, want %q", text, "buffered words")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no transcription reached the pipeline")
	}
	task.StopWhenDone()
	<-runDone

	if got.fields["model"] != defaultSTTModel {
		t.Errorf("model = %q, want the OpenAI default %q", got.fields["model"], defaultSTTModel)
	}
	if !bytes.HasSuffix(got.file, pcm) {
		t.Error("the uploaded WAV does not carry the buffered speech")
	}
}

// TestNewCompatSTTLabel checks a compatible provider gets its own label on the
// service it builds from this base.
func TestNewCompatSTTLabel(t *testing.T) {
	wantLabel(t, "GroqSTT", NewCompatSTT("GroqSTT", "https://base.example", "whisper-large-v3", STTConfig{APIKey: "k"}))
}

// TestTranscribeUploadsWAV checks the segment is uploaded as a WAV file whose
// header describes the PCM it wraps, since the API reads the format from the
// file rather than from a parameter.
func TestTranscribeUploadsWAV(t *testing.T) {
	srv, got := newSTTServer(t, "hello world")
	pcm := bytes.Repeat([]byte{0x01, 0x02}, 160)

	if text := transcribe(t, srv, STTConfig{APIKey: "test-key"}, pcm, 16000); text != "hello world" {
		t.Errorf("transcript = %q, want %q", text, "hello world")
	}

	if got.path != "/audio/transcriptions" {
		t.Errorf("path = %q, want /audio/transcriptions", got.path)
	}
	if h := got.header.Get("Authorization"); h != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", h)
	}
	if got.filename != "audio.wav" {
		t.Errorf("filename = %q, want audio.wav", got.filename)
	}
	if !bytes.HasPrefix(got.file, []byte("RIFF")) || !bytes.Contains(got.file[:16], []byte("WAVE")) {
		t.Fatalf("uploaded file is not a WAV: % x", got.file[:16])
	}
	if !bytes.HasSuffix(got.file, pcm) {
		t.Error("uploaded WAV does not end with the PCM it was given")
	}
}

// TestTranscribeFields checks the form carries the fields the API expects, and
// omits every optional one the caller left unset.
func TestTranscribeFields(t *testing.T) {
	srv, got := newSTTServer(t, "")
	transcribe(t, srv, STTConfig{APIKey: "k"}, []byte{0, 0}, 16000)

	// gpt-4o-transcribe supports the plain json response format.
	if got.fields["response_format"] != "json" {
		t.Errorf("response_format = %q, want json", got.fields["response_format"])
	}
	for _, f := range []string{"language", "prompt", "temperature"} {
		if _, present := got.fields[f]; present {
			t.Errorf("%s was sent for an unset config: %q", f, got.fields[f])
		}
	}
}

// TestTranscribeOptionalFields checks each optional field crosses the wire when
// set, with the language reduced to its base code.
func TestTranscribeOptionalFields(t *testing.T) {
	srv, got := newSTTServer(t, "")
	temp := 0.2
	transcribe(t, srv, STTConfig{
		APIKey:      "k",
		Language:    language.FrenchCA,
		Prompt:      "a product demo",
		Temperature: &temp,
	}, []byte{0, 0}, 16000)

	if got.fields["language"] != "fr" {
		t.Errorf("language = %q, want the base code fr", got.fields["language"])
	}
	if got.fields["prompt"] != "a product demo" {
		t.Errorf("prompt = %q", got.fields["prompt"])
	}
	if got.fields["temperature"] != "0.2" {
		t.Errorf("temperature = %q, want 0.2", got.fields["temperature"])
	}
}

// TestTranscribeOmitsEmptyModel checks an empty model sends no model field.
// Azure addresses the model by deployment in the URL and rejects the field.
func TestTranscribeOmitsEmptyModel(t *testing.T) {
	srv, got := newSTTServer(t, "")
	transcribe(t, srv, STTConfig{APIKey: "k", Model: ""}, []byte{0, 0}, 16000)
	if _, present := got.fields["model"]; present {
		t.Errorf("model = %q, want the field omitted", got.fields["model"])
	}
}

// sttShaper addresses transcription the way a deployment with its own URL layout
// and auth scheme does.
type sttShaper struct{}

func (sttShaper) Endpoint(baseURL string) string {
	return baseURL + "/openai/deployments/mine/audio/transcriptions"
}

func (sttShaper) Authorize(req *http.Request, apiKey string) {
	req.Header.Set("api-key", apiKey)
}

// TestNewShapedSTTUsesShaper checks the shaper decides both the URL and the
// authorization for transcription too.
func TestNewShapedSTTUsesShaper(t *testing.T) {
	srv, got := newSTTServer(t, "bonjour")
	tr := &transcriber{
		cfg:    STTConfig{APIKey: "azure-key", BaseURL: srv.URL},
		http:   &http.Client{},
		shaper: sttShaper{},
	}
	text, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "bonjour" {
		t.Errorf("transcript = %q", text)
	}
	if got.path != "/openai/deployments/mine/audio/transcriptions" {
		t.Errorf("path = %q, want the shaper's URL", got.path)
	}
	if got.header.Get("api-key") != "azure-key" {
		t.Errorf("api-key = %q, want the shaper's scheme", got.header.Get("api-key"))
	}
	if got.header.Get("Authorization") != "" {
		t.Error("the default Bearer header was sent alongside the shaper's scheme")
	}

	svc := NewShapedSTT("AzureSTT", "https://base.example", "", sttShaper{}, STTConfig{APIKey: "k"})
	wantLabel(t, "AzureSTT", svc)
}

// TestTranscribeStatusError checks a non-200 is reported with the body, rather
// than decoded as an empty transcript.
func TestTranscribeStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("unsupported audio"))
	}))
	defer srv.Close()

	tr := &transcriber{
		cfg:    STTConfig{APIKey: "k", BaseURL: srv.URL},
		http:   &http.Client{},
		shaper: defaultSTTShaper{},
	}
	_, err := tr.Transcribe(t.Context(), []byte{0, 0}, 16000)
	if !errors.Is(err, errStatus) {
		t.Fatalf("Transcribe error = %v, want errStatus", err)
	}
	if !strings.Contains(err.Error(), "unsupported audio") {
		t.Errorf("error = %v, want it to carry the response body", err)
	}
}

// TestHTTPClientOr checks a caller-supplied client is adopted as-is and a nil
// one yields a usable default, so a caller can set their own timeout, proxy or
// connection limits.
func TestHTTPClientOr(t *testing.T) {
	mine := &http.Client{}
	if got := httpClientOr(mine); got != mine {
		t.Error("a supplied client was not adopted")
	}
	if got := httpClientOr(nil); got == nil {
		t.Error("httpClientOr(nil) = nil, want a default client")
	}
}

// TestSTTUsesConfiguredHTTPClient checks the configured client is the one that
// makes the request, not merely stored.
func TestSTTUsesConfiguredHTTPClient(t *testing.T) {
	srv, _ := newSTTServer(t, "ok")
	rt := &countingTransport{base: http.DefaultTransport}
	transcribe(t, srv, STTConfig{APIKey: "k", HTTPClient: &http.Client{Transport: rt}}, []byte{0, 0}, 16000)
	if rt.calls != 1 {
		t.Errorf("configured client made %d requests, want 1", rt.calls)
	}
}

// countingTransport counts the requests that pass through it.
type countingTransport struct {
	base  http.RoundTripper
	calls int
}

func (c *countingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	c.calls++
	return c.base.RoundTrip(req)
}
