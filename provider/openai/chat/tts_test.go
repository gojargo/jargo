package chat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// speak runs one synthesis against srv and returns the PCM the frames carried,
// along with the rate they declared.
func speak(t *testing.T, srv *httptest.Server, cfg TTSConfig, text string) ([]byte, int) {
	t.Helper()
	cfg.BaseURL = srv.URL
	s := &synthesizer{cfg: cfg, http: httpClientOr(cfg.HTTPClient)}

	var pcm bytes.Buffer
	rate := 0
	err := s.RunTTS(t.Context(), text, "", func(f frames.Frame) error {
		audio, ok := f.(*frames.TTSAudioRawFrame)
		if !ok {
			t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
			return nil
		}
		rate = audio.SampleRate
		pcm.Write(audio.Audio)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return pcm.Bytes(), rate
}

// newTTSServer stands in for /audio/speech, recording the request and replying
// with pcm as the raw body.
func newTTSServer(t *testing.T, pcm []byte) (*httptest.Server, *uploaded, *map[string]any) {
	t.Helper()
	got := &uploaded{fields: map[string]string{}}
	body := map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.path = r.URL.Path
		got.header = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		_, _ = w.Write(pcm)
	}))
	t.Cleanup(srv.Close)
	return srv, got, &body
}

// TestTTSConfigValidate pins the credential the speech API requires.
func TestTTSConfigValidate(t *testing.T) {
	if err := (TTSConfig{}).Validate(); err == nil {
		t.Error("Validate() on a keyless config = nil, want an error")
	}
	if err := (TTSConfig{APIKey: "k"}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}

// TestSynthesizerSampleRate checks the service reports the rate OpenAI's "pcm"
// response format is fixed at, since nothing in the response declares it.
func TestSynthesizerSampleRate(t *testing.T) {
	s := &synthesizer{}
	if got := s.SampleRate(); got != ttsSampleRate {
		t.Errorf("SampleRate() = %d, want %d", got, ttsSampleRate)
	}
}

// TestRunTTSRequestShape checks the synthesis request carries the model, voice,
// text and the raw-PCM response format, and that the audio comes back as TTS
// frames tagged with that rate.
func TestRunTTSRequestShape(t *testing.T) {
	want := bytes.Repeat([]byte{0x11, 0x22}, 64)
	srv, got, body := newTTSServer(t, want)

	pcm, rate := speak(t, srv, TTSConfig{APIKey: "test-key"}, "hello there")

	if got.path != "/audio/speech" {
		t.Errorf("path = %q, want /audio/speech", got.path)
	}
	if h := got.header.Get("Authorization"); h != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", h)
	}
	if h := got.header.Get("Content-Type"); h != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", h)
	}
	if (*body)["input"] != "hello there" {
		t.Errorf("input = %v, want the text to speak", (*body)["input"])
	}
	if (*body)["response_format"] != "pcm" {
		t.Errorf("response_format = %v, want pcm", (*body)["response_format"])
	}
	if !bytes.Equal(pcm, want) {
		t.Errorf("PCM = % x, want the response body verbatim", pcm)
	}
	if rate != ttsSampleRate {
		t.Errorf("frame sample rate = %d, want %d", rate, ttsSampleRate)
	}
}

// TestTTSSpeaksThroughThePipeline drives the service the pipeline does: a spoken
// line is synthesized and the audio travels downstream. It is also where the
// constructor's defaults become visible, since only the service it builds knows
// which model and voice it settled on.
func TestTTSSpeaksThroughThePipeline(t *testing.T) {
	want := bytes.Repeat([]byte{0x11, 0x22}, 64)
	srv, _, body := newTTSServer(t, want)
	svc := NewTTS(TTSConfig{APIKey: "k", BaseURL: srv.URL})

	audio := make(chan []byte, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.TTSAudioRawFrame); ok {
			select {
			case audio <- fr.Audio:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSSpeakFrame("hello there"))

	select {
	case got := <-audio:
		if len(got) == 0 {
			t.Error("an empty audio frame reached the pipeline")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no TTS audio reached the pipeline")
	}
	task.StopWhenDone()
	<-runDone

	if (*body)["model"] != defaultTTSModel {
		t.Errorf("model = %v, want the default %q", (*body)["model"], defaultTTSModel)
	}
	if (*body)["voice"] != defaultTTSVoice {
		t.Errorf("voice = %v, want the default %q", (*body)["voice"], defaultTTSVoice)
	}
}

// TestNewCompatTTSKeepsConfiguredValues checks the model and voice a compatible
// provider hands in fill the config only where the caller left a gap.
func TestNewCompatTTSKeepsConfiguredValues(t *testing.T) {
	srv, _, body := newTTSServer(t, []byte{0, 0})

	kokoro := NewCompatTTS("KokoroTTS", "https://base.example", "kokoro", "af_heart", TTSConfig{APIKey: "k"})
	wantLabel(t, "KokoroTTS", kokoro)

	speak(t, srv, TTSConfig{APIKey: "k", Model: "mine", Voice: "nova"}, "hi")
	if (*body)["model"] != "mine" || (*body)["voice"] != "nova" {
		t.Errorf("configured model/voice were overwritten: %v / %v", (*body)["model"], (*body)["voice"])
	}
}

// TestRunTTSOptionalFields checks instructions and speed are sent only when set,
// so the API applies its own defaults otherwise.
func TestRunTTSOptionalFields(t *testing.T) {
	srv, _, body := newTTSServer(t, []byte{0, 0})

	speak(t, srv, TTSConfig{APIKey: "k"}, "hi")
	for _, f := range []string{"instructions", "speed"} {
		if _, present := (*body)[f]; present {
			t.Errorf("%s was sent for an unset config: %v", f, (*body)[f])
		}
	}

	speed := 1.25
	speak(t, srv, TTSConfig{APIKey: "k", Instructions: "sound cheerful", Speed: &speed}, "hi")
	if (*body)["instructions"] != "sound cheerful" {
		t.Errorf("instructions = %v", (*body)["instructions"])
	}
	if (*body)["speed"] != 1.25 {
		t.Errorf("speed = %v, want 1.25", (*body)["speed"])
	}
}

// TestRunTTSStatusError checks a non-200 is reported rather than streamed on as
// though the body were audio.
func TestRunTTSStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte("no access"))
	}))
	defer srv.Close()

	s := &synthesizer{cfg: TTSConfig{APIKey: "k", BaseURL: srv.URL}, http: &http.Client{}}
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error {
		t.Error("a frame was yielded for a failed request")
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS on a 403 = nil, want an error")
	}
}

// TestRunTTSYieldError checks an error from downstream stops the stream and is
// reported back.
func TestRunTTSYieldError(t *testing.T) {
	srv, _, _ := newTTSServer(t, bytes.Repeat([]byte{1}, 4096))

	s := &synthesizer{cfg: TTSConfig{APIKey: "k", BaseURL: srv.URL}, http: &http.Client{}}
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error { return errDownstream })
	if !errors.Is(err, errDownstream) {
		t.Fatalf("RunTTS error = %v, want the downstream error", err)
	}
}

// TestTTSUsesConfiguredHTTPClient checks the configured client is the one that
// makes the request.
func TestTTSUsesConfiguredHTTPClient(t *testing.T) {
	srv, _, _ := newTTSServer(t, []byte{0, 0})
	rt := &countingTransport{base: http.DefaultTransport}
	speak(t, srv, TTSConfig{APIKey: "k", HTTPClient: &http.Client{Transport: rt}}, "hi")
	if rt.calls != 1 {
		t.Errorf("configured client made %d requests, want 1", rt.calls)
	}
}
