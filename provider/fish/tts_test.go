package fish

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Tests for the synthesis request. Fish takes the text and the voice as JSON and
// streams PCM back, with the backend model selected by a header rather than in
// the body.

// captured is the request the fake endpoint saw.
type captured struct {
	path   string
	header http.Header
	body   map[string]any
}

// fishServer stands in for Fish Audio's API, replying with pcm.
func fishServer(t *testing.T, status int, pcm []byte) (base string, seen func() *captured) {
	t.Helper()
	got := make(chan *captured, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := &captured{path: r.URL.Path, header: r.Header.Clone(), body: map[string]any{}}
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading the request: %v", err)
			return
		}
		if err := json.Unmarshal(raw, &c.body); err != nil {
			t.Errorf("decoding the request: %v", err)
			return
		}
		got <- c
		w.WriteHeader(status)
		_, _ = w.Write(pcm)
	}))
	t.Cleanup(srv.Close)

	return srv.URL, func() *captured {
		select {
		case c := <-got:
			return c
		default:
			t.Fatal("the endpoint was never called")
			return nil
		}
	}
}

// synth builds a synthesizer with the defaults NewTTS fills in.
func synth(base string, opts ...func(*Config)) *synthesizer {
	cfg := Config{
		APIKey:     "test-key",
		BaseURL:    base,
		Model:      defaultModel,
		SampleRate: defaultSampleRate,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &synthesizer{cfg: cfg, http: &http.Client{}}
}

// run performs a synthesis and returns the PCM it emitted.
func run(t *testing.T, s *synthesizer, text string) ([]byte, error) {
	t.Helper()
	var pcm []byte
	err := s.RunTTS(context.Background(), text, "", func(f frames.Frame) error {
		if af, ok := f.(frames.OutputAudioFrame); ok {
			pcm = append(pcm, af.AudioData().Audio...)
		}
		return nil
	})
	return pcm, err
}

func TestRequestCarriesTheTextAndFormat(t *testing.T) {
	want := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	base, seen := fishServer(t, http.StatusOK, want)

	got, err := run(t, synth(base, func(c *Config) { c.ReferenceID = "voice-7" }), "hello there")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("audio = %v, want the PCM the endpoint streamed", got)
	}

	c := seen()
	if c.path != "/v1/tts" {
		t.Errorf("path = %s, want /v1/tts", c.path)
	}
	if c.header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", c.header.Get("Authorization"))
	}
	// Fish selects the backend through a header rather than the body.
	if c.header.Get("model") != defaultModel {
		t.Errorf("model header = %q, want %q", c.header.Get("model"), defaultModel)
	}
	if c.body["text"] != "hello there" {
		t.Errorf("text = %v, want the sentence", c.body["text"])
	}
	// Raw PCM is what streams straight downstream; a container would not.
	if c.body["format"] != "pcm" {
		t.Errorf("format = %v, want pcm", c.body["format"])
	}
	if c.body["sample_rate"] != float64(defaultSampleRate) {
		t.Errorf("sample_rate = %v, want %d", c.body["sample_rate"], defaultSampleRate)
	}
	if c.body["reference_id"] != "voice-7" {
		t.Errorf("reference_id = %v, want the configured voice", c.body["reference_id"])
	}
}

// TestSpeedIsSentAsProsody checks the speaking rate travels in the shape Fish
// reads it, nested under prosody.
func TestSpeedIsSentAsProsody(t *testing.T) {
	base, seen := fishServer(t, http.StatusOK, nil)
	speed := 1.3

	if _, err := run(t, synth(base, func(c *Config) { c.Speed = &speed }), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	prosody, _ := seen().body["prosody"].(map[string]any)
	if prosody["speed"] != speed {
		t.Errorf("prosody = %v, want the speed nested in it", prosody)
	}
}

// TestUnsetOptionsAreOmitted checks the optional fields are left out rather than
// sent empty, which the API would read as a request to use them.
func TestUnsetOptionsAreOmitted(t *testing.T) {
	base, seen := fishServer(t, http.StatusOK, nil)

	if _, err := run(t, synth(base), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	c := seen()
	for _, key := range []string{"reference_id", "prosody"} {
		if v, ok := c.body[key]; ok {
			t.Errorf("%s was sent as %v, want it omitted", key, v)
		}
	}
}

// TestReportsAFailedRequest checks a non-OK reply is reported rather than read
// as silence.
func TestReportsAFailedRequest(t *testing.T) {
	base, _ := fishServer(t, http.StatusUnauthorized, nil)
	if _, err := run(t, synth(base), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unauthorized reply")
	}
}

// TestReportsAnUnreachableEndpoint checks a transport failure is reported.
func TestReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := run(t, synth("http://127.0.0.1:1"), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unreachable endpoint")
	}
}

// TestSampleRate checks the rate the emitted frames are stamped with is the one
// asked of Fish.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{cfg: Config{SampleRate: 16000}}).SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}
