package lmnt

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Tests for the synthesis request. LMNT takes the text, voice and model as JSON
// and streams raw PCM back at the rate asked for.

// captured is the request the fake endpoint saw.
type captured struct {
	path   string
	header http.Header
	body   map[string]any
}

// lmntServer stands in for LMNT's API, replying with pcm.
func lmntServer(t *testing.T, status int, pcm []byte) (*http.Client, func() *captured) {
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

	client := &http.Client{Transport: rewriteHost{target: srv.URL, base: srv.Client().Transport}}
	return client, func() *captured {
		select {
		case c := <-got:
			return c
		default:
			t.Fatal("the endpoint was never called")
			return nil
		}
	}
}

// rewriteHost sends a request to the test server instead of the real host, since
// the endpoint is a package constant.
type rewriteHost struct {
	target string
	base   http.RoundTripper
}

func (r rewriteHost) RoundTrip(req *http.Request) (*http.Response, error) {
	u, err := url.Parse(r.target)
	if err != nil {
		return nil, err
	}
	req = req.Clone(req.Context())
	req.URL.Scheme, req.URL.Host, req.Host = u.Scheme, u.Host, u.Host
	return r.base.RoundTrip(req)
}

// synth builds a synthesizer with the defaults NewTTS fills in.
func synth(client *http.Client, opts ...func(*Config)) *synthesizer {
	cfg := Config{
		APIKey:     "test-key",
		Model:      defaultModel,
		Voice:      defaultVoice,
		Lang:       defaultLang,
		SampleRate: defaultSampleRate,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &synthesizer{cfg: cfg, http: client}
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

func TestRequestCarriesTheTextAndVoice(t *testing.T) {
	want := []byte{1, 2, 3, 4, 5, 6}
	client, seen := lmntServer(t, http.StatusOK, want)

	got, err := run(t, synth(client), "hello there")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("audio = %v, want the PCM the endpoint streamed", got)
	}

	c := seen()
	if c.header.Get("X-API-Key") != "test-key" {
		t.Errorf("X-API-Key = %q, want the configured key", c.header.Get("X-API-Key"))
	}
	if c.body["text"] != "hello there" {
		t.Errorf("text = %v, want the sentence", c.body["text"])
	}
	if c.body["voice"] != defaultVoice {
		t.Errorf("voice = %v, want %q", c.body["voice"], defaultVoice)
	}
	if c.body["model"] != defaultModel {
		t.Errorf("model = %v, want %q", c.body["model"], defaultModel)
	}
	if c.body["language"] != defaultLang {
		t.Errorf("language = %v, want %q", c.body["language"], defaultLang)
	}
	// Raw PCM is what streams straight downstream; a container would not.
	if c.body["format"] != "raw" {
		t.Errorf("format = %v, want raw", c.body["format"])
	}
	if c.body["sample_rate"] != float64(defaultSampleRate) {
		t.Errorf("sample_rate = %v, want %d", c.body["sample_rate"], defaultSampleRate)
	}
}

// TestRequestCarriesTheConfiguredVoiceAndModel checks a caller's own choices
// reach the API rather than the defaults.
func TestRequestCarriesTheConfiguredVoiceAndModel(t *testing.T) {
	client, seen := lmntServer(t, http.StatusOK, nil)

	s := synth(client, func(c *Config) {
		c.Voice, c.Model, c.Lang, c.SampleRate = "morgan", "aurora", "fr", 16000
	})
	if _, err := run(t, s, "bonjour"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}

	c := seen()
	if c.body["voice"] != "morgan" || c.body["model"] != "aurora" {
		t.Errorf("voice/model = %v/%v, want the configured ones", c.body["voice"], c.body["model"])
	}
	if c.body["language"] != "fr" {
		t.Errorf("language = %v, want fr", c.body["language"])
	}
	if c.body["sample_rate"] != float64(16000) {
		t.Errorf("sample_rate = %v, want 16000", c.body["sample_rate"])
	}
}

// TestReportsAFailedRequest checks a non-OK reply is reported rather than read
// as silence.
func TestReportsAFailedRequest(t *testing.T) {
	client, _ := lmntServer(t, http.StatusUnauthorized, nil)
	if _, err := run(t, synth(client), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unauthorized reply")
	}
}

// TestSampleRate checks the rate the emitted frames are stamped with is the one
// asked of LMNT.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{cfg: Config{SampleRate: 16000}}).SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}
