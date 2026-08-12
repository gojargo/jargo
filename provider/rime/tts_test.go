package rime

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

// Tests for the synthesis request. Rime takes the text and the voice over HTTP
// and streams raw PCM back, and which of the model controls it accepts depends
// on the model: each one rejects the other's.

// captured is the request the fake endpoint saw.
type captured struct {
	header http.Header
	body   map[string]any
}

// rimeServer stands in for Rime's endpoint, replying with pcm.
func rimeServer(t *testing.T, status int, pcm []byte) (*http.Client, func() *captured) {
	t.Helper()
	got := make(chan *captured, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := &captured{header: r.Header.Clone(), body: map[string]any{}}
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
	want := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	client, seen := rimeServer(t, http.StatusOK, want)

	got, err := run(t, synth(client, func(c *Config) { c.Speaker = "astra" }), "hello there")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("audio = %v, want the PCM the endpoint streamed", got)
	}

	c := seen()
	if c.header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", c.header.Get("Authorization"))
	}
	// Rime returns raw PCM rather than a container only when asked to.
	if c.header.Get("Accept") != "audio/pcm" {
		t.Errorf("Accept = %q, want audio/pcm", c.header.Get("Accept"))
	}
	if c.body["text"] != "hello there" {
		t.Errorf("text = %v, want the sentence", c.body["text"])
	}
	if c.body["speaker"] != "astra" {
		t.Errorf("speaker = %v, want the configured voice", c.body["speaker"])
	}
	if c.body["modelId"] != defaultModel {
		t.Errorf("modelId = %v, want %q", c.body["modelId"], defaultModel)
	}
	if c.body["samplingRate"] != float64(defaultSampleRate) {
		t.Errorf("samplingRate = %v, want %d", c.body["samplingRate"], defaultSampleRate)
	}
	if c.body["lang"] != defaultLang {
		t.Errorf("lang = %v, want %q", c.body["lang"], defaultLang)
	}
}

// TestDefaults checks what a caller who names only their key sends. Rime's
// current model is coda, it names languages with three letters, and it picks the
// voice itself when none is given.
func TestDefaults(t *testing.T) {
	if defaultModel != "coda" {
		t.Errorf("defaultModel = %q, want coda", defaultModel)
	}
	if defaultLang != "eng" {
		t.Errorf("defaultLang = %q, want the three-letter code eng", defaultLang)
	}

	client, seen := rimeServer(t, http.StatusOK, nil)
	if _, err := run(t, synth(client), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}

	// An unset voice is sent as null, which is how Rime is asked to choose.
	c := seen()
	if v, ok := c.body["speaker"]; !ok || v != nil {
		t.Errorf("speaker = %v (present %v), want null", v, ok)
	}
}

// TestControlsFollowTheModel is the regression test for the defect this file
// was written for: the controls sent have to be the ones the chosen model
// accepts, since each model rejects the other's.
func TestControlsFollowTheModel(t *testing.T) {
	f, b := 0.5, true

	withAll := func(c *Config) {
		c.RepetitionPenalty, c.Temperature, c.TopP, c.TimeScaleFactor = &f, &f, &f, &f
		c.ReduceLatency, c.PauseBetweenBrackets, c.PhonemizeBetweenBrackets = &b, &b, &b
	}

	coda := []string{"repetition_penalty", "temperature", "top_p", "timeScaleFactor"}
	mist := []string{"reduceLatency", "pauseBetweenBrackets", "phonemizeBetweenBrackets"}

	tests := []struct {
		model   string
		want    []string
		notWant []string
	}{
		{model: "coda", want: coda, notWant: mist},
		{model: "mistv2", want: mist, notWant: coda},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			client, seen := rimeServer(t, http.StatusOK, nil)
			s := synth(client, func(c *Config) { c.Model = tt.model; withAll(c) })
			if _, err := run(t, s, "hello"); err != nil {
				t.Fatalf("RunTTS: %v", err)
			}

			c := seen()
			for _, key := range tt.want {
				if _, ok := c.body[key]; !ok {
					t.Errorf("%s was not sent to %s, want it", key, tt.model)
				}
			}
			for _, key := range tt.notWant {
				if v, ok := c.body[key]; ok {
					t.Errorf("%s was sent to %s as %v, want it left out", key, tt.model, v)
				}
			}
		})
	}
}

// TestSpeedIsSentToEveryModel checks the speed controls are not gated: every
// model takes them.
func TestSpeedIsSentToEveryModel(t *testing.T) {
	speed := 1.2
	for _, model := range []string{"coda", "mistv2"} {
		t.Run(model, func(t *testing.T) {
			client, seen := rimeServer(t, http.StatusOK, nil)
			s := synth(client, func(c *Config) {
				c.Model = model
				c.SpeedAlpha = &speed
				c.InlineSpeedAlpha = "1.0 2.0"
			})
			if _, err := run(t, s, "hello"); err != nil {
				t.Fatalf("RunTTS: %v", err)
			}

			c := seen()
			if c.body["speedAlpha"] != speed {
				t.Errorf("speedAlpha = %v, want %v", c.body["speedAlpha"], speed)
			}
			if c.body["inlineSpeedAlpha"] != "1.0 2.0" {
				t.Errorf("inlineSpeedAlpha = %v, want the markup", c.body["inlineSpeedAlpha"])
			}
		})
	}
}

// TestUnsetControlsAreOmitted checks a control nobody set is left out rather
// than sent as a zero, which Rime would read as a request to use it.
func TestUnsetControlsAreOmitted(t *testing.T) {
	client, seen := rimeServer(t, http.StatusOK, nil)
	if _, err := run(t, synth(client), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}

	c := seen()
	for _, key := range []string{
		"repetition_penalty", "temperature", "top_p", "timeScaleFactor",
		"speedAlpha", "inlineSpeedAlpha",
	} {
		if v, ok := c.body[key]; ok {
			t.Errorf("%s was sent as %v, want it omitted", key, v)
		}
	}
}

// TestLangIsOmittedWhenUnset checks a caller who clears the language leaves it
// to Rime rather than sending an empty one.
func TestLangIsOmittedWhenUnset(t *testing.T) {
	client, seen := rimeServer(t, http.StatusOK, nil)
	if _, err := run(t, synth(client, func(c *Config) { c.Lang = "" }), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if v, ok := seen().body["lang"]; ok {
		t.Errorf("lang = %v, want it omitted when unset", v)
	}
}

// TestReportsAFailedRequest checks a non-OK reply is reported rather than read
// as silence.
func TestReportsAFailedRequest(t *testing.T) {
	client, _ := rimeServer(t, http.StatusUnauthorized, nil)
	if _, err := run(t, synth(client), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unauthorized reply")
	}
}

// TestSampleRate checks the rate the emitted frames are stamped with is the one
// asked of Rime.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{cfg: Config{SampleRate: 16000}}).SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}
