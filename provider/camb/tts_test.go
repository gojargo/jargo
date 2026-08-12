package camb

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
)

// Tests for the synthesis request. Camb.ai takes the text, a numeric voice and
// a region-qualified language, and streams raw PCM back at the rate the chosen
// MARS model runs at.

// captured is the request the fake endpoint saw.
type captured struct {
	header http.Header
	body   map[string]any
}

// cambServer stands in for Camb.ai's endpoint, replying with pcm.
func cambServer(t *testing.T, status int, pcm []byte) (url string, seen func() *captured) {
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
func synth(url string, opts ...func(*Config)) *synthesizer {
	cfg := Config{APIKey: "test-key", URL: url, Model: defaultModel, VoiceID: defaultVoiceID}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = modelSampleRate(cfg.Model)
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

func TestRequestCarriesTheTextAndVoice(t *testing.T) {
	want := []byte{1, 2, 3, 4}
	url, seen := cambServer(t, http.StatusOK, want)

	got, err := run(t, synth(url), "hello there")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("audio = %v, want the PCM the endpoint streamed", got)
	}

	c := seen()
	if c.header.Get("x-api-key") != "test-key" {
		t.Errorf("x-api-key = %q, want the configured key", c.header.Get("x-api-key"))
	}
	if c.body["text"] != "hello there" {
		t.Errorf("text = %v, want the sentence", c.body["text"])
	}
	if c.body["voice_id"] != float64(defaultVoiceID) {
		t.Errorf("voice_id = %v, want %d", c.body["voice_id"], defaultVoiceID)
	}
	if c.body["speech_model"] != defaultModel {
		t.Errorf("speech_model = %v, want %q", c.body["speech_model"], defaultModel)
	}
	if c.body["language"] != "en-us" {
		t.Errorf("language = %v, want the default en-us", c.body["language"])
	}

	out, _ := c.body["output_configuration"].(map[string]any)
	if out["format"] != "pcm_s16le" {
		t.Errorf("format = %v, want pcm_s16le", out["format"])
	}
	if out["sample_rate"] != float64(flashSampleRate) {
		t.Errorf("sample_rate = %v, want the flash model's %d", out["sample_rate"], flashSampleRate)
	}
}

// TestSampleRateFollowsTheModel checks each MARS model is asked for the rate it
// actually runs at, since asking for another makes it resample.
func TestSampleRateFollowsTheModel(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{model: "mars-flash", want: flashSampleRate},
		{model: "mars-pro", want: proSampleRate},
		{model: "mars-instruct", want: flashSampleRate},
		{model: "something-else", want: flashSampleRate},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := modelSampleRate(tt.model); got != tt.want {
				t.Errorf("modelSampleRate(%q) = %d, want %d", tt.model, got, tt.want)
			}
		})
	}
}

// TestUserInstructionsOnlyGoToTheInstructModel checks the delivery guidance is
// sent to the one model that honors it and left out of the others, which reject
// it.
func TestUserInstructionsOnlyGoToTheInstructModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{model: instructModel, want: true},
		{model: "mars-flash"},
		{model: "mars-pro"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			url, seen := cambServer(t, http.StatusOK, nil)
			s := synth(url, func(c *Config) {
				c.Model = tt.model
				c.UserInstructions = "speak warmly"
			})
			if _, err := run(t, s, "hello"); err != nil {
				t.Fatalf("RunTTS: %v", err)
			}

			v, ok := seen().body["user_instructions"]
			if ok != tt.want {
				t.Errorf("user_instructions present = %v (%v), want %v", ok, v, tt.want)
			}
		})
	}
}

// TestTextIsTruncated checks input past what Camb.ai accepts is cut rather than
// sent whole, since the request would be rejected and the sentence lost.
func TestTextIsTruncated(t *testing.T) {
	url, seen := cambServer(t, http.StatusOK, nil)

	long := strings.Repeat("a", maxTextLen+500)
	if _, err := run(t, synth(url), long); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	text, _ := seen().body["text"].(string)
	if len(text) != maxTextLen {
		t.Errorf("text was %d characters, want it cut to %d", len(text), maxTextLen)
	}
}

// TestCambLanguage checks the codes Camb.ai names languages by: it wants a
// region-qualified code, so a base language is given the region it defaults to
// and anything unmapped falls back to English.
func TestCambLanguage(t *testing.T) {
	tests := []struct {
		in   language.Language
		want string
	}{
		{in: "", want: "en-us"},
		{in: language.English, want: "en-us"},
		{in: language.EnglishGB, want: "en-gb"},
		{in: language.EnglishAU, want: "en-au"},
		{in: language.French, want: "fr-fr"},
		{in: language.FrenchCA, want: "fr-ca"},
		{in: language.Spanish, want: "es-es"},
		{in: language.SpanishMX, want: "es-mx"},
		{in: language.PortugueseBR, want: "pt-br"},
		{in: language.ChineseTW, want: "zh-tw"},
		{in: language.Chinese, want: "zh-cn"},
	}
	for _, tt := range tests {
		if got := cambLanguage(tt.in); got != tt.want {
			t.Errorf("cambLanguage(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestReportsAFailedRequest checks a non-OK reply is reported rather than read
// as silence.
func TestReportsAFailedRequest(t *testing.T) {
	url, _ := cambServer(t, http.StatusUnauthorized, nil)
	if _, err := run(t, synth(url), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unauthorized reply")
	}
}

// TestReportsAnUnreachableEndpoint checks a transport failure is reported.
func TestReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := run(t, synth("http://127.0.0.1:1"), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unreachable endpoint")
	}
}

// TestSampleRate checks the rate the emitted frames are stamped with.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{cfg: Config{SampleRate: 16000}}).SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}
