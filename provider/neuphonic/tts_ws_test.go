package neuphonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
)

// Tests for the synthesis session. Neuphonic streams base64 PCM over a socket
// per language, and the text is sent with a sentinel that tells the provider the
// utterance is complete; the reply ends on a message flagged stop.

// ttsSession is what the fake endpoint saw.
type ttsSession struct {
	path   string
	query  url.Values
	header http.Header
	sent   map[string]any
}

// ttsServer starts a fake Neuphonic endpoint that reads the request and replies
// with the scripted messages.
func ttsServer(t *testing.T, reply []map[string]any) (endpoint string, seen func() *ttsSession) {
	t.Helper()
	sessions := make(chan *ttsSession, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := &ttsSession{path: r.URL.Path, query: r.URL.Query(), header: r.Header.Clone()}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		got.sent = map[string]any{}
		if err := json.Unmarshal(data, &got.sent); err != nil {
			t.Errorf("decoding the request: %v", err)
			return
		}
		select {
		case sessions <- got:
		default:
		}

		for _, m := range reply {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), func() *ttsSession {
		select {
		case s := <-sessions:
			return s
		case <-time.After(2 * time.Second):
			t.Fatal("the endpoint was never called")
			return nil
		}
	}
}

// audio is one chunk as Neuphonic sends it; stop marks the last message.
func audio(pcm []byte, stop bool) map[string]any {
	return map[string]any{"data": map[string]any{
		"audio": base64.StdEncoding.EncodeToString(pcm),
		"stop":  stop,
	}}
}

// synth builds a synthesizer pointed at endpoint with the defaults NewTTS fills
// in, so the request under test is the real one.
func synth(endpoint string, opts ...func(*Config)) *synthesizer {
	cfg := Config{
		APIKey:     "test-key",
		URL:        endpoint,
		SampleRate: defaultSampleRate,
		Encoding:   defaultEncoding,
		Speed:      defaultSpeed,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &synthesizer{cfg: cfg}
}

// collect runs a synthesis and returns the PCM it emitted.
func collect(t *testing.T, s *synthesizer, text string) ([]byte, error) {
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

func TestSynthesizeStreamsTheAudio(t *testing.T) {
	want := []byte{1, 2, 3, 4, 5, 6}
	endpoint, seen := ttsServer(t, []map[string]any{
		audio(want[:3], false), audio(want[3:], true),
	})

	got, err := collect(t, synth(endpoint), "hello there")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("audio = %v, want the chunks joined up: %v", got, want)
	}

	s := seen()
	if s.header.Get("x-api-key") != "test-key" {
		t.Errorf("x-api-key = %q, want the configured key", s.header.Get("x-api-key"))
	}
	if s.path != "/speak/en" {
		t.Errorf("path = %s, want the language in the path", s.path)
	}
	for field, want := range map[string]string{
		"lang_code":     "en",
		"speed":         "1",
		"encoding":      defaultEncoding,
		"sampling_rate": "22050",
	} {
		if got := s.query.Get(field); got != want {
			t.Errorf("%s = %q, want %q", field, got, want)
		}
	}
	if s.query.Has("voice_id") {
		t.Error("voice_id was sent though none was configured")
	}

	// The sentinel is what tells the provider the utterance is complete.
	if s.sent["text"] != "hello there <STOP>" {
		t.Errorf("text = %v, want it sent with the stop sentinel", s.sent["text"])
	}
}

// TestSynthesizeSendsTheVoice checks a configured voice reaches the endpoint.
func TestSynthesizeSendsTheVoice(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{audio(nil, true)})

	if _, err := collect(t, synth(endpoint, func(c *Config) {
		c.VoiceID = "voice-9"
	}), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if got := seen().query.Get("voice_id"); got != "voice-9" {
		t.Errorf("voice_id = %q, want voice-9", got)
	}
}

// TestSynthesizeStopsOnTheStopFlag checks the session ends on the message
// flagged stop rather than waiting for the socket to close.
func TestSynthesizeStopsOnTheStopFlag(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		audio([]byte{1, 1}, false), audio([]byte{2, 2}, true), audio([]byte{3, 3}, false),
	})

	got, err := collect(t, synth(endpoint), "hello")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(got) != 4 {
		t.Errorf("audio = %v, want only what arrived up to the stop", got)
	}
}

// TestSynthesizeReportsProviderErrors checks the errors the provider reports are
// surfaced together, carrying what it said.
func TestSynthesizeReportsProviderErrors(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"errors": []string{"voice not found", "bad speed"}},
	})

	_, err := collect(t, synth(endpoint), "hello")
	if err == nil {
		t.Fatal("RunTTS accepted a failed synthesis")
	}
	if !errors.Is(err, errProtocol) {
		t.Errorf("error = %v, want it reported as a protocol error", err)
	}
	if !strings.Contains(err.Error(), "voice not found") ||
		!strings.Contains(err.Error(), "bad speed") {
		t.Errorf("error = %v, want it to carry both messages", err)
	}
}

// TestSynthesizeSkipsWhatCarriesNoAudio checks a message that is not the
// expected JSON, and one carrying no data at all, are stepped over.
func TestSynthesizeSkipsWhatCarriesNoAudio(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"message": "connected"},
		{"data": map[string]any{"audio": "", "stop": false}},
		audio([]byte{7, 7}, true),
	})

	got, err := collect(t, synth(endpoint), "hello")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("audio = %v, want the one real chunk", got)
	}
}

// TestSynthesizeReportsUndecodableAudio checks audio that is not base64 is
// reported rather than played as rubbish.
func TestSynthesizeReportsUndecodableAudio(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"data": map[string]any{"audio": "not base64!!"}},
	})

	if _, err := collect(t, synth(endpoint), "hello"); err == nil {
		t.Fatal("RunTTS accepted audio that is not base64")
	}
}

// TestSynthesizeReportsAnUnreachableEndpoint checks a dial that fails is
// reported.
func TestSynthesizeReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := collect(t, synth("ws://127.0.0.1:1"), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unreachable endpoint")
	}
}

// TestNeuphonicLanguage checks the codes the provider names languages by. Hindi
// is the one it spells in capitals; everything else is the base code, and an
// unset language is English.
func TestNeuphonicLanguage(t *testing.T) {
	tests := []struct {
		in   language.Language
		want string
	}{
		{in: "", want: "en"},
		{in: language.English, want: "en"},
		{in: language.EnglishUS, want: "en"},
		{in: language.Hindi, want: "HI"},
		{in: language.French, want: "fr"},
		{in: language.Spanish, want: "es"},
	}
	for _, tt := range tests {
		if got := neuphonicLanguage(tt.in); got != tt.want {
			t.Errorf("neuphonicLanguage(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestSampleRate checks the rate the frames are stamped with is the one asked
// of the provider.
func TestSampleRate(t *testing.T) {
	s := &synthesizer{cfg: Config{SampleRate: 16000}}
	if got := s.SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}
