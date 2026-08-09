package sarvam

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
)

// ttsSession is what the fake synthesis endpoint saw.
type ttsSession struct {
	query    url.Values
	header   http.Header
	messages chan map[string]any
}

// ttsServer starts a fake Sarvam synthesis endpoint that reads the client's
// messages and replies with the scripted ones once the flush arrives.
func ttsServer(t *testing.T, reply []map[string]any) (endpoint string, got *ttsSession) {
	t.Helper()
	got = &ttsSession{messages: make(chan map[string]any, 8)}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.query = r.URL.Query()
		got.header = r.Header.Clone()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) != nil {
				continue
			}
			select {
			case got.messages <- msg:
			default:
			}
			// The flush is what says the text is complete, so the audio follows it.
			if msg["type"] != "flush" {
				continue
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
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// audioMessage is a base64 audio chunk as Sarvam sends it.
func audioMessage(pcm []byte) map[string]any {
	return map[string]any{
		"type": "audio",
		"data": map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm)},
	}
}

// finalEvent is the completion event that ends a synthesis.
func finalEvent() map[string]any {
	return map[string]any{"type": "event", "data": map[string]any{"event_type": "final"}}
}

// synthesizerAt builds the synthesizer NewTTS would build, pointed at endpoint.
func synthesizerAt(t *testing.T, endpoint string, cfg TTSConfig) *synthesizer {
	t.Helper()
	cfg.URL = endpoint
	return newSynthesizer(cfg)
}

// collect runs one synthesis and returns the PCM the frames carried.
func collect(t *testing.T, s *synthesizer, text string) []byte {
	t.Helper()
	var pcm bytes.Buffer
	err := s.RunTTS(t.Context(), text, "", func(f frames.Frame) error {
		audio, ok := f.(*frames.TTSAudioRawFrame)
		if !ok {
			t.Errorf("yielded %T, want a TTSAudioRawFrame", f)
			return nil
		}
		if audio.SampleRate != s.SampleRate() {
			t.Errorf("frame rate = %d, want the configured %d", audio.SampleRate, s.SampleRate())
		}
		pcm.Write(audio.Audio)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	return pcm.Bytes()
}

// TestTTSConfigValidate pins the credential the synthesis API requires.
func TestTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: TTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewTTS checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewTTS(t *testing.T) {
	providertest.Service(t, "SarvamTTS", NewTTS(TTSConfig{APIKey: "k"}))
}

// TestRunTTSSessionShape checks the session is authorized, asks for the
// completion event that ends a synthesis, and sends config, text and flush in
// that order. The flush is what tells Sarvam the text is complete.
func TestRunTTSSessionShape(t *testing.T) {
	want := []byte{0x11, 0x22, 0x33, 0x44}
	endpoint, got := ttsServer(t, []map[string]any{audioMessage(want), finalEvent()})

	s := synthesizerAt(t, endpoint, TTSConfig{APIKey: "test-key"})
	if pcm := collect(t, s, "hello there"); !bytes.Equal(pcm, want) {
		t.Errorf("PCM = % x, want % x", pcm, want)
	}

	if h := got.header.Get("api-subscription-key"); h != "test-key" {
		t.Errorf("api-subscription-key = %q, want the configured key", h)
	}
	if got.query.Get("send_completion_event") != "true" {
		t.Errorf("send_completion_event = %q, want true so the synthesis knows when to stop",
			got.query.Get("send_completion_event"))
	}
	if got.query.Get("model") != defaultTTSModel {
		t.Errorf("model = %q, want %q", got.query.Get("model"), defaultTTSModel)
	}

	config := <-got.messages
	if config["type"] != "config" {
		t.Fatalf("first message = %v, want the config", config)
	}
	data, _ := config["data"].(map[string]any)
	// Sarvam takes the rate as a string, not a number.
	wantRate := strconv.Itoa(ttsModelConfigs[defaultTTSModel].defaultSampleRate)
	if data["speech_sample_rate"] != wantRate {
		t.Errorf("speech_sample_rate = %v, want the model default %q as a string",
			data["speech_sample_rate"], wantRate)
	}
	if data["target_language_code"] != "en-IN" {
		t.Errorf("target_language_code = %v, want the default en-IN", data["target_language_code"])
	}
	if data["model"] != defaultTTSModel {
		t.Errorf("model = %v, want %q", data["model"], defaultTTSModel)
	}

	text := <-got.messages
	if text["type"] != "text" {
		t.Fatalf("second message = %v, want the text", text)
	}
	textData, _ := text["data"].(map[string]any)
	if textData["text"] != "hello there" {
		t.Errorf("text = %v, want the text to speak", textData["text"])
	}

	flush := <-got.messages
	if flush["type"] != "flush" {
		t.Fatalf("third message = %v, want the flush", flush)
	}
}

// TestRunTTSJoinsChunks checks every audio chunk before the completion event
// reaches the caller, and that a message the service does not model is skipped
// rather than ending the synthesis.
func TestRunTTSJoinsChunks(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		audioMessage([]byte{1, 2}),
		{"type": "event", "data": map[string]any{"event_type": "started"}},
		audioMessage([]byte{3, 4}),
		finalEvent(),
		audioMessage([]byte{5, 6}),
	})

	s := synthesizerAt(t, endpoint, TTSConfig{APIKey: "k"})
	if pcm := collect(t, s, "hi"); !bytes.Equal(pcm, []byte{1, 2, 3, 4}) {
		t.Errorf("PCM = % x, want the chunks up to the completion event", pcm)
	}
}

// TestRunTTSServerError checks an error message is reported rather than leaving
// the synthesis waiting for audio that will not come.
func TestRunTTSServerError(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"type": "error", "data": map[string]any{"message": "unknown speaker"}},
	})

	s := synthesizerAt(t, endpoint, TTSConfig{APIKey: "k"})
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error { return nil })
	if !errors.Is(err, errTTSProtocol) {
		t.Fatalf("RunTTS error = %v, want errTTSProtocol", err)
	}
	if !strings.Contains(err.Error(), "unknown speaker") {
		t.Errorf("error = %v, want it to carry the server's message", err)
	}
}

// TestRunTTSDialError checks a session that cannot be opened is reported rather
// than treated as an empty synthesis.
func TestRunTTSDialError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	s := synthesizerAt(t, "ws"+strings.TrimPrefix(srv.URL, "http"), TTSConfig{APIKey: "k"})
	err := s.RunTTS(t.Context(), "hi", "", func(frames.Frame) error {
		t.Error("a frame was yielded for a session that never opened")
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS on a refused session = nil, want an error")
	}
}

// TestConfigMessageOmitsUnsupportedControls checks a voice control is sent only
// to a model that understands it, and left off otherwise rather than being sent
// and ignored.
func TestConfigMessageOmitsUnsupportedControls(t *testing.T) {
	pitch, loudness, temperature := 1.1, 0.9, 0.5
	cfg := TTSConfig{APIKey: "k", Pitch: &pitch, Loudness: &loudness, Temperature: &temperature}

	s := newSynthesizer(cfg)
	var msg map[string]any
	if err := json.Unmarshal(s.configMessage(), &msg); err != nil {
		t.Fatalf("decoding the config message: %v", err)
	}
	data, _ := msg["data"].(map[string]any)

	mc := ttsModelConfigs[defaultTTSModel]
	for key, supported := range map[string]bool{
		"pitch":       mc.supportsPitch,
		"loudness":    mc.supportsLoudness,
		"temperature": mc.supportsTemperature,
	} {
		_, present := data[key]
		if present != supported {
			t.Errorf("%s present = %v, want %v for the default model", key, present, supported)
		}
	}
}
