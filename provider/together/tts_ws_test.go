package together

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
)

// Tests for the synthesis session. Together streams 24 kHz PCM over an
// OpenAI-compatible realtime socket: the voice is pinned on the session, the
// text is buffered and committed, and the audio comes back base64 in deltas
// until the session says it is done.

// ttsSession is what the fake endpoint saw.
type ttsSession struct {
	query  url.Values
	header http.Header
	sent   []map[string]any
}

// ttsServer starts a fake Together TTS endpoint that reads the request messages
// and replies with the scripted events.
func ttsServer(t *testing.T, reply []map[string]any) (endpoint string, seen func() *ttsSession) {
	t.Helper()
	sessions := make(chan *ttsSession, 1)

	srv := httptestServer(t, func(w http.ResponseWriter, r *http.Request) {
		got := &ttsSession{query: r.URL.Query(), header: r.Header.Clone()}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The request is three messages: pin the voice, append, commit.
		for range 3 {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			m := map[string]any{}
			if err := json.Unmarshal(data, &m); err != nil {
				t.Errorf("decoding a request message: %v", err)
				return
			}
			got.sent = append(got.sent, m)
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
	})

	return "ws" + strings.TrimPrefix(srv, "http"), func() *ttsSession {
		select {
		case s := <-sessions:
			return s
		case <-time.After(2 * time.Second):
			t.Fatal("the endpoint was never called")
			return nil
		}
	}
}

// delta is an audio delta as Together sends it.
func delta(pcm []byte) map[string]any {
	return map[string]any{
		"type":  "conversation.item.audio_output.delta",
		"delta": base64.StdEncoding.EncodeToString(pcm),
	}
}

// done ends the synthesis.
func done() map[string]any {
	return map[string]any{"type": "conversation.item.audio_output.done"}
}

// synth builds a synthesizer pointed at endpoint with the defaults NewTTS fills
// in, so the request under test is the real one.
func synth(endpoint string, opts ...func(*TTSConfig)) *synthesizer {
	cfg := TTSConfig{
		APIKey: "test-key",
		URL:    endpoint,
		Model:  defaultTTSModel,
		Voice:  defaultTTSVoice,
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
	want := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	endpoint, seen := ttsServer(t, []map[string]any{
		delta(want[:4]), delta(want[4:]), done(),
	})

	got, err := collect(t, synth(endpoint), "hello there")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("audio = %v, want the deltas joined up: %v", got, want)
	}

	s := seen()
	if auth := s.header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", auth)
	}
	if s.query.Get("model") != defaultTTSModel {
		t.Errorf("model = %q, want %q", s.query.Get("model"), defaultTTSModel)
	}
	if s.query.Get("voice") != defaultTTSVoice {
		t.Errorf("voice = %q, want %q", s.query.Get("voice"), defaultTTSVoice)
	}
	if s.query.Has("max_partial_length") {
		t.Error("max_partial_length was sent though none was configured")
	}

	// The voice is pinned on the session, then the text is buffered and
	// committed: the provider synthesizes nothing until the commit.
	if len(s.sent) != 3 {
		t.Fatalf("sent %d messages, want three", len(s.sent))
	}
	if s.sent[0]["type"] != "tts_session.updated" {
		t.Errorf("first message = %v, want the session update", s.sent[0])
	}
	session, _ := s.sent[0]["session"].(map[string]any)
	if session["voice"] != defaultTTSVoice {
		t.Errorf("the session update pinned %v, want the voice", session)
	}
	if s.sent[1]["type"] != "input_text_buffer.append" || s.sent[1]["text"] != "hello there" {
		t.Errorf("second message = %v, want the text appended", s.sent[1])
	}
	if s.sent[2]["type"] != "input_text_buffer.commit" {
		t.Errorf("third message = %v, want the commit", s.sent[2])
	}
}

// TestSynthesizeSendsMaxPartialLength checks the optional cap reaches the
// endpoint as a query parameter when it is configured.
func TestSynthesizeSendsMaxPartialLength(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{done()})
	partial := 120

	if _, err := collect(t, synth(endpoint, func(c *TTSConfig) {
		c.MaxPartialLength = &partial
	}), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if got := seen().query.Get("max_partial_length"); got != "120" {
		t.Errorf("max_partial_length = %q, want 120", got)
	}
}

// TestSynthesizeStopsOnDone checks the session ends on the done event without
// waiting for the socket to close.
func TestSynthesizeStopsOnDone(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{delta([]byte{9, 9}), done(), delta([]byte{7, 7})})

	got, err := collect(t, synth(endpoint), "hello")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("audio = %v, want only what arrived before the done event", got)
	}
}

// TestSynthesizeReportsAFailure checks the two ways the provider reports a
// problem are both surfaced, carrying what it said.
func TestSynthesizeReportsAFailure(t *testing.T) {
	for _, evt := range []string{"conversation.item.tts.failed", "error"} {
		t.Run(evt, func(t *testing.T) {
			endpoint, _ := ttsServer(t, []map[string]any{{
				"type":  evt,
				"error": map[string]any{"message": "voice not found", "code": "bad_voice"},
			}})

			_, err := collect(t, synth(endpoint), "hello")
			if err == nil {
				t.Fatal("RunTTS accepted a failed synthesis")
			}
			if !errors.Is(err, errTTS) {
				t.Errorf("error = %v, want it reported as a TTS failure", err)
			}
			if !strings.Contains(err.Error(), "voice not found") {
				t.Errorf("error = %v, want it to carry what the provider said", err)
			}
		})
	}
}

// TestSynthesizeSkipsWhatItCannotRead checks a message that is not the expected
// JSON, and an empty delta, are stepped over rather than ending the session.
func TestSynthesizeSkipsWhatItCannotRead(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"type": "conversation.item.audio_output.delta", "delta": ""},
		{"type": "session.created"},
		delta([]byte{5, 5}),
		done(),
	})

	got, err := collect(t, synth(endpoint), "hello")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("audio = %v, want the one real delta", got)
	}
}

// TestSynthesizeReportsAnUndecodableDelta checks audio that is not base64 is
// reported rather than emitted as rubbish a synthesizer would play.
func TestSynthesizeReportsAnUndecodableDelta(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{"type": "conversation.item.audio_output.delta", "delta": "not base64!!"},
	})

	if _, err := collect(t, synth(endpoint), "hello"); err == nil {
		t.Fatal("RunTTS accepted a delta that is not base64")
	}
}

// TestSynthesizeReportsAnUnreachableEndpoint checks a dial that fails is
// reported.
func TestSynthesizeReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := collect(t, synth("ws://127.0.0.1:1"), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unreachable endpoint")
	}
}

// TestSampleRate checks the fixed rate Together streams at, which is what the
// emitted frames are stamped with.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{}).SampleRate(); got != ttsSampleRate {
		t.Errorf("SampleRate() = %d, want %d", got, ttsSampleRate)
	}
}

// httptestServer starts a test HTTP server and returns its URL.
func httptestServer(t *testing.T, h http.HandlerFunc) string {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv.URL
}
