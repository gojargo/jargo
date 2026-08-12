package gradium

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
)

// Tests for the synthesis session. Gradium takes a setup message pinning the
// voice and the output format, then the text, then an end-of-stream that makes
// it flush; the audio comes back base64 until it says the stream has ended.

// ttsSession is what the fake endpoint saw.
type ttsSession struct {
	header http.Header
	sent   []map[string]any
}

// ttsServer starts a fake Gradium TTS endpoint that reads the three request
// messages and replies with the scripted ones.
func ttsServer(t *testing.T, reply []map[string]any) (endpoint string, seen func() *ttsSession) {
	t.Helper()
	sessions := make(chan *ttsSession, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := &ttsSession{header: r.Header.Clone()}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The request is setup, then the text, then end-of-stream.
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

// ttsAudio is one audio chunk as Gradium sends it.
func ttsAudio(pcm []byte) map[string]any {
	return map[string]any{
		msgType: msgAudio,
		"audio": base64.StdEncoding.EncodeToString(pcm),
	}
}

// endStream ends the synthesis.
func endStream() map[string]any { return map[string]any{msgType: msgEndStream} }

// ttsSynth builds a synthesizer pointed at endpoint with the defaults NewTTS
// fills in, so the request under test is the real one.
func ttsSynth(endpoint string, opts ...func(*TTSConfig)) *ttsSynthesizer {
	cfg := TTSConfig{APIKey: "test-key", URL: endpoint, VoiceID: defaultVoiceID}
	for _, o := range opts {
		o(&cfg)
	}
	return &ttsSynthesizer{cfg: cfg}
}

// collectTTS runs a synthesis and returns the PCM it emitted.
func collectTTS(t *testing.T, s *ttsSynthesizer, text string) ([]byte, error) {
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

func TestTTSStreamsTheAudio(t *testing.T) {
	want := []byte{1, 2, 3, 4, 5, 6}
	endpoint, seen := ttsServer(t, []map[string]any{
		ttsAudio(want[:3]), ttsAudio(want[3:]), endStream(),
	})

	got, err := collectTTS(t, ttsSynth(endpoint), "hello there")
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
	if len(s.sent) != 3 {
		t.Fatalf("sent %d messages, want three", len(s.sent))
	}

	setup := s.sent[0]
	if setup[msgType] != "setup" {
		t.Errorf("first message = %v, want the setup", setup)
	}
	if setup["output_format"] != encPCM {
		t.Errorf("output_format = %v, want %q", setup["output_format"], encPCM)
	}
	if setup["voice_id"] != defaultVoiceID {
		t.Errorf("voice_id = %v, want the default voice", setup["voice_id"])
	}
	// The socket is reused for the next sentence, so the provider must not
	// close it when this one flushes.
	if setup["close_ws_on_eos"] != false {
		t.Errorf("close_ws_on_eos = %v, want false", setup["close_ws_on_eos"])
	}
	if _, ok := setup["json_config"]; ok {
		t.Error("json_config was sent though none was configured")
	}

	if s.sent[1][msgType] != msgText || s.sent[1][msgText] != "hello there" {
		t.Errorf("second message = %v, want the text", s.sent[1])
	}
	// Without this the provider waits for more text and never flushes.
	if s.sent[2][msgType] != msgEndStream {
		t.Errorf("third message = %v, want the end of stream", s.sent[2])
	}

	// Every message labels the same synthesis context.
	for i, m := range s.sent {
		if m[keyClientReqID] != ttsClientReqID {
			t.Errorf("message %d labels context %v, want %q", i, m[keyClientReqID], ttsClientReqID)
		}
	}
}

// TestTTSSendsTheJSONConfig checks the optional model configuration is
// forwarded on the setup when it is given.
func TestTTSSendsTheJSONConfig(t *testing.T) {
	endpoint, seen := ttsServer(t, []map[string]any{endStream()})

	if _, err := collectTTS(t, ttsSynth(endpoint, func(c *TTSConfig) {
		c.JSONConfig = `{"speed":1.2}`
	}), "hello"); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if got := seen().sent[0]["json_config"]; got != `{"speed":1.2}` {
		t.Errorf("json_config = %v, want the configured string", got)
	}
}

// TestTTSStopsOnEndOfStream checks the session ends when the provider says the
// stream has, rather than waiting for the socket to close.
func TestTTSStopsOnEndOfStream(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		ttsAudio([]byte{1, 1}), endStream(), ttsAudio([]byte{2, 2}),
	})

	got, err := collectTTS(t, ttsSynth(endpoint), "hello")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("audio = %v, want only what arrived before the end of stream", got)
	}
}

// TestTTSReportsAnError checks a failure is surfaced carrying what the provider
// said.
func TestTTSReportsAnError(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{msgType: msgError, "message": "unknown voice"},
	})

	_, err := collectTTS(t, ttsSynth(endpoint), "hello")
	if err == nil {
		t.Fatal("RunTTS accepted a failed synthesis")
	}
	if !errors.Is(err, errTTSProtocol) {
		t.Errorf("error = %v, want it reported as a protocol error", err)
	}
	if !strings.Contains(err.Error(), "unknown voice") {
		t.Errorf("error = %v, want it to carry what the provider said", err)
	}
}

// TestTTSSkipsWhatItCannotRead checks a message of a kind the service does not
// act on is stepped over rather than ending the session.
func TestTTSSkipsWhatItCannotRead(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{msgType: "ready"},
		ttsAudio([]byte{9, 9}),
		endStream(),
	})

	got, err := collectTTS(t, ttsSynth(endpoint), "hello")
	if err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("audio = %v, want the one real chunk", got)
	}
}

// TestTTSReportsUndecodableAudio checks audio that is not base64 is reported
// rather than played as rubbish.
func TestTTSReportsUndecodableAudio(t *testing.T) {
	endpoint, _ := ttsServer(t, []map[string]any{
		{msgType: msgAudio, "audio": "not base64!!"},
	})

	if _, err := collectTTS(t, ttsSynth(endpoint), "hello"); err == nil {
		t.Fatal("RunTTS accepted audio that is not base64")
	}
}

// TestTTSReportsAnUnreachableEndpoint checks a dial that fails is reported.
func TestTTSReportsAnUnreachableEndpoint(t *testing.T) {
	if _, err := collectTTS(t, ttsSynth("ws://127.0.0.1:1"), "hello"); err == nil {
		t.Fatal("RunTTS accepted an unreachable endpoint")
	}
}

// TestTTSSampleRate checks the fixed rate Gradium streams at, which is what the
// emitted frames are stamped with.
func TestTTSSampleRate(t *testing.T) {
	if got := (&ttsSynthesizer{}).SampleRate(); got != ttsSampleRate {
		t.Errorf("SampleRate() = %d, want %d", got, ttsSampleRate)
	}
}
