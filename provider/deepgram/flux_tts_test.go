package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

func TestFluxTTSValidate(t *testing.T) {
	if err := (FluxTTSConfig{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (FluxTTSConfig{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	if err := (FluxTTSConfig{APIKey: "k", SampleRate: 24000}).Validate(); err != nil {
		t.Fatalf("supported sample rate rejected: %v", err)
	}
	err := FluxTTSConfig{APIKey: "k", SampleRate: 22050}.Validate()
	if err == nil || !errors.Is(err, errFluxUnsupportedRate) {
		t.Fatalf("unsupported sample rate should be rejected, got %v", err)
	}
}

func TestFluxTTSDefaults(t *testing.T) {
	cfg := FluxTTSConfig{APIKey: "k"}.withDefaults()
	if cfg.Voice != defaultFluxVoice {
		t.Fatalf("voice = %q, want %q", cfg.Voice, defaultFluxVoice)
	}
	if cfg.SampleRate != defaultTTSSampleRate {
		t.Fatalf("sample_rate = %d, want %d", cfg.SampleRate, defaultTTSSampleRate)
	}
	if cfg.SpeakURL != fluxSpeakURL {
		t.Fatalf("speak URL = %q, want %q", cfg.SpeakURL, fluxSpeakURL)
	}

	// Explicit values are preserved.
	custom := FluxTTSConfig{APIKey: "k", Voice: "flux-other-en", SampleRate: 16000}.withDefaults()
	if custom.Voice != "flux-other-en" || custom.SampleRate != 16000 {
		t.Fatalf("explicit config overwritten: %+v", custom)
	}

	// The constructor names the service and reports the configured rate.
	svc := NewFluxTTS(FluxTTSConfig{APIKey: "k"})
	if !strings.HasPrefix(svc.Name(), "DeepgramFluxTTS") {
		t.Fatalf("service name = %q", svc.Name())
	}
	if (&fluxSynth{cfg: cfg}).SampleRate() != defaultTTSSampleRate {
		t.Fatalf("SampleRate() mismatch")
	}
}

func TestFluxTTSQuery(t *testing.T) {
	mip := true
	cfg := FluxTTSConfig{
		Voice:      defaultFluxVoice,
		SampleRate: 24000,
		MipOptOut:  &mip,
		Tag:        []string{"prod", "agent"},
	}
	q := fluxTTSQuery(cfg)

	if got := q.Get("model"); got != defaultFluxVoice {
		t.Fatalf("model = %q (voice is sent as model)", got)
	}
	if got := q.Get("encoding"); got != fluxEncoding {
		t.Fatalf("encoding = %q", got)
	}
	if got := q.Get("sample_rate"); got != "24000" {
		t.Fatalf("sample_rate = %q", got)
	}
	if got := q.Get("mip_opt_out"); got != "true" {
		t.Fatalf("mip_opt_out = %q", got)
	}
	if got := q["tag"]; len(got) != 2 || got[0] != "prod" || got[1] != "agent" {
		t.Fatalf("tag = %v", got)
	}
}

func TestFluxTTSControl(t *testing.T) {
	// SpeechMetadata is the definitive per-turn completion signal.
	meta, _ := json.Marshal(map[string]any{"type": fluxTTSMsgSpeechMetadata, "speech_id": "x"})
	if done, err := fluxTTSControl(meta); err != nil || !done {
		t.Fatalf("SpeechMetadata: done=%v err=%v", done, err)
	}

	// Error surfaces as an error.
	errMsg, _ := json.Marshal(map[string]any{"type": "Error", "code": "bad", "description": "nope"})
	done, err := fluxTTSControl(errMsg)
	if done || err == nil || !errors.Is(err, errFluxServer) {
		t.Fatalf("Error: done=%v err=%v", done, err)
	}

	// Interim/informational control messages neither finish nor error.
	for _, typ := range []string{"Connected", "SpeechStarted", "Flushed", "SessionMetadata", "Warning"} {
		msg, _ := json.Marshal(map[string]any{"type": typ})
		if done, err := fluxTTSControl(msg); err != nil || done {
			t.Fatalf("%s: done=%v err=%v", typ, done, err)
		}
	}

	// Non-JSON frames are ignored.
	if done, err := fluxTTSControl([]byte("not json")); err != nil || done {
		t.Fatalf("garbage: done=%v err=%v", done, err)
	}
}

func TestFluxTTSSynthesize(t *testing.T) {
	var gotSpeak fluxSpeak
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// First message is the Speak with the text.
		_, speak, err := c.Read(ctx)
		if err != nil {
			return
		}
		_ = json.Unmarshal(speak, &gotSpeak)
		// Second message is the Flush that ends the turn.
		if _, _, err := c.Read(ctx); err != nil {
			return
		}

		_ = c.Write(ctx, websocket.MessageText, mustJSON(map[string]any{"type": "SpeechStarted"}))
		_ = c.Write(ctx, websocket.MessageBinary, []byte{1, 2, 3, 4})
		_ = c.Write(ctx, websocket.MessageBinary, []byte{5, 6})
		_ = c.Write(ctx, websocket.MessageText, mustJSON(map[string]any{
			"type": fluxTTSMsgSpeechMetadata, "speech_id": "s1", "audio_duration_ms": 10,
		}))
	}))
	defer srv.Close()

	syn := &fluxSynth{cfg: FluxTTSConfig{
		APIKey:     "k",
		SpeakURL:   wsURL(srv.URL),
		Voice:      defaultFluxVoice,
		SampleRate: 24000,
	}}

	var got []byte
	err := runPCM(syn, context.Background(), "hello world", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if gotSpeak.Type != "Speak" || gotSpeak.Text != "hello world" {
		t.Fatalf("server received speak = %+v", gotSpeak)
	}
	if string(got) != string([]byte{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("streamed PCM = %v, want [1 2 3 4 5 6]", got)
	}
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

// runPCM drives a synthesizer the way the base does, handing back the raw audio
// it yields.
func runPCM(s tts.Synthesizer, ctx context.Context, text string, emit func(pcm []byte) error) error {
	return s.RunTTS(ctx, text, "", func(f frames.Frame) error {
		if af, ok := f.(*frames.TTSAudioRawFrame); ok {
			return emit(af.Audio)
		}
		return nil
	})
}
