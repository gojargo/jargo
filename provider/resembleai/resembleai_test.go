package resembleai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

func TestValidate(t *testing.T) {
	if err := (Config{APIKey: "k", Voice: "v"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{Voice: "v"}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	if err := (Config{APIKey: "k"}).Validate(); err == nil {
		t.Fatal("config without Voice should be rejected")
	}
}

func TestSynthesizeStreamsPCM(t *testing.T) {
	want := []byte{0x01, 0x02, 0x03, 0x04}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil { // consume the request
			return
		}
		audio, _ := json.Marshal(map[string]any{"type": "audio", "audio_content": base64.StdEncoding.EncodeToString(want)})
		_ = c.Write(ctx, websocket.MessageText, audio)
		end, _ := json.Marshal(map[string]any{"type": "audio_end"})
		_ = c.Write(ctx, websocket.MessageText, end)
	}))
	defer srv.Close()

	syn := &synthesizer{cfg: Config{
		APIKey: "k", Voice: "v", URL: wsURL(srv.URL),
		SampleRate: defaultSampleRate, Precision: defaultPrecision,
	}}

	var got []byte
	err := runPCM(syn, context.Background(), "hello", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("streamed PCM = %v, want %v", got, want)
	}
}

func wsURL(httpURL string) string { return "ws" + strings.TrimPrefix(httpURL, "http") }

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

// A sentence's synthesis ends at the server's audio_end, not after a wait for a
// close acknowledgment Resemble AI never sends: waiting would hold the next
// sentence back by the whole close timeout.
func TestUnacknowledgedCloseDoesNotDelay(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil {
			return
		}
		audio, _ := json.Marshal(map[string]any{
			"type": "audio", "audio_content": base64.StdEncoding.EncodeToString(make([]byte, 64)),
		})
		_ = c.Write(ctx, websocket.MessageText, audio)
		end, _ := json.Marshal(map[string]any{"type": "audio_end"})
		_ = c.Write(ctx, websocket.MessageText, end)
		// Deaf from here: the client's close frame is never answered.
		<-ctx.Done()
	}))
	defer srv.Close()

	syn := &synthesizer{cfg: Config{
		APIKey: "k", Voice: "v", URL: wsURL(srv.URL),
		SampleRate: defaultSampleRate, Precision: defaultPrecision,
	}}
	start := time.Now()
	if err := runPCM(syn, context.Background(), "hello", func([]byte) error { return nil }); err != nil {
		t.Fatalf("RunTTS: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Errorf("RunTTS took %v, want it to return without waiting on the close", elapsed)
	}
}
