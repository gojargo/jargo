package resembleai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
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
	err := syn.Synthesize(context.Background(), "hello", func(pcm []byte) error {
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
