package smallest

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
	if err := (Config{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
}

func TestSynthesizeStreamsPCM(t *testing.T) {
	want := []byte{0x10, 0x20, 0x30, 0x40}
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
		b64 := base64.StdEncoding.EncodeToString(want)
		chunk, _ := json.Marshal(map[string]any{"status": "chunk", "data": map[string]any{"audio": b64}})
		_ = c.Write(ctx, websocket.MessageText, chunk)
		done, _ := json.Marshal(map[string]any{"status": "complete"})
		_ = c.Write(ctx, websocket.MessageText, done)
	}))
	defer srv.Close()

	syn := &synthesizer{cfg: Config{
		APIKey: "k", URL: wsURL(srv.URL), Model: defaultModel,
		Voice: defaultVoice, Language: defaultLanguage, SampleRate: defaultSampleRate,
	}}
	if syn.SampleRate() != defaultSampleRate {
		t.Fatalf("SampleRate = %d, want %d", syn.SampleRate(), defaultSampleRate)
	}

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
