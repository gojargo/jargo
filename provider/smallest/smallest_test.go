package smallest

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
		Voice: modelDefaultVoices[defaultModel], Language: defaultLanguage, SampleRate: defaultSampleRate,
	}}
	if syn.SampleRate() != defaultSampleRate {
		t.Fatalf("SampleRate = %d, want %d", syn.SampleRate(), defaultSampleRate)
	}

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

// The two Smallest models serve different voice catalogs, so the voice a caller
// gets when they name none has to follow the model they chose. One default for
// both would name a voice the other model does not serve.
func TestDefaultVoiceFollowsTheModel(t *testing.T) {
	for model, want := range map[string]string{
		modelLightning:  "sophia",
		modelLightningP: "meher",
	} {
		if got := voiceSent(t, Config{APIKey: "k", Model: model}); got != want {
			t.Errorf("model %q sent voice %q, want %q", model, got, want)
		}
	}
}

// The model the service defaults to gets that model's own voice.
func TestDefaultModelGetsItsOwnVoice(t *testing.T) {
	want := modelDefaultVoices[defaultModel]
	if got := voiceSent(t, Config{APIKey: "k"}); got != want {
		t.Errorf("voice = %q, want %q", got, want)
	}
}

// A voice the caller named is sent as given, whatever the model.
func TestNamedVoiceIsSentAsGiven(t *testing.T) {
	if got := voiceSent(t, Config{APIKey: "k", Model: modelLightning, Voice: "someone"}); got != "someone" {
		t.Errorf("voice = %q, want the one the caller named", got)
	}
}

// voiceSent synthesizes one sentence against a fake endpoint and reports the
// voice the request carried.
func voiceSent(t *testing.T, cfg Config) string {
	t.Helper()

	got := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		var payload struct {
			VoiceID string `json:"voice_id"`
		}
		_ = json.Unmarshal(data, &payload)
		select {
		case got <- payload.VoiceID:
		default:
		}
		done, _ := json.Marshal(map[string]any{"status": "complete"})
		_ = c.Write(ctx, websocket.MessageText, done)
	}))
	defer srv.Close()

	cfg.URL = wsURL(srv.URL)
	syn := &synthesizer{cfg: withDefaults(cfg)}
	_ = runPCM(syn, t.Context(), "hello", func([]byte) error { return nil })

	select {
	case v := <-got:
		return v
	case <-time.After(3 * time.Second):
		t.Fatal("the request never reached the endpoint")
		return ""
	}
}
