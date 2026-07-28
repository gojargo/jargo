package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
)

// TestSTTConfigValidate pins which fields the provider requires.
func TestSTTConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: STTConfig{}, Valid: false},
		{Name: "API key only", Cfg: STTConfig{APIKey: "k"}, Valid: true},
		{Name: "supported noise reduction", Cfg: STTConfig{APIKey: "k", NoiseReduction: "near_field"}, Valid: true},
		{Name: "unsupported noise reduction", Cfg: STTConfig{APIKey: "k", NoiseReduction: "loud"}, Valid: false},
	})
}

// TestNewSTT checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewSTT(t *testing.T) {
	providertest.Service(t, "OpenAIRealtimeSTT", NewSTT(STTConfig{APIKey: "k"}))
}

// TestSTTMetadata checks the service reports that it detects turns itself.
func TestSTTMetadata(t *testing.T) {
	c := &sttConnector{cfg: STTConfig{APIKey: "k", Model: defaultSTTModel}}
	if got := c.Metadata().RecommendedUserTurns; got != frames.UserTurnExternal {
		t.Errorf("RecommendedUserTurns = %v, want UserTurnExternal", got)
	}
}

// TestSTTSessionUpdate checks the transcription session configuration: the
// session type, the audio format, and the server-side turn detection that
// delimits each utterance.
func TestSTTSessionUpdate(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{APIKey: "k", Model: defaultSTTModel}}
		msg := c.sessionUpdate()
		if msg[keyType] != msgSessionUpdate {
			t.Errorf("type = %v, want %q", msg[keyType], msgSessionUpdate)
		}
		session, _ := msg["session"].(map[string]any)
		if session[keyType] != "transcription" {
			t.Errorf("session type = %v, want transcription", session[keyType])
		}
		audio, _ := session[keyAudio].(map[string]any)
		input, _ := audio["input"].(map[string]any)

		format, _ := input["format"].(map[string]any)
		if format[keyType] != "audio/pcm" || format["rate"] != sttSampleRate {
			t.Errorf("format = %v, want PCM at %d", format, sttSampleRate)
		}
		transcription, _ := input["transcription"].(map[string]any)
		if transcription["model"] != defaultSTTModel {
			t.Errorf("model = %v, want %q", transcription["model"], defaultSTTModel)
		}
		if _, ok := transcription["language"]; ok {
			t.Error("language is present when unset, want it omitted")
		}
		if _, ok := transcription["prompt"]; ok {
			t.Error("prompt is present when unset, want it omitted")
		}
		turn, _ := input["turn_detection"].(map[string]any)
		if turn[keyType] != "server_vad" {
			t.Errorf("turn_detection = %v, want server_vad", turn)
		}
		if _, ok := input["noise_reduction"]; ok {
			t.Error("noise_reduction is present when unset, want it omitted")
		}
	})

	t.Run("optional settings", func(t *testing.T) {
		c := &sttConnector{cfg: STTConfig{
			APIKey:         "k",
			Model:          "gpt-4o-transcribe",
			Language:       language.FrenchCA,
			Prompt:         "product names",
			NoiseReduction: "far_field",
			SilenceMS:      700,
		}}
		session, _ := c.sessionUpdate()["session"].(map[string]any)
		audio, _ := session[keyAudio].(map[string]any)
		input, _ := audio["input"].(map[string]any)

		transcription, _ := input["transcription"].(map[string]any)
		if transcription["language"] != "fr" {
			t.Errorf("language = %v, want the base code fr", transcription["language"])
		}
		if transcription["prompt"] != "product names" {
			t.Errorf("prompt = %v, want the configured prompt", transcription["prompt"])
		}
		turn, _ := input["turn_detection"].(map[string]any)
		if turn["silence_duration_ms"] != 700 {
			t.Errorf("silence_duration_ms = %v, want 700", turn["silence_duration_ms"])
		}
		nr, _ := input["noise_reduction"].(map[string]any)
		if nr[keyType] != "far_field" {
			t.Errorf("noise_reduction = %v, want far_field", nr)
		}
	})
}

// sttServer starts a fake transcription endpoint replaying scripted events and
// recording what the client sent.
func sttServer(
	t *testing.T, events []map[string]any,
) (endpoint string, sent <-chan map[string]any, query func() url.Values) {
	t.Helper()
	messages := make(chan map[string]any, 8)
	queryCh := make(chan url.Values, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case queryCh <- r.URL.Query():
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// The client configures the session before any audio flows.
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var setup map[string]any
		if json.Unmarshal(data, &setup) == nil {
			messages <- setup
		}
		for _, ev := range events {
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) == nil {
				messages <- msg
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), messages, func() url.Values {
		select {
		case q := <-queryCh:
			return q
		default:
			return nil
		}
	}
}

// TestSTTRecv checks the interim and final mapping. The API streams interims as
// fragments, so they are accumulated into the running transcript rather than
// emitted piecemeal.
func TestSTTRecv(t *testing.T) {
	endpoint, sent, query := sttServer(t, []map[string]any{
		{"type": sttEventDelta, "delta": "hello"},
		{"type": sttEventDelta, "delta": " there"},
		{"type": sttEventCompleted, "transcript": "Hello there."},
	})

	conn := &sttConnector{cfg: STTConfig{
		APIKey:   "test-key",
		BaseURL:  endpoint,
		Model:    defaultSTTModel,
		Language: language.English,
	}}
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if q := query(); q.Get("intent") != "transcription" {
		t.Errorf("intent = %q, want transcription so the model only transcribes", q.Get("intent"))
	}
	if setup := <-sent; setup[keyType] != msgSessionUpdate {
		t.Errorf("first message = %v, want the session configuration", setup)
	}

	want := []struct {
		text  string
		final bool
	}{
		{"hello", false},
		{"hello there", false}, // accumulated, not the raw fragment
		{"Hello there.", true},
	}
	for i, w := range want {
		res, rerr := stream.Recv()
		if rerr != nil {
			t.Fatalf("Recv %d: %v", i, rerr)
		}
		if len(res) != 1 || res[0].Text != w.text || res[0].Final != w.final {
			t.Fatalf("result %d = %+v, want text %q final=%v", i, res, w.text, w.final)
		}
		if res[0].Language != "en" {
			t.Errorf("result %d language = %q, want en", i, res[0].Language)
		}
	}

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	audio := <-sent
	if audio[keyType] != "input_audio_buffer.append" {
		t.Errorf("audio message type = %v, want the buffer append", audio[keyType])
	}
	if got, _ := audio[keyAudio].(string); got != base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4}) {
		t.Errorf("audio = %q, want the base64 PCM", got)
	}
}

// TestSTTPartialResetsOnFinal checks the accumulator is cleared between
// utterances, so the next one does not inherit the previous one's text.
func TestSTTPartialResetsOnFinal(t *testing.T) {
	endpoint, _, _ := sttServer(t, []map[string]any{
		{"type": sttEventDelta, "delta": "first"},
		{"type": sttEventCompleted, "transcript": "First."},
		{"type": sttEventDelta, "delta": "second"},
	})

	conn := &sttConnector{cfg: STTConfig{APIKey: "k", BaseURL: endpoint, Model: defaultSTTModel}}
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	for range 2 {
		if _, err := stream.Recv(); err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "second" {
		t.Errorf("result = %+v, want only the new utterance's text", res)
	}
}

// TestSTTServerError surfaces a session failure.
func TestSTTServerError(t *testing.T) {
	endpoint, _, _ := sttServer(t, []map[string]any{
		{"type": sttEventError, "error": map[string]any{"message": "invalid model"}},
	})
	conn := &sttConnector{cfg: STTConfig{APIKey: "k", BaseURL: endpoint, Model: defaultSTTModel}}
	stream, err := conn.Connect(context.Background(), sttSampleRate)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "invalid model") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}
}
