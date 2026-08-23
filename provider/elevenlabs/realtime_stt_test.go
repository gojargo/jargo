package elevenlabs

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
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor/turns"
)

// TestRealtimeSTTConfigValidate pins the fields the provider requires and the
// ranges ElevenLabs accepts.
func TestRealtimeSTTConfigValidate(t *testing.T) {
	inRange, tooLow := 0.3, 0.2
	conf, tooHigh := 0.5, 1.5
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: RealtimeSTTConfig{}, Valid: false},
		{Name: "API key only", Cfg: RealtimeSTTConfig{APIKey: "k"}, Valid: true},
		{
			Name:  "silence threshold in range",
			Cfg:   RealtimeSTTConfig{APIKey: "k", SilenceThresholdSecs: &inRange},
			Valid: true,
		},
		{Name: "silence threshold too low", Cfg: RealtimeSTTConfig{APIKey: "k", SilenceThresholdSecs: &tooLow}, Valid: false},
		{Name: "vad threshold in range", Cfg: RealtimeSTTConfig{APIKey: "k", VADThreshold: &conf}, Valid: true},
		{Name: "vad threshold too high", Cfg: RealtimeSTTConfig{APIKey: "k", VADThreshold: &tooHigh}, Valid: false},
	})
}

// TestNewRealtimeSTT checks the constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewRealtimeSTT(t *testing.T) {
	providertest.Service(t, "ElevenLabsRealtimeSTT", NewRealtimeSTT(RealtimeSTTConfig{APIKey: "k"}))
}

// TestRealtimeSTTMetadata checks the service reports that it detects utterance
// boundaries itself, unlike the batch service next to it.
func TestRealtimeSTTMetadata(t *testing.T) {
	c := &realtimeSTTConnector{cfg: RealtimeSTTConfig{APIKey: "k", Model: defaultRealtimeSTTModel}}
	got, ok := c.Metadata().UserTurnStrategies.(turns.UserTurnStrategies)
	if !ok {
		t.Fatalf("UserTurnStrategies = %T, want external turn strategies", c.Metadata().UserTurnStrategies)
	}
	if _, external := got.ExternalInterruptions(); !external {
		t.Error("the recommended strategies are not the external ones")
	}
}

// TestRealtimeSTTEndpoint checks the session configuration, which travels
// entirely as query parameters.
func TestRealtimeSTTEndpoint(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &realtimeSTTConnector{cfg: RealtimeSTTConfig{
			APIKey: "k",
			Host:   defaultRealtimeSTTHost,
			Model:  defaultRealtimeSTTModel,
		}}
		q := parseRealtimeQuery(t, c.endpoint(16000))
		if q.Get("model_id") != defaultRealtimeSTTModel {
			t.Errorf("model_id = %q, want %q", q.Get("model_id"), defaultRealtimeSTTModel)
		}
		if q.Get("audio_format") != realtimeSTTFormat {
			t.Errorf("audio_format = %q, want %q", q.Get("audio_format"), realtimeSTTFormat)
		}
		if q.Get("sample_rate") != "16000" {
			t.Errorf("sample_rate = %q, want the transport rate", q.Get("sample_rate"))
		}
		if q.Get("commit_strategy") != "vad" {
			t.Errorf("commit_strategy = %q, want vad so the service delimits utterances", q.Get("commit_strategy"))
		}
		for _, key := range []string{
			"language_code", "keyterms", "vad_silence_threshold_secs",
			"vad_threshold", "min_speech_duration_ms", "min_silence_duration_ms",
		} {
			if q.Has(key) {
				t.Errorf("%s = %q, want it omitted when unset", key, q.Get(key))
			}
		}
	})

	t.Run("optional settings", func(t *testing.T) {
		silence, confidence := 0.8, 0.6
		c := &realtimeSTTConnector{cfg: RealtimeSTTConfig{
			APIKey:               "k",
			Host:                 defaultRealtimeSTTHost,
			Model:                defaultRealtimeSTTModel,
			Language:             language.French,
			Keyterms:             []string{"jargo", "Pion"},
			SilenceThresholdSecs: &silence,
			VADThreshold:         &confidence,
			MinSpeechMS:          120,
			MinSilenceMS:         240,
		}}
		q := parseRealtimeQuery(t, c.endpoint(16000))
		if q.Get("language_code") == "" {
			t.Error("language_code is absent, want the mapped language")
		}
		if terms := q["keyterms"]; len(terms) != 2 {
			t.Errorf("keyterms = %v, want both terms", terms)
		}
		if q.Get("vad_silence_threshold_secs") != "0.8" {
			t.Errorf("vad_silence_threshold_secs = %q, want 0.8", q.Get("vad_silence_threshold_secs"))
		}
		if q.Get("vad_threshold") != "0.6" {
			t.Errorf("vad_threshold = %q, want 0.6", q.Get("vad_threshold"))
		}
		if q.Get("min_speech_duration_ms") != "120" || q.Get("min_silence_duration_ms") != "240" {
			t.Errorf("speech/silence durations = %q/%q, want 120/240",
				q.Get("min_speech_duration_ms"), q.Get("min_silence_duration_ms"))
		}
	})
}

// realtimeSTTServer starts a fake streaming endpoint replaying scripted
// messages and recording what the client sent.
func realtimeSTTServer(
	t *testing.T, events []map[string]any,
) (endpoint string, sent <-chan map[string]any, auth func() string) {
	t.Helper()
	messages := make(chan map[string]any, 8)
	authCh := make(chan string, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case authCh <- r.Header.Get("xi-api-key"):
		default:
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
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

	return "ws" + strings.TrimPrefix(srv.URL, "http"), messages, func() string {
		select {
		case v := <-authCh:
			return v
		default:
			return ""
		}
	}
}

// TestRealtimeSTTRecv checks the partial and committed transcripts map onto
// interim and finalized results, and that audio is sent uncommitted since the
// service decides where an utterance ends.
func TestRealtimeSTTRecv(t *testing.T) {
	endpoint, sent, auth := realtimeSTTServer(t, []map[string]any{
		{"message_type": rtEventStarted},
		{"message_type": rtEventPartial, "text": "hello"},
		{"message_type": rtEventCommitted, "text": "Hello there."},
	})

	conn := &realtimeSTTConnector{cfg: RealtimeSTTConfig{
		APIKey: "test-key",
		Host:   endpoint,
		Model:  defaultRealtimeSTTModel,
	}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if got := auth(); got != "test-key" {
		t.Errorf("xi-api-key = %q, want the key", got)
	}

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv partial: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello" || res[0].Final {
		t.Fatalf("partial = %+v, want an unfinalized \"hello\"", res)
	}

	res, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv committed: %v", err)
	}
	if len(res) != 1 || res[0].Text != "Hello there." || !res[0].Final || !res[0].EndOfTurn {
		t.Fatalf("committed = %+v, want a finalized transcript ending the turn", res)
	}

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	chunk := <-sent
	if chunk["message_type"] != "input_audio_chunk" {
		t.Errorf("message_type = %v, want input_audio_chunk", chunk["message_type"])
	}
	if got, _ := chunk["audio_chunk"].(string); got != base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4}) {
		t.Errorf("audio_chunk = %q, want the base64 PCM", got)
	}
	if chunk["commit"] != false {
		t.Errorf("commit = %v, want false: the service commits on its own silence detection", chunk["commit"])
	}
}

// TestRealtimeSTTServerError surfaces a session failure.
func TestRealtimeSTTServerError(t *testing.T) {
	endpoint, _, _ := realtimeSTTServer(t, []map[string]any{
		{"message_type": rtEventError, "message": "unsupported format"},
	})
	conn := &realtimeSTTConnector{cfg: RealtimeSTTConfig{APIKey: "k", Host: endpoint, Model: defaultRealtimeSTTModel}}
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "unsupported format") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}
}

// parseRealtimeQuery pulls the query parameters off a built endpoint URL.
func parseRealtimeQuery(t *testing.T, endpoint string) url.Values {
	t.Helper()
	u, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("endpoint %q is not a URL: %v", endpoint, err)
	}
	return u.Query()
}
