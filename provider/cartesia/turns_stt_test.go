package cartesia

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/stt"
)

// TestTurnsSTTConfigValidate pins which fields the provider requires.
func TestTurnsSTTConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: TurnsSTTConfig{}, Valid: false},
		{Name: "API key only", Cfg: TurnsSTTConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewTurnsSTT checks the constructor returns a service under the label that
// identifies it in logs, metrics and traces.
func TestNewTurnsSTT(t *testing.T) {
	providertest.Service(t, "CartesiaTurnsSTT", NewTurnsSTT(TurnsSTTConfig{APIKey: "k"}))
}

// TestTurnsSTTMetadata checks the service tells downstream it detects turns
// itself, which is the whole point of this endpoint over the plain STT one.
func TestTurnsSTTMetadata(t *testing.T) {
	cfg := TurnsSTTConfig{APIKey: "k", Model: defaultTurnsModel}
	c := newTurnsConnector(cfg)
	meta := c.Metadata()
	got, ok := meta.UserTurnStrategies.(turns.UserTurnStrategies)
	if !ok {
		t.Fatalf("UserTurnStrategies = %T, want external turn strategies", meta.UserTurnStrategies)
	}
	if _, external := got.ExternalInterruptions(); !external {
		t.Error("the recommended strategies are not the external ones")
	}
	if meta.Model != defaultTurnsModel {
		t.Errorf("Model = %q, want %q", meta.Model, defaultTurnsModel)
	}
}

// turnsServer starts a fake turn-detection endpoint replaying scripted events.
func turnsServer(t *testing.T, events []map[string]any) (endpoint string, query func() url.Values) {
	t.Helper()
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
		for _, ev := range events {
			b, _ := json.Marshal(ev)
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

	return "ws" + strings.TrimPrefix(srv.URL, "http"), func() url.Values {
		select {
		case q := <-queryCh:
			return q
		default:
			return nil
		}
	}
}

// TestTurnsSTTRecv checks the turn lifecycle: an update is the running
// transcript, the turn's end finalizes it, and the boundary-only events emit
// nothing. An eager end is a prediction the server may retract, so it must not
// finalize the turn.
func TestTurnsSTTRecv(t *testing.T) {
	endpoint, query := turnsServer(t, []map[string]any{
		{"type": "connected", "request_id": "r1"},
		{"type": "turn.start", "transcript": ""},
		{"type": "turn.update", "transcript": "hello"},
		{"type": "turn.eager_end", "transcript": "hello there"},
		{"type": "turn.resume"},
		{"type": "turn.update", "transcript": "hello there friend"},
		{"type": "turn.end", "transcript": "hello there friend"},
	})

	conn := newTurnsConnector(TurnsSTTConfig{
		APIKey:  "k",
		URL:     endpoint,
		Version: defaultVersion,
		Model:   defaultTurnsModel,
	})
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	// The turn boundaries the server detects are reported alongside the text, as
	// proposals for the pipeline's turn strategies to resolve.
	want := []struct {
		text      string
		final     bool
		endOfTurn bool
		speech    stt.SpeechState
	}{
		{"", false, false, stt.SpeechStarted},
		{"hello", false, false, stt.SpeechUnknown},
		{"hello there friend", false, false, stt.SpeechUnknown},
		{"hello there friend", true, true, stt.SpeechStopped},
	}
	for i, w := range want {
		res, rerr := stream.Recv()
		if rerr != nil {
			t.Fatalf("Recv %d: %v", i, rerr)
		}
		if len(res) != 1 {
			t.Fatalf("Recv %d returned %d results, want 1", i, len(res))
		}
		if res[0].Text != w.text || res[0].Final != w.final || res[0].EndOfTurn != w.endOfTurn ||
			res[0].Speech != w.speech {
			t.Errorf("result %d = %+v, want text %q final=%v endOfTurn=%v speech=%v",
				i, res[0], w.text, w.final, w.endOfTurn, w.speech)
		}
	}

	q := query()
	if q.Get("model") != defaultTurnsModel {
		t.Errorf("model = %q, want %q", q.Get("model"), defaultTurnsModel)
	}
	if q.Get("encoding") != defaultSTTEncoding {
		t.Errorf("encoding = %q, want %q", q.Get("encoding"), defaultSTTEncoding)
	}
	if q.Get("sample_rate") != "16000" {
		t.Errorf("sample_rate = %q, want the transport rate", q.Get("sample_rate"))
	}
}

// TestTurnsSTTServerError surfaces a server-reported failure.
func TestTurnsSTTServerError(t *testing.T) {
	endpoint, _ := turnsServer(t, []map[string]any{
		{"type": "error", "message": "unsupported sample rate"},
	})
	conn := newTurnsConnector(TurnsSTTConfig{APIKey: "k", URL: endpoint, Version: defaultVersion})
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "unsupported sample rate") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}
}
