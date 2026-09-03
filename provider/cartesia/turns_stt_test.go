package cartesia

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

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

// TestTurnsSTTServerError surfaces a server-reported failure. One that arrives
// before the session is acknowledged fails the connect, since there is no
// session to read on; one that arrives after it ends the session.
func TestTurnsSTTServerError(t *testing.T) {
	endpoint, _ := turnsServer(t, []map[string]any{
		{"type": "error", "message": "unsupported sample rate"},
	})
	conn := newTurnsConnector(TurnsSTTConfig{APIKey: "k", URL: endpoint, Version: defaultVersion})
	if _, err := conn.Connect(context.Background(), 16000); err == nil {
		t.Fatal("Connect() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "unsupported sample rate") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}

	endpoint, _ = turnsServer(t, []map[string]any{
		{"type": "connected", "request_id": "r1"},
		{"type": "error", "message": "internal failure"},
	})
	conn = newTurnsConnector(TurnsSTTConfig{APIKey: "k", URL: endpoint, Version: defaultVersion})
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv() = nil error, want the server error surfaced")
	} else if !strings.Contains(err.Error(), "internal failure") {
		t.Errorf("error = %v, want it to carry the server message", err)
	}
}

// TestTurnsSTTWaitsForTheAcknowledgement checks the connect does not return
// until the server has opened the session, so no audio is sent to one that does
// not exist yet.
func TestTurnsSTTWaitsForTheAcknowledgement(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		<-release
		b, _ := json.Marshal(map[string]any{"type": "connected", "request_id": "r1"})
		if c.Write(r.Context(), websocket.MessageText, b) != nil {
			return
		}
		for {
			if _, _, err := c.Read(r.Context()); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	conn := newTurnsConnector(TurnsSTTConfig{
		APIKey: "k", URL: "ws" + strings.TrimPrefix(srv.URL, "http"), Version: defaultVersion,
	})
	connected := make(chan error, 1)
	go func() {
		stream, err := conn.Connect(context.Background(), 16000)
		if stream != nil {
			defer func() { _ = stream.Close() }()
		}
		connected <- err
	}()

	select {
	case err := <-connected:
		t.Fatalf("Connect returned before the session was acknowledged: %v", err)
	case <-time.After(100 * time.Millisecond):
	}

	close(release)
	select {
	case err := <-connected:
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Connect did not return once the session was acknowledged")
	}
}

// TestTurnsSTTDrainAsksTheServerToFinish checks the session is told to finish
// before the socket goes, so a turn the server was still holding is flushed
// rather than lost with the connection.
func TestTurnsSTTDrainAsksTheServerToFinish(t *testing.T) {
	sent := make(chan string, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		b, _ := json.Marshal(map[string]any{"type": "connected"})
		if c.Write(r.Context(), websocket.MessageText, b) != nil {
			return
		}
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if typ == websocket.MessageText {
				var m map[string]any
				if json.Unmarshal(data, &m) == nil {
					s, _ := m["type"].(string)
					select {
					case sent <- s:
					default:
					}
				}
			}
		}
	}))
	t.Cleanup(srv.Close)

	conn := newTurnsConnector(TurnsSTTConfig{
		APIKey: "k", URL: "ws" + strings.TrimPrefix(srv.URL, "http"), Version: defaultVersion,
	})
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	d, ok := stream.(stt.Drainer)
	if !ok {
		t.Fatal("the stream does not implement stt.Drainer, so the last turn is lost at hangup")
	}
	window, err := d.Drain()
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if window != turnsDrainWindow {
		t.Errorf("drain window = %s, want %s", window, turnsDrainWindow)
	}

	select {
	case msg := <-sent:
		if msg != turnsClose {
			t.Errorf("message = %q, want %q", msg, turnsClose)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the server was never told the session was finishing")
	}
}

// TestTurnsSTTWatchdog checks the service asks for a watchdog. The server ends a
// turn from the audio it is fed, so a turn it has opened hangs open for as long
// as the audio is stopped.
func TestTurnsSTTWatchdog(t *testing.T) {
	c := newTurnsConnector(TurnsSTTConfig{APIKey: "k"})
	w, ok := any(c).(stt.TurnWatchdoger)
	if !ok {
		t.Fatal("the connector does not implement stt.TurnWatchdoger, so a stalled turn hangs")
	}
	if got := w.TurnWatchdog().MinTimeout; got != 0 {
		t.Errorf("MinTimeout = %s, want 0 so the service default applies", got)
	}

	chosen := 250 * time.Millisecond
	c = newTurnsConnector(TurnsSTTConfig{APIKey: "k", WatchdogMinTimeout: chosen})
	if got := c.TurnWatchdog().MinTimeout; got != chosen {
		t.Errorf("MinTimeout = %s, want the configured %s", got, chosen)
	}
}

// TestTurnsSTTReportsTheTurnEvents checks every event the server sends reaches
// the listeners, including the eager end and the resume that retracts it, which
// carry no transcript into the pipeline and would otherwise be unobservable.
func TestTurnsSTTReportsTheTurnEvents(t *testing.T) {
	endpoint, _ := turnsServer(t, []map[string]any{
		{"type": "connected"},
		{"type": "turn.start", "transcript": ""},
		{"type": "turn.update", "transcript": "hello"},
		{"type": "turn.eager_end", "transcript": "hello there"},
		{"type": "turn.resume"},
		{"type": "turn.end", "transcript": "hello there friend"},
	})

	var got []string
	cfg := TurnsSTTConfig{APIKey: "k", URL: endpoint, Version: defaultVersion}
	cfg.OnTurnStart = func(string) { got = append(got, "start") }
	cfg.OnTurnUpdate = func(tx string) { got = append(got, "update:"+tx) }
	cfg.OnTurnEagerEnd = func(tx string) { got = append(got, "eager_end:"+tx) }
	cfg.OnTurnResume = func() { got = append(got, "resume") }
	cfg.OnTurnEnd = func(tx string) { got = append(got, "end:"+tx) }

	stream, err := newTurnsConnector(cfg).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()

	// Three results carry text or a boundary; the eager end and the resume reach
	// the listeners without producing one.
	for range 3 {
		if _, err := stream.Recv(); err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	want := []string{"start", "update:hello", "eager_end:hello there", "resume", "end:hello there friend"}
	if !slices.Equal(got, want) {
		t.Errorf("events = %q, want %q", got, want)
	}
}
