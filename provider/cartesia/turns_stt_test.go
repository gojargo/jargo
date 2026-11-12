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
	errs "github.com/gojargo/jargo/utils/errors"
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
// finalize the turn: it is reported as a prediction, and a resume as its
// withdrawal.
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
	// The eager end and the resume are reported as a prediction and its
	// withdrawal, which the service answers only with eager end of turn on.
	want := []struct {
		text      string
		final     bool
		endOfTurn bool
		speech    stt.SpeechState
		eager     bool
		withdrawn bool
	}{
		{"", false, false, stt.SpeechStarted, false, false},
		{"hello", false, false, stt.SpeechUnknown, false, false},
		{"hello there", false, false, stt.SpeechUnknown, true, false},
		{"", false, false, stt.SpeechUnknown, false, true},
		{"hello there friend", false, false, stt.SpeechUnknown, false, false},
		{"hello there friend", true, true, stt.SpeechStopped, false, false},
	}
	for i, w := range want {
		res, rerr := stream.Recv()
		if rerr != nil {
			t.Fatalf("Recv %d: %v", i, rerr)
		}
		if len(res) != 1 {
			t.Fatalf("Recv %d returned %d results, want 1", i, len(res))
		}
		r := res[0]
		if r.Text != w.text || r.Final != w.final || r.EndOfTurn != w.endOfTurn || r.Speech != w.speech ||
			r.EagerEndOfTurn != w.eager || r.EagerEndOfTurnWithdrawn != w.withdrawn {
			t.Errorf("result %d = %+v, want text %q final=%v endOfTurn=%v speech=%v eager=%v withdrawn=%v",
				i, r, w.text, w.final, w.endOfTurn, w.speech, w.eager, w.withdrawn)
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

// TestTurnsSTTServerError covers the errors the server sends. One that arrives
// before the session is confirmed rejects the connection's settings, so it ends
// the session, carrying the HTTP status the server gave it; one that arrives
// after it is reported and the session carries on.
func TestTurnsSTTServerError(t *testing.T) {
	endpoint, _ := turnsServer(t, []map[string]any{
		{
			"type": "error", "error_code": "model_not_found", "message": "Could not resolve model.",
			"status_code": 400,
		},
	})
	conn := newTurnsConnector(TurnsSTTConfig{APIKey: "k", URL: endpoint, Version: defaultVersion})
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_, err = stream.Recv()
	_ = stream.Close()
	if err == nil {
		t.Fatal("Recv() = nil error, want the rejected settings surfaced")
	}
	if !strings.Contains(err.Error(), "model_not_found") || !strings.Contains(err.Error(), "Could not resolve model.") {
		t.Errorf("error = %v, want it to carry the server's code and message", err)
	}
	if got := errs.ClassifyError(err); got != errs.InvalidRequest {
		t.Errorf("category = %v, want %v from the status the server gave", got, errs.InvalidRequest)
	}

	endpoint, _ = turnsServer(t, []map[string]any{
		{"type": "connected", "request_id": "r1"},
		{"type": "error", "message": "internal failure"},
		{"type": "turn.start", "transcript": ""},
	})
	conn = newTurnsConnector(TurnsSTTConfig{APIKey: "k", URL: endpoint, Version: defaultVersion})
	var reported []error
	stream, err = conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = stream.Close() }()
	ts, ok := stream.(*turnsStream)
	if !ok {
		t.Fatalf("stream is %T, want a turns stream", stream)
	}
	ts.report = func(_ context.Context, err error) { reported = append(reported, err) }

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv() = %v, want the session to carry on past the error", err)
	}
	if len(res) != 1 || res[0].Speech != stt.SpeechStarted {
		t.Errorf("results = %+v, want the turn start after the error", res)
	}
	if len(reported) != 1 || !strings.Contains(reported[0].Error(), "internal failure") {
		t.Errorf("reported %v, want the server error", reported)
	}
}

// TestTurnsSTTWaitsForTheAcknowledgement checks the connect returns once the
// socket is open, and the first read waits for the server to confirm the
// session, so a dial never holds up the audio path for the length of that wait.
func TestTurnsSTTWaitsForTheAcknowledgement(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		<-release
		for _, ev := range []map[string]any{
			{"type": "connected", "request_id": "r1"},
			{"type": "turn.start", "transcript": ""},
		} {
			b, _ := json.Marshal(ev)
			if c.Write(r.Context(), websocket.MessageText, b) != nil {
				return
			}
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
	stream, err := conn.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })

	received := make(chan error, 1)
	go func() {
		_, err := stream.Recv()
		received <- err
	}()
	select {
	case err := <-received:
		t.Fatalf("Recv returned before the session was acknowledged: %v", err)
	case <-time.After(100 * time.Millisecond):
	}

	close(release)
	select {
	case err := <-received:
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Recv did not return once the session was acknowledged")
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

	// Five results: the start, the update, the eager end and its withdrawal,
	// and the end.
	for range 5 {
		if _, err := stream.Recv(); err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	want := []string{"start", "update:hello", "eager_end:hello there", "resume", "end:hello there friend"}
	if !slices.Equal(got, want) {
		t.Errorf("events = %q, want %q", got, want)
	}
}
