package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
)

func wsURL(httpURL string) string {
	return strings.Replace(httpURL, "http", "ws", 1)
}

func TestFluxValidate(t *testing.T) {
	if err := (FluxConfig{APIKey: "k"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (FluxConfig{}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
}

func TestFluxQuery(t *testing.T) {
	cfg := FluxConfig{
		Model:             defaultFluxModel,
		EOTThreshold:      new(0.8),
		EagerEOTThreshold: new(0.5),
		Keyterm:           []string{"AI", "neural network"},
		Tag:               []string{"prod"},
		LanguageHints:     []language.Language{language.French},
	}
	q := fluxQuery(cfg, 16000)

	if got := q.Get("model"); got != defaultFluxModel {
		t.Fatalf("model = %q", got)
	}
	if got := q.Get("sample_rate"); got != "16000" {
		t.Fatalf("sample_rate = %q", got)
	}
	if got := q.Get("encoding"); got != fluxEncoding {
		t.Fatalf("encoding = %q", got)
	}
	if got := q.Get("eot_threshold"); got != "0.8" {
		t.Fatalf("eot_threshold = %q", got)
	}
	if got := q.Get("eager_eot_threshold"); got != "0.5" {
		t.Fatalf("eager_eot_threshold = %q", got)
	}
	if got := q["keyterm"]; len(got) != 2 || got[0] != "AI" || got[1] != "neural network" {
		t.Fatalf("keyterm = %v", got)
	}
	if got := q["tag"]; len(got) != 1 || got[0] != "prod" {
		t.Fatalf("tag = %v", got)
	}
	// Language hints are ignored on the English-only model.
	if got := q["language_hint"]; len(got) != 0 {
		t.Fatalf("language_hint should be empty on %s, got %v", defaultFluxModel, got)
	}
}

func TestFluxQueryLanguageHintsMultilingual(t *testing.T) {
	cfg := FluxConfig{
		Model:         fluxMultilingualModel,
		LanguageHints: []language.Language{language.EnglishGB, language.French},
	}
	q := fluxQuery(cfg, 24000)
	got := q["language_hint"]
	if len(got) != 2 || got[0] != "en" || got[1] != "fr" {
		t.Fatalf("language_hint = %v (want base codes en, fr)", got)
	}
}

func TestFluxResults(t *testing.T) {
	// Interim events carry a non-final result.
	for _, event := range []string{fluxEventUpdate, fluxEventEagerEndOfTurn, fluxEventStartOfTurn} {
		m := fluxMessage{Type: fluxMsgTurnInfo, Event: event, Transcript: "hello there"}
		res := fluxResults(m, defaultFluxModel, nil)
		if len(res) != 1 || res[0].Final || res[0].Text != "hello there" {
			t.Fatalf("%s result = %+v", event, res)
		}
		if res[0].Language != "en" {
			t.Fatalf("%s language = %q (want en fallback)", event, res[0].Language)
		}
	}

	// Empty-transcript interim events are dropped.
	empty := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventStartOfTurn}
	if res := fluxResults(empty, defaultFluxModel, nil); len(res) != 0 {
		t.Fatalf("empty StartOfTurn result = %+v", res)
	}

	// EndOfTurn is finalized and marks end of turn; detected language is honored.
	m := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "bonjour", Languages: []string{"fr"}}
	res := fluxResults(m, fluxMultilingualModel, nil)
	if len(res) != 1 || !res[0].Final || !res[0].EndOfTurn || res[0].Text != "bonjour" || res[0].Language != "fr" {
		t.Fatalf("EndOfTurn result = %+v", res)
	}

	// TurnResumed and non-TurnInfo messages produce nothing.
	resumed := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventTurnResumed}
	if res := fluxResults(resumed, defaultFluxModel, nil); len(res) != 0 {
		t.Fatalf("TurnResumed result = %+v", res)
	}
	if res := fluxResults(fluxMessage{Type: fluxMsgConnected}, defaultFluxModel, nil); len(res) != 0 {
		t.Fatalf("Connected result = %+v", res)
	}
}

func TestFluxConfidenceGate(t *testing.T) {
	words := []fluxWord{{Confidence: new(0.9)}, {Confidence: new(0.7)}} // avg 0.8

	// No threshold: always accepted.
	m := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "ok", Words: words}
	if res := fluxResults(m, defaultFluxModel, nil); len(res) != 1 {
		t.Fatalf("no-threshold result = %+v", res)
	}

	// Below threshold: dropped.
	if res := fluxResults(m, defaultFluxModel, new(0.85)); len(res) != 0 {
		t.Fatalf("below-threshold result should be dropped, got %+v", res)
	}

	// Above threshold: accepted.
	if res := fluxResults(m, defaultFluxModel, new(0.5)); len(res) != 1 {
		t.Fatalf("above-threshold result = %+v", res)
	}

	// Threshold set but no confidence data: dropped.
	noWords := fluxMessage{Type: fluxMsgTurnInfo, Event: fluxEventEndOfTurn, Transcript: "ok"}
	if res := fluxResults(noWords, defaultFluxModel, new(0.5)); len(res) != 0 {
		t.Fatalf("missing-confidence result should be dropped, got %+v", res)
	}
}

func TestFluxConnectAndRecv(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil { // consume the binary audio
			return
		}
		// A control message with no text should be skipped by Recv.
		connected, _ := json.Marshal(map[string]any{"type": fluxMsgConnected})
		_ = c.Write(ctx, websocket.MessageText, connected)
		interim, _ := json.Marshal(map[string]any{"type": fluxMsgTurnInfo, "event": fluxEventUpdate, "transcript": "hel"})
		_ = c.Write(ctx, websocket.MessageText, interim)
		final, _ := json.Marshal(map[string]any{
			"type":       fluxMsgTurnInfo,
			"event":      fluxEventEndOfTurn,
			"transcript": "hello",
			"languages":  []string{"en"},
		})
		_ = c.Write(ctx, websocket.MessageText, final)
	}))
	defer srv.Close()

	conn := &fluxConnector{cfg: FluxConfig{APIKey: "k", ListenURL: wsURL(srv.URL), Model: defaultFluxModel}}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := conn.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { cancel(); _ = stream.Close() }()

	if err := stream.Send([]byte{1, 2, 3, 4}); err != nil {
		t.Fatalf("Send: %v", err)
	}

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv interim: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hel" || res[0].Final {
		t.Fatalf("interim result = %+v", res)
	}

	res, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv final: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello" || !res[0].Final || !res[0].EndOfTurn || res[0].Language != "en" {
		t.Fatalf("final result = %+v", res)
	}
}
