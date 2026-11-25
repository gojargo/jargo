package cartesia

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// sevenLanguages is one more distinct base code than Cartesia accepts as hints.
//
//nolint:gochecknoglobals // a fixed list
var sevenLanguages = []language.Language{
	language.English, language.Spanish, language.Portuguese, language.French,
	language.German, language.Hindi, language.Japanese,
}

// Ported from upstream (test_cartesia_turns_websocket_url_includes_language_hints_as_base_codes).
func TestTurnsURLIncludesLanguageHintsAsBaseCodes(t *testing.T) {
	q := turnsQuery(t, TurnsSTTConfig{LanguageHints: []language.Language{
		language.Hindi, language.EnglishUS, language.English, language.PortugueseBR,
		language.Spanish, language.French, language.German,
	}})
	if got, want := q["language_hint"], []string{"hi", "en", "pt", "es", "fr", "de"}; !slices.Equal(got, want) {
		t.Errorf("language_hint = %v, want %v", got, want)
	}
}

// Ported from upstream (test_cartesia_turns_rejects_more_than_six_distinct_language_hints).
func TestTurnsRejectsMoreThanSixDistinctLanguageHints(t *testing.T) {
	err := TurnsSTTConfig{APIKey: "k", LanguageHints: sevenLanguages}.Validate()
	if !errors.Is(err, errTooManyLanguageHints) ||
		!strings.Contains(err.Error(), "language_hints accepts at most 6 distinct base codes") {
		t.Errorf("Validate() = %v, want too many language hints", err)
	}
	if err := (TurnsSTTConfig{APIKey: "k", LanguageHints: sevenLanguages[:6]}).Validate(); err != nil {
		t.Errorf("Validate() = %v, want six hints accepted", err)
	}
}

// Ported from upstream (test_cartesia_turns_websocket_url_omits_language_hints_when_not_set).
func TestTurnsURLOmitsLanguageHintsWhenNotSet(t *testing.T) {
	for _, hints := range [][]language.Language{nil, {}} {
		if q := turnsQuery(t, TurnsSTTConfig{LanguageHints: hints}); q["language_hint"] != nil {
			t.Errorf("hints %v: language_hint = %v, want none", hints, q["language_hint"])
		}
	}
}

// hintsServer is a fake turn-detection endpoint that confirms each session,
// records the query of each handshake and the text messages it is sent, and
// closes a session it is asked to finish.
type hintsServer struct {
	mu       sync.Mutex
	queries  []url.Values
	messages []map[string]any
}

func (h *hintsServer) start(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		h.queries = append(h.queries, r.URL.Query())
		h.mu.Unlock()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		if c.Write(ctx, websocket.MessageText, []byte(`{"type":"connected","request_id":"r1"}`)) != nil {
			return
		}
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var m map[string]any
			if typ != websocket.MessageText || json.Unmarshal(data, &m) != nil {
				continue
			}
			h.mu.Lock()
			h.messages = append(h.messages, m)
			h.mu.Unlock()
			// Asked to finish, the server has no turn to flush and closes the
			// session, as Cartesia does.
			if m["type"] == turnsClose {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// configs is the config messages the server was sent.
func (h *hintsServer) configs() []map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []map[string]any
	for _, m := range h.messages {
		if m["type"] == turnsConfig {
			out = append(out, m)
		}
	}
	return out
}

// handshakes is the query of each handshake the server took.
func (h *hintsServer) handshakes() []url.Values {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.queries)
}

// hintsRun is a Turns service running in a pipeline against a hintsServer.
type hintsRun struct {
	server *hintsServer
	task   *pipeline.Worker

	mu     sync.Mutex
	errors []*frames.ErrorFrame
}

// runHints starts a Turns service built from cfg and waits for its first
// session.
func runHints(t *testing.T, cfg TurnsSTTConfig) *hintsRun {
	t.Helper()
	r := &hintsRun{server: &hintsServer{}}
	cfg.APIKey, cfg.URL, cfg.SampleRate = "k", r.server.start(t), 16000
	svc := NewTurnsSTT(cfg)
	event.On(svc.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		r.mu.Lock()
		r.errors = append(r.errors, ef)
		r.mu.Unlock()
	})
	r.task = pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- r.task.Run(context.Background()) }()
	t.Cleanup(func() {
		r.task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("the pipeline did not shut down")
		}
	})
	waitUntil(t, "the first session never opened", func() bool { return len(r.server.handshakes()) == 1 })
	return r
}

// update sends the service a settings update.
func (r *hintsRun) update(delta *TurnsSTTSettings) {
	r.task.QueueFrame(frames.NewSTTUpdateSettingsFrame(delta))
}

// reported is the errors the service reported.
func (r *hintsRun) reported() []*frames.ErrorFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.errors)
}

// waitUntil waits for cond, failing with msg when it does not come.
func waitUntil(t *testing.T, msg string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal(msg)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// Ported from upstream (test_cartesia_turns_update_language_hints_sends_config).
func TestTurnsUpdateLanguageHintsSendsConfig(t *testing.T) {
	tests := []struct {
		name  string
		hints settings.Opt[[]language.Language]
		sent  []any
	}{
		{"new hints", settings.Set([]language.Language{language.Spanish, language.EnglishGB}), []any{"es", "en"}},
		{"no hints", settings.Set([]language.Language{}), []any{}},
		{"hints cleared", settings.Cleared[[]language.Language](), []any{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := runHints(t, TurnsSTTConfig{LanguageHints: []language.Language{language.Hindi}})

			r.update(&TurnsSTTSettings{LanguageHints: tt.hints})
			waitUntil(t, "no config message was sent", func() bool { return len(r.server.configs()) > 0 })

			got := r.server.configs()
			if len(got) != 1 {
				t.Fatalf("%d config messages sent, want 1", len(got))
			}
			if hints, _ := got[0]["language_hints"].([]any); !slices.Equal(hints, tt.sent) {
				t.Errorf("config = %v, want language_hints %v", got[0], tt.sent)
			}
			// Settle before counting handshakes: a reopen would follow the update.
			time.Sleep(100 * time.Millisecond)
			if n := len(r.server.handshakes()); n != 1 {
				t.Errorf("%d handshakes, want the session kept", n)
			}
		})
	}
}

// Ported from upstream (test_cartesia_turns_rejects_an_oversized_language_hint_update).
func TestTurnsRejectsAnOversizedLanguageHintUpdate(t *testing.T) {
	r := runHints(t, TurnsSTTConfig{LanguageHints: []language.Language{language.Hindi}})

	r.update(&TurnsSTTSettings{
		Keyterm:       settings.Set([]string{"Cartesia"}),
		LanguageHints: settings.Set(sevenLanguages),
	})
	waitUntil(t, "the update was not refused", func() bool { return len(r.reported()) > 0 })

	if ef := r.reported()[0]; !errors.Is(ef.Err, errTooManyLanguageHints) || ef.Fatal {
		t.Errorf("error = %v (fatal %v), want a non-fatal too many language hints", ef.Err, ef.Fatal)
	}
	time.Sleep(100 * time.Millisecond)
	if got := r.server.configs(); len(got) != 0 {
		t.Errorf("config messages %v were sent, want none", got)
	}
	if n := len(r.server.handshakes()); n != 1 {
		t.Errorf("%d handshakes, want the session kept", n)
	}
}

// Ported from upstream
// (test_cartesia_turns_update_language_hints_without_a_connection_waits_for_the_url).
func TestTurnsUpdateLanguageHintsWithoutAConnectionWaitsForTheURL(t *testing.T) {
	c := newTurnsConnector(TurnsSTTConfig{APIKey: "k"})
	delta := &TurnsSTTSettings{LanguageHints: settings.Set([]language.Language{language.Japanese})}
	if err := c.PrepareSettings(delta); err != nil {
		t.Fatalf("PrepareSettings: %v", err)
	}
	changed, err := settings.Apply(c.live, delta)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if reopen, err := c.UpdateSettings(t.Context(), changed); err != nil || reopen {
		t.Errorf("reopen = %v, %v; want the hints left for the next session", reopen, err)
	}
	if got := c.languageHints(); !slices.Equal(got, []string{"ja"}) {
		t.Errorf("hints = %v, want [ja] for the next session's URL", got)
	}
}

// Ported from upstream
// (test_cartesia_turns_update_with_a_reconnect_carries_language_hints_in_the_url).
func TestTurnsUpdateWithAReopenCarriesLanguageHintsInTheURL(t *testing.T) {
	r := runHints(t, TurnsSTTConfig{})

	r.update(&TurnsSTTSettings{
		Keyterm:       settings.Set([]string{"Cartesia"}),
		LanguageHints: settings.Set([]language.Language{language.French}),
	})
	waitUntil(t, "the session was not reopened", func() bool { return len(r.server.handshakes()) == 2 })

	if got := r.server.handshakes()[1]["language_hint"]; !slices.Equal(got, []string{"fr"}) {
		t.Errorf("language_hint = %v, want [fr]", got)
	}
	if got := r.server.configs(); len(got) != 0 {
		t.Errorf("config messages %v were sent, want none", got)
	}
}

// turnsResults connects to a server replaying events and returns the results
// of the first n reads.
func turnsResults(t *testing.T, cfg TurnsSTTConfig, n int, evs ...map[string]any) []stt.Result {
	t.Helper()
	endpoint, _ := turnsServer(t, append([]map[string]any{{"type": "connected", "request_id": "r1"}}, evs...))
	cfg.APIKey, cfg.URL, cfg.Version, cfg.Model = "k", endpoint, defaultVersion, defaultTurnsModel
	stream, err := newTurnsConnector(cfg).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	var out []stt.Result
	for range n {
		res, err := stream.Recv()
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		out = append(out, res...)
	}
	return out
}

// Ported from upstream (test_cartesia_turns_labels_transcripts_with_the_detected_language).
func TestTurnsLabelsTranscriptsWithTheDetectedLanguage(t *testing.T) {
	tests := []struct {
		name      string
		languages any
		want      string
	}{
		{"strongest", []map[string]any{{"language": "es", "confidence": 0.8}, {"language": "en", "confidence": 0.2}}, "es"},
		{"three-letter-code", []map[string]any{{"language": "yue", "confidence": 0.9}}, "yue"},
		{"empty", []map[string]any{}, ""},
		{"null", nil, ""},
		{"unknown", []map[string]any{{"language": "not-a-language", "confidence": 0.9}}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := turnsResults(t, TurnsSTTConfig{LanguageHints: []language.Language{language.Hindi}}, 1,
				map[string]any{"type": "turn.end", "transcript": "hola", "languages": tt.languages})
			if len(got) != 1 || !got[0].Final || got[0].Language != tt.want {
				t.Errorf("results = %+v, want a final in %q", got, tt.want)
			}
		})
	}
}

// Ported from upstream (test_cartesia_turns_uses_each_events_language_detection).
func TestTurnsUsesEachEventsLanguageDetection(t *testing.T) {
	got := turnsResults(t, TurnsSTTConfig{EnableEagerEndOfTurn: true}, 4,
		map[string]any{
			"type": "turn.eager_end", "transcript": "bonjour",
			"languages": []map[string]any{{"language": "fr", "confidence": 0.9}},
		},
		map[string]any{"type": "turn.resume"},
		map[string]any{"type": "turn.update", "transcript": "bonjour à tous"},
		map[string]any{"type": "turn.end", "transcript": "bonjour à tous"},
	)
	type labeled struct {
		eager, final bool
		language     string
	}
	var transcripts []labeled
	for _, r := range got {
		if r.Text != "" {
			transcripts = append(transcripts, labeled{r.EagerEndOfTurn, r.Final, r.Language})
		}
	}
	want := []labeled{{true, false, "fr"}, {false, false, ""}, {false, true, ""}}
	if !slices.Equal(transcripts, want) {
		t.Errorf("transcripts = %+v, want the eager one in fr and the rest in none", transcripts)
	}
}
