package cartesia

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/utils/events"
)

// turnsQuery opens a session on a connector built from cfg and returns the
// query the server was asked with.
func turnsQuery(t *testing.T, cfg TurnsSTTConfig) map[string][]string {
	t.Helper()
	endpoint, query := turnsServer(t, []map[string]any{{"type": "connected", "request_id": "r1"}})
	cfg.APIKey, cfg.URL, cfg.Version = "k", endpoint, defaultVersion
	stream, err := newTurnsConnector(cfg).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close() })
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if q := query(); q != nil {
			return q
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("the server was never asked")
	return nil
}

// Ported from upstream (test_cartesia_turns_websocket_url_includes_turn_detection_thresholds).
func TestTurnsURLIncludesTheTurnDetectionThresholds(t *testing.T) {
	q := turnsQuery(t, TurnsSTTConfig{
		TurnStartThreshold: new(0.7), TurnEagerEndThreshold: new(0.5),
		TurnEndThreshold: new(0.4), TurnEndTimeoutMs: new(4500),
	})
	for name, want := range map[string]string{
		"turn_start_threshold": "0.7", "turn_eager_end_threshold": "0.5",
		"turn_end_threshold": "0.4", "turn_end_timeout_ms": "4500",
	} {
		if got := q[name]; len(got) != 1 || got[0] != want {
			t.Errorf("%s = %v, want %s", name, got, want)
		}
	}
}

// Ported from upstream (test_cartesia_turns_websocket_url_omits_unset_turn_detection_thresholds).
func TestTurnsURLOmitsUnsetTurnDetectionThresholds(t *testing.T) {
	q := turnsQuery(t, TurnsSTTConfig{TurnEndTimeoutMs: new(8000)})
	if got := q["turn_end_timeout_ms"]; len(got) != 1 || got[0] != "8000" {
		t.Errorf("turn_end_timeout_ms = %v, want 8000", got)
	}
	for _, name := range []string{"turn_start_threshold", "turn_eager_end_threshold", "turn_end_threshold"} {
		if _, ok := q[name]; ok {
			t.Errorf("%s was sent unset", name)
		}
	}
}

// A turn detection setting is bound to the connection, so changing one reopens
// the session, as a keyterm change does; a model change does not.
func TestTurnsTurnDetectionChangeReopens(t *testing.T) {
	c := newTurnsConnector(TurnsSTTConfig{APIKey: "k"})
	reopen, err := c.UpdateSettings(t.Context(), settings.Changed{"turn_end_timeout_ms": nil})
	if err != nil || !reopen {
		t.Errorf("reopen = %v, %v; want a turn end timeout change to reopen the session", reopen, err)
	}
	if reopen, _ := c.UpdateSettings(t.Context(), settings.Changed{"model": "x"}); reopen {
		t.Error("a model change reopened the session")
	}
}

// Ported from upstream (test_cartesia_turns_recommends_eager_strategies_when_asked
// and ..._external_strategies_by_default).
func TestTurnsRecommendsEagerStrategiesWhenAsked(t *testing.T) {
	isEager := func(recommended any) bool {
		s, _ := recommended.(turns.UserTurnStrategies)
		for _, stop := range s.Stop {
			if _, ok := stop.(*turns.EagerStop); ok {
				return true
			}
		}
		return false
	}
	if isEager(newTurnsConnector(TurnsSTTConfig{}).Metadata().UserTurnStrategies) {
		t.Error("the eager strategies are recommended by default")
	}
	if !isEager(newTurnsConnector(TurnsSTTConfig{EnableEagerEndOfTurn: true}).Metadata().UserTurnStrategies) {
		t.Error("the eager strategies were not recommended with eager end of turn on")
	}
}

// Ported from upstream (test_cartesia_turns_reports_an_eager_end_of_turn_only_when_enabled).
// The server's predicted end of a turn becomes an EagerTranscriptionFrame only
// with eager end of turn on; either way its text is not pushed as a transcript.
func TestTurnsReportsAnEagerEndOfTurnOnlyWhenEnabled(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		endpoint, _ := turnsServer(t, []map[string]any{
			{"type": "connected", "request_id": "r1"},
			{"type": "turn.start", "transcript": ""},
			{"type": "turn.eager_end", "transcript": "book a flight"},
		})
		svc := NewTurnsSTT(TurnsSTTConfig{APIKey: "k", URL: endpoint, SampleRate: 16000, EnableEagerEndOfTurn: enabled})
		var mu sync.Mutex
		var eager, interim int
		task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
		events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.EagerTranscriptionFrame:
				eager++
			case *frames.InterimTranscriptionFrame:
				if fr.Text == "book a flight" {
					interim++
				}
			}
		})
		done := make(chan error, 1)
		go func() { done <- task.Run(context.Background()) }()
		time.Sleep(400 * time.Millisecond)
		task.StopWhenDone()
		<-done

		mu.Lock()
		want := 0
		if enabled {
			want = 1
		}
		if eager != want || interim != 0 {
			t.Errorf("enabled %v: %d eager and %d interim frames, want %d and 0", enabled, eager, interim, want)
		}
		mu.Unlock()
	}
}
