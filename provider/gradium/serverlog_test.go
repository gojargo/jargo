package gradium

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// Tests for the server's log messages: asked for with a log level, logged at
// their own level.

// serverLog is the warning the server sends about a setup it does not fully
// understand.
func serverLog() map[string]any {
	return map[string]any{msgType: msgLog, "level": "warning", "message": "unused json_config keys: tmep"}
}

// wantServerLog is how serverLog is logged.
const wantServerLog = "WARN gradium warning: unused json_config keys: tmep"

// logRecorder keeps what is logged during a test, as level and message.
type logRecorder struct {
	mu      sync.Mutex
	records []string
}

func (r *logRecorder) Enabled(context.Context, slog.Level) bool { return true }

func (r *logRecorder) Handle(_ context.Context, rec slog.Record) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, fmt.Sprintf("%s %s", rec.Level, rec.Message))
	return nil
}

func (r *logRecorder) WithAttrs([]slog.Attr) slog.Handler { return r }
func (r *logRecorder) WithGroup(string) slog.Handler      { return r }

func (r *logRecorder) logged() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.records)
}

// recordLogs routes the default logger to a recorder for the test.
func recordLogs(t *testing.T) *logRecorder {
	t.Helper()
	rec := &logRecorder{}
	prev := slog.Default()
	slog.SetDefault(slog.New(rec))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return rec
}

// Ported from upstream (test_a_server_log_is_logged_at_its_own_level).
func TestAServerLogIsLoggedAtItsOwnLevel(t *testing.T) {
	for _, tc := range []struct{ level, want string }{
		{"debug", "DEBUG gradium debug: m"},
		{"info", "INFO gradium info: m"},
		{"warning", "WARN gradium warning: m"},
		{"error", "ERROR gradium error: m"},
		{"unheard-of", "WARN gradium warning: m"},
	} {
		t.Run(tc.level, func(t *testing.T) {
			rec := recordLogs(t)

			logServerMessage(context.Background(), tc.level, "m")

			if got := rec.logged(); !slices.Equal(got, []string{tc.want}) {
				t.Errorf("logged %q, want %q", got, tc.want)
			}
		})
	}
}

// Ported from upstream (test_stt_asks_for_server_logs_only_when_enabled). A log
// that comes before ready is logged, and the connection still completes.
func TestSTTAsksForServerLogsOnlyWhenEnabled(t *testing.T) {
	for _, tc := range []struct {
		level LogLevel
		want  any
	}{
		{"", nil},
		{LogLevelWarning, "warning"},
	} {
		t.Run(fmt.Sprintf("level %q", tc.level), func(t *testing.T) {
			rec := recordLogs(t)
			endpoint, seen := sttServer(t, serverLog(), []map[string]any{{msgType: "ready"}})

			s, err := sttConn(endpoint, func(c *STTConfig) { c.LogLevel = tc.level }).
				Connect(context.Background(), 16000)
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() { _ = s.Close() }()

			if got := seen.await(t)[keyLogLevel]; got != tc.want {
				t.Errorf("setup log_level = %v, want %v", got, tc.want)
			}
			if !slices.Contains(rec.logged(), wantServerLog) {
				t.Errorf("logged %q, want the server's warning", rec.logged())
			}
		})
	}
}

// Ported from upstream (test_stt_logs_a_server_log_received_during_the_session).
func TestSTTLogsAServerLogReceivedDuringTheSession(t *testing.T) {
	rec := recordLogs(t)
	s := newTestStream()
	s.feed(sttMessage{Type: msgLog, Level: "warning", Message: "unused json_config keys: tmep"}, text("hello"))

	// The log is no result: the next one is the text after it.
	if got := recvOne(t, s); got.Text != "hello" {
		t.Errorf("result = %+v, want the text after the log", got)
	}
	if !slices.Contains(rec.logged(), wantServerLog) {
		t.Errorf("logged %q, want the server's warning", rec.logged())
	}
}

// Ported from upstream (test_tts_asks_for_server_logs_only_when_enabled).
func TestTTSAsksForServerLogsOnlyWhenEnabled(t *testing.T) {
	for _, tc := range []struct {
		level LogLevel
		want  any
	}{
		{"", nil},
		{LogLevelError, LogLevelError},
	} {
		_, s := testTTS("ws://unused", func(c *TTSConfig) { c.LogLevel = tc.level })
		if got := s.setupMessage("ctx")[keyLogLevel]; got != tc.want {
			t.Errorf("level %q: setup log_level = %v, want %v", tc.level, got, tc.want)
		}
	}
}

// Ported from upstream (test_tts_logs_a_server_log_received_during_the_session).
// The log is logged, raises no error and pushes no frame: the request goes on.
func TestTTSLogsAServerLogReceivedDuringTheSession(t *testing.T) {
	rec := recordLogs(t)
	endpoint, url := newTTSEndpoint(t, serverLog(), ttsAudio([]byte{9, 9}))
	svc, _ := testTTS(url)

	down, up := runTTSPipeline(t, svc, frames.NewTTSSpeakFrame("hello"), 300*time.Millisecond)

	if !slices.Contains(rec.logged(), wantServerLog) {
		t.Errorf("logged %q, want the server's warning", rec.logged())
	}
	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	if got := playedAudio(down); len(got) != 2 {
		t.Errorf("audio = %v, want the chunk after the log", got)
	}
	if n := endpoint.dialCount(); n != 1 {
		t.Errorf("dialed %d connections, want the first one kept", n)
	}
}
