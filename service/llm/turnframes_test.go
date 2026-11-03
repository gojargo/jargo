package llm_test

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/gojargo/jargo/service/llm"
)

// A realtime service emitting no turn frames is reported once; one that emits
// them is not reported. (Not a port of an upstream test.)
func TestARealtimeServiceWithoutTurnFramesIsReportedOnce(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	svc := llm.New("FakeLLM", &fakeGen{})
	svc.WarnIfRealtimeServiceEmitsNoTurnFrames(true)
	if buf.Len() != 0 {
		t.Fatalf("logged %q for a service that emits turn frames", buf.String())
	}
	svc.WarnIfRealtimeServiceEmitsNoTurnFrames(false)
	svc.WarnIfRealtimeServiceEmitsNoTurnFrames(false)
	if n := strings.Count(buf.String(), "not emitting turn frames"); n != 1 {
		t.Errorf("reported %d times, want once: %q", n, buf.String())
	}
}
