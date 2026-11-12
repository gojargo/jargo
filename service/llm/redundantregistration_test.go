package llm_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// captureWarningLog collects what is logged at warning level or above while the
// test runs.
func captureWarningLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

func lookupOrderTool() frames.Tool {
	return frames.Tool{
		Name:    "lookup_order",
		Handler: llm.FunctionCallHandler(func(context.Context, llm.FunctionCallParams) error { return nil }),
	}
}

// Ported from upstream (test_explicit_registration_wins_and_warns and
// test_warning_fires_once_across_repeated_syncs). A handler registered by hand
// wins over the one its advertised tool carries, and the redundant registration
// is warned about once, however often the tools are synced.
func TestExplicitRegistrationWinsAndWarnsOnce(t *testing.T) {
	logs := captureWarningLog(t)
	svc := llm.New("FakeLLM", &advertisingGen{})
	svc.RegisterFunction("lookup_order", noop)
	for range 3 {
		svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{Standard: []frames.Tool{lookupOrderTool()}})
	}
	if n := strings.Count(logs.String(), "lookup_order"); n != 1 {
		t.Errorf("warned %d times, want once:\n%s", n, logs)
	}
	if !strings.Contains(logs.String(), "unnecessary") {
		t.Errorf("logs = %s, want the registration called unnecessary", logs)
	}
}

// Ported from upstream (test_no_warning_for_advertise_only_schema). Registering
// by hand the handler of a tool advertised without one is the ordinary pattern,
// and is not warned about.
func TestNoWarningForAnAdvertiseOnlyTool(t *testing.T) {
	logs := captureWarningLog(t)
	svc := llm.New("FakeLLM", &advertisingGen{})
	svc.RegisterFunction("advertise_only", noop)
	svc.SyncToolHandlers(context.Background(), frames.ToolsSchema{Standard: []frames.Tool{{Name: "advertise_only"}}})
	if logs.Len() != 0 {
		t.Errorf("logs = %s, want nothing", logs)
	}
}
