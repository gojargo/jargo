package service_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

// captureWarnings routes the default logger to a buffer until the returned
// function is called.
func captureWarnings(t *testing.T) (*bytes.Buffer, func()) {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	return &buf, func() { slog.SetDefault(prev) }
}

// The settings an update changed that will not take effect are reported once,
// named in a stable order, and nothing is reported when there are none. (Not a
// port of an upstream test.)
func TestWarnUnhandledUpdatedSettings(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()
	svc := newDescriber("Svc")

	svc.WarnUnhandledUpdatedSettings(context.Background(), nil)
	if buf.Len() != 0 {
		t.Fatalf("logged %q for nothing unhandled", buf.String())
	}

	svc.WarnUnhandledUpdatedSettings(context.Background(), []string{"voice", "language"})
	out := buf.String()
	if strings.Count(out, "not currently supported") != 1 {
		t.Fatalf("log = %q, want one warning", out)
	}
	if !strings.Contains(out, `fields="language, voice"`) || !strings.Contains(out, "service=Svc") {
		t.Errorf("log = %q, want the service and the sorted fields", out)
	}
}
