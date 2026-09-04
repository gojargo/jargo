package processor_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Upstream has no tests for this processor; these are jargo's own.

// captureLogs points the default logger at a buffer for the duration of a test.
func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &buf
}

// A logger says what went by and which way, and passes the frame on unchanged.
func TestFrameLoggerLogsAndPassesOn(t *testing.T) {
	logs := captureLogs(t)
	p := processor.NewFrameLogger("Logger", processor.WithLoggedPrefix("mine"))
	_, down := linkAndStart(t, p)

	if err := p.QueueFrame(context.Background(), frames.NewTextFrame("hello"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.TextFrame](t, down.got, "TextFrame")

	if got := logs.String(); !strings.Contains(got, "mine: TextFrame") {
		t.Errorf("logs = %q, want the prefix and the frame name", got)
	}
	if got := logs.String(); !strings.Contains(got, "> mine") {
		t.Errorf("logs = %q, want the direction the frame traveled", got)
	}
}

// The high-frequency frames are skipped by default: audio arrives many times a
// second and would bury everything else.
func TestFrameLoggerSkipsTheNoisyFramesByDefault(t *testing.T) {
	logs := captureLogs(t)
	p := processor.NewFrameLogger("Logger")
	_, down := linkAndStart(t, p)

	ctx := context.Background()
	if err := p.QueueFrame(ctx, frames.NewBotSpeakingFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if err := p.QueueFrame(ctx, frames.NewTextFrame("hello"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.BotSpeakingFrame](t, down.got, "BotSpeakingFrame")
	mustReceive[*frames.TextFrame](t, down.got, "TextFrame")

	got := logs.String()
	if strings.Contains(got, "BotSpeakingFrame") {
		t.Errorf("logs = %q, want the speaking frame skipped", got)
	}
	if !strings.Contains(got, "TextFrame") {
		t.Errorf("logs = %q, want the text frame logged", got)
	}
}

// A caller who wants the whole stream, audio included, can have it.
func TestFrameLoggerCanLogEverything(t *testing.T) {
	logs := captureLogs(t)
	p := processor.NewFrameLogger("Logger", processor.WithIgnoredFrames(nil))
	_, down := linkAndStart(t, p)

	if err := p.QueueFrame(context.Background(), frames.NewBotSpeakingFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.BotSpeakingFrame](t, down.got, "BotSpeakingFrame")

	if got := logs.String(); !strings.Contains(got, "BotSpeakingFrame") {
		t.Errorf("logs = %q, want the speaking frame logged", got)
	}
}

// keyedService stands in for a service holding the credentials it was built
// with. The key is an exported field, which is what a structured handler walks
// into when it is handed the frame itself.
type keyedService struct{ APIKey string }

func newKeyedService(apiKey string) *keyedService { return &keyedService{APIKey: apiKey} }

func (s *keyedService) Name() string { return "KeyedSTT" }

// TestFrameLoggerDoesNotWalkIntoAFrame covers the frames that carry a service
// rather than data: the log line is what the frame says about itself, which
// names the service it points at. Handing the frame to the logger instead would
// let a structured handler render its fields, and with them the API key the
// service was built with.
func TestFrameLoggerDoesNotWalkIntoAFrame(t *testing.T) {
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })

	p := processor.NewFrameLogger("Logger")
	_, down := linkAndStart(t, p)

	svc := newKeyedService("sk-not-in-the-log")
	f := frames.NewManuallySwitchServiceFrame(svc)
	if err := p.QueueFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.ManuallySwitchServiceFrame](t, down.got, "ManuallySwitchServiceFrame")

	got := buf.String()
	if strings.Contains(got, "sk-not-in-the-log") {
		t.Errorf("logs = %q, want the service's API key left out", got)
	}
	if !strings.Contains(got, svc.Name()) {
		t.Errorf("logs = %q, want the service named", got)
	}
}
