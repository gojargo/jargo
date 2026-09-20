package live

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/service/llm"
)

// Which model is in force decides how a function call is declared, and whether
// a completed generation ends the turn. The id is the only thing that says so,
// and it names its version with or without a minor part.

func TestVersion(t *testing.T) {
	for _, tc := range []struct {
		model        string
		major, minor int
		known        bool
	}{
		{"gemini-2.0-flash-live-001", 2, 0, true},
		{"gemini-2.5-flash-native-audio-preview-12-2025", 2, 5, true},
		{"gemini-live-2.5-flash-native-audio", 2, 5, true},
		{"gemini-3-pro-preview", 3, 0, true},
		{"gemini-3.8-live", 3, 8, true},
		{"models/gemini-3.8-live-extended-thinking", 3, 8, true},
		{"gemini-4-live", 4, 0, true},
		{"some-other-model", 0, 0, false},
		{"", 0, 0, false},
	} {
		major, minor, known := version(tc.model)
		if major != tc.major || minor != tc.minor || known != tc.known {
			t.Errorf("version(%q) = (%d, %d, %v), want (%d, %d, %v)",
				tc.model, major, minor, known, tc.major, tc.minor, tc.known)
		}
	}
}

// The 3.x models before 3.8 take no declaration saying whether to wait for a
// call; everything else does, a model of no readable generation included.
func TestSupportsNonBlockingTools(t *testing.T) {
	for model, want := range map[string]bool{
		"gemini-2.5-flash-native-audio-preview-12-2025": true,
		"gemini-2.0-flash-live-001":                     true,
		"gemini-3-pro-preview":                          false,
		"gemini-3.8-live":                               true,
		"gemini-3.8-live-extended-thinking":             true,
		"gemini-4-live":                                 true,
		"some-other-model":                              true,
	} {
		if got := supportsNonBlockingTools(model); got != want {
			t.Errorf("supportsNonBlockingTools(%q) = %v, want %v", model, got, want)
		}
	}
}

// The 3.8 family runs a call without waiting unless the declaration asks it to,
// and the thinking models refuse to be asked.
func TestToolWaitingDefaults(t *testing.T) {
	for _, tc := range []struct {
		model                    string
		defaultsNonBlocking      bool
		acceptsBlocking          bool
		reportsInteractionStatus bool
	}{
		{"gemini-2.0-flash-live-001", false, true, false},
		{"gemini-3-pro-preview", false, true, false},
		{"gemini-3.8-live", true, true, false},
		{"gemini-3.8-live-extended-thinking", true, false, true},
		// Named for thinking, but older than the status it would report.
		{"gemini-2.5-flash-exp-native-audio-thinking-dialog", false, true, false},
	} {
		if got := toolsDefaultToNonBlocking(tc.model); got != tc.defaultsNonBlocking {
			t.Errorf("toolsDefaultToNonBlocking(%q) = %v, want %v", tc.model, got, tc.defaultsNonBlocking)
		}
		if got := supportsBlockingTools(tc.model); got != tc.acceptsBlocking {
			t.Errorf("supportsBlockingTools(%q) = %v, want %v", tc.model, got, tc.acceptsBlocking)
		}
		if got := expectsInteractionStatus(tc.model); got != tc.reportsInteractionStatus {
			t.Errorf("expectsInteractionStatus(%q) = %v, want %v", tc.model, got, tc.reportsInteractionStatus)
		}
	}
}

// taggedDeclarations tags one synchronous and one asynchronous declaration on a
// service running the given model, and returns them.
func taggedDeclarations(t *testing.T, model string) []map[string]any {
	t.Helper()
	svc := New(Config{APIKey: "k", Model: model})
	noop := func(ctx context.Context, p llm.FunctionCallParams) error { return p.Result(ctx, "", nil) }
	svc.RegisterFunction("sync_tool", noop)
	svc.RegisterFunction("async_tool", noop, llm.WithCancelOnInterruption(false))

	decls := []map[string]any{{"name": "sync_tool"}, {"name": "async_tool"}}
	svc.tagToolBehaviors([]map[string]any{{"functionDeclarations": decls}})
	return decls
}

// A synchronous tool keeps its blocking semantics wherever the model can honor
// them: the model finishes its turn only once the result has landed, which is
// what keeps it from announcing that it will go and look something up.
func TestSyncToolsBlockWhereTheModelAllowsIt(t *testing.T) {
	for _, tc := range []struct {
		model string
		sync  any
	}{
		// Waiting is already the default outside the 3.8 family, so nothing is said.
		{"gemini-2.0-flash-live-001", nil},
		// The 3.8 family flipped it, so waiting is asked for explicitly.
		{"gemini-3.8-live", "BLOCKING"},
		// The thinking models refuse to be asked, so nothing is said there either.
		{"gemini-3.8-live-extended-thinking", nil},
	} {
		decls := taggedDeclarations(t, tc.model)
		if got := decls[0]["behavior"]; got != tc.sync {
			t.Errorf("%s: the synchronous tool was declared %v, want %v", tc.model, got, tc.sync)
		}
		if got := decls[1]["behavior"]; got != "NON_BLOCKING" {
			t.Errorf("%s: the asynchronous tool was declared %v, want NON_BLOCKING", tc.model, got)
		}
	}
}

// A synchronous tool cannot block on a model that runs every call without
// waiting, which is worth saying, and worth saying once however many tools the
// session advertises.
func TestASyncToolThatCannotBlockIsReportedOnce(t *testing.T) {
	svc := New(Config{APIKey: "k", Model: "gemini-3.8-live-extended-thinking"})
	noop := func(ctx context.Context, p llm.FunctionCallParams) error { return p.Result(ctx, "", nil) }
	svc.RegisterFunction("sync_tool", noop)

	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	for range 2 {
		svc.tagToolBehaviors([]map[string]any{{"functionDeclarations": []map[string]any{{"name": "sync_tool"}}}})
	}
	if got := strings.Count(buf.String(), "does not pause the conversation"); got != 1 {
		t.Errorf("the warning was logged %d times, want once", got)
	}
}

// On a model that takes a blocking declaration the synchronous semantics are
// restored rather than lost, so there is nothing to report.
func TestNothingIsReportedWhenASyncToolCanBlock(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	taggedDeclarations(t, "gemini-3.8-live")
	if buf.Len() != 0 {
		t.Errorf("logged %q, want nothing said", buf.String())
	}
}

// The Live thinking models require a thinking level and refuse a setup that
// sets none, so an unset one takes the lowest they accept. Every other model is
// left exactly as it was configured.
func TestThinkingLevelDefaultsOnlyOnLiveThinkingModels(t *testing.T) {
	for _, tc := range []struct {
		model string
		level string // "" means no thinking configuration at all
	}{
		{"gemini-3.8-live-extended-thinking", lowestThinkingLevel},
		{"gemini-3.8-live", ""},
		{"gemini-2.0-flash-live-001", ""},
		{"gemini-2.5-flash-exp-native-audio-thinking-dialog", ""},
	} {
		got := resolvedThinking(tc.model, nil)
		switch {
		case tc.level == "" && got != nil:
			t.Errorf("%s: thinking = %+v, want none configured", tc.model, got)
		case tc.level != "" && (got == nil || got.Level != tc.level):
			t.Errorf("%s: thinking = %+v, want level %q", tc.model, got, tc.level)
		}
	}
}

// A level the caller chose is never overridden by the default.
func TestAConfiguredThinkingLevelIsKept(t *testing.T) {
	got := resolvedThinking("gemini-3.8-live-extended-thinking", &gemini.ThinkingConfig{Level: "HIGH"})
	if got == nil || got.Level != "HIGH" {
		t.Errorf("thinking = %+v, want the configured level", got)
	}
}

// Defaulting the level keeps the rest of the configuration, and leaves the
// caller's own value as it stands.
func TestDefaultingTheLevelKeepsTheRestOfTheConfiguration(t *testing.T) {
	cfg := &gemini.ThinkingConfig{IncludeThoughts: true}
	got := resolvedThinking("gemini-3.8-live-extended-thinking", cfg)

	if got == nil || got.Level != lowestThinkingLevel || !got.IncludeThoughts {
		t.Errorf("thinking = %+v, want the level defaulted and the rest kept", got)
	}
	if cfg.Level != "" {
		t.Errorf("the configured value was edited: %+v", cfg)
	}
}

// The setup message is the only place the session takes its configuration, and
// the API reads the thinking block inside the generation config rather than
// beside it.
func TestTheSetupCarriesTheThinkingConfig(t *testing.T) {
	svc := New(Config{APIKey: "k", Model: "gemini-3.8-live-extended-thinking"})
	gen, ok := svc.setup()["setup"].(map[string]any)["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("setup carries no generation config: %v", svc.setup())
	}
	thinking, ok := gen["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generation config carries no thinking block: %v", gen)
	}
	if thinking["thinkingLevel"] != lowestThinkingLevel {
		t.Errorf("thinkingLevel = %v, want %q", thinking["thinkingLevel"], lowestThinkingLevel)
	}

	// A model that needs no level is left without the block entirely.
	plain := New(Config{APIKey: "k", Model: "gemini-3.8-live"})
	gen, _ = plain.setup()["setup"].(map[string]any)["generationConfig"].(map[string]any)
	if _, ok := gen["thinkingConfig"]; ok {
		t.Errorf("generation config = %v, want no thinking block", gen)
	}
}
