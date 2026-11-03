package realtime

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

// sentEvents records the client events a service sends.
type sentEvents struct {
	mu     sync.Mutex
	events []any
}

func (r *sentEvents) sessionUpdates() []events.SessionUpdateEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []events.SessionUpdateEvent
	for _, e := range r.events {
		if u, ok := e.(events.SessionUpdateEvent); ok {
			out = append(out, u)
		}
	}
	return out
}

// offlineService is a service set up against a server that never answers, its
// events captured rather than sent.
func offlineService(t *testing.T, cfg Config) (*Service, *sentEvents) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		for {
			if _, _, err := c.Read(r.Context()); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	cfg.APIKey = "test-key"
	cfg.BaseURL = "ws" + strings.TrimPrefix(srv.URL, "http")
	s := New(cfg)
	ctx := context.Background()
	if err := s.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = s.Cleanup(ctx) })
	sent := &sentEvents{}
	s.write = func(v any) error {
		sent.mu.Lock()
		sent.events = append(sent.events, v)
		sent.mu.Unlock()
		return nil
	}
	return s, sent
}

// sendAndCapture runs a session update and returns the session it sent.
func sendAndCapture(t *testing.T, s *Service, sent *sentEvents) events.SessionProperties {
	t.Helper()
	s.sendSessionUpdate(context.Background())
	updates := sent.sessionUpdates()
	if len(updates) != 1 {
		t.Fatalf("sent %d session updates, want one", len(updates))
	}
	return updates[0].Session
}

func captureWarnings(t *testing.T) (*bytes.Buffer, func()) {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	return &buf, func() { slog.SetDefault(prev) }
}

// Ports of the service cases of tests/test_openai_realtime_reasoning.py.

func TestOutgoingSessionUpdateStripsReasoningOnUnsupportedModel(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()
	s, sent := offlineService(t, Config{
		Model: "gpt-realtime-1.5", Instructions: "be helpful",
		SessionProperties: events.SessionProperties{Reasoning: &events.Reasoning{Effort: "high"}},
	})
	outgoing := sendAndCapture(t, s, sent)
	if outgoing.Reasoning != nil {
		t.Errorf("reasoning = %+v, want it stripped on the wire", outgoing.Reasoning)
	}
	text := buf.String()
	if !strings.Contains(text, "stripping reasoning") || !strings.Contains(text, "gpt-realtime-1.5") {
		t.Errorf("log = %q, want the strip reported with the model", text)
	}
	if r := s.sessionProperties().Reasoning; r == nil || r.Effort != "high" {
		t.Errorf("stored reasoning = %+v, want it kept", r)
	}
}

func TestOutgoingSessionUpdateKeepsReasoningOnSupportedModel(t *testing.T) {
	for _, model := range []string{"gpt-realtime-2", "gpt-realtime-2-some-variant"} {
		buf, restore := captureWarnings(t)
		s, sent := offlineService(t, Config{
			Model: model, Instructions: "be helpful",
			SessionProperties: events.SessionProperties{Reasoning: &events.Reasoning{Effort: "high"}},
		})
		outgoing := sendAndCapture(t, s, sent)
		restore()
		if outgoing.Reasoning == nil || outgoing.Reasoning.Effort != "high" {
			t.Errorf("%s: reasoning = %+v, want it kept", model, outgoing.Reasoning)
		}
		if strings.Contains(buf.String(), "stripping") {
			t.Errorf("%s: log = %q, want nothing", model, buf.String())
		}
	}
}

func TestNoWarningWhenReasoningIsUnsetOnUnsupportedModel(t *testing.T) {
	buf, restore := captureWarnings(t)
	defer restore()
	s, sent := offlineService(t, Config{Model: "gpt-realtime-1.5", Instructions: "be helpful"})
	if outgoing := sendAndCapture(t, s, sent); outgoing.Reasoning != nil {
		t.Errorf("reasoning = %+v, want none", outgoing.Reasoning)
	}
	if strings.Contains(buf.String(), "stripping") {
		t.Errorf("log = %q, want nothing", buf.String())
	}
}

// update sends a settings update the way the pipeline does.
func update(t *testing.T, s *Service, delta *Settings) {
	t.Helper()
	if err := s.Base.ProcessFrame(context.Background(), frames.NewLLMUpdateSettingsFrame(delta),
		processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
}

func TestRuntimeSettingsUpdateWithReasoningTriggersSessionUpdate(t *testing.T) {
	s, sent := offlineService(t, Config{
		Model: "gpt-realtime-2", Instructions: "be helpful",
		SessionProperties: events.SessionProperties{Reasoning: &events.Reasoning{Effort: "low"}},
	})
	update(t, s, &Settings{SessionProperties: settings.Set(events.SessionProperties{
		Reasoning: &events.Reasoning{Effort: "high"},
	})})
	updates := sent.sessionUpdates()
	if len(updates) != 1 {
		t.Fatalf("sent %d session updates, want one", len(updates))
	}
	if r := updates[0].Session.Reasoning; r == nil || r.Effort != "high" {
		t.Errorf("reasoning = %+v, want high", r)
	}
}

func TestRuntimeSettingsUpdateStripsReasoningOnUnsupportedModel(t *testing.T) {
	s, sent := offlineService(t, Config{Model: "gpt-realtime-1.5", Instructions: "be helpful"})
	update(t, s, &Settings{SessionProperties: settings.Set(events.SessionProperties{
		Reasoning: &events.Reasoning{Effort: "high"},
	})})
	updates := sent.sessionUpdates()
	if len(updates) != 1 || updates[0].Session.Reasoning != nil {
		t.Errorf("updates = %+v, want one with reasoning stripped", updates)
	}
}

// Ports of TestOpenAIRealtimeSettingsApplyUpdate (tests/test_settings.py).

func store(sp events.SessionProperties, instruction string) *Settings {
	return newSettings(Config{Model: "gpt-realtime-1.5", Instructions: instruction, SessionProperties: sp})
}

func TestTopLevelModelSyncsToSP(t *testing.T) {
	s := store(events.SessionProperties{}, "")
	delta := &Settings{}
	delta.Model = settings.Set("gpt-realtime-2.0")
	changed, err := settings.Apply(s, delta)
	if err != nil || !changed.Has("model") {
		t.Fatalf("changed = %v, err = %v, want the model", changed, err)
	}
	if sp, _ := s.SessionProperties.Value(); s.Model.Or("") != "gpt-realtime-2.0" || sp.Model != "gpt-realtime-2.0" {
		t.Errorf("model = %q, properties' = %q, want both updated", s.Model.Or(""), sp.Model)
	}
}

func TestTopLevelSystemInstructionSyncsToSP(t *testing.T) {
	s := store(events.SessionProperties{}, "")
	delta := &Settings{}
	delta.SystemInstruction = settings.Set("Be helpful.")
	changed, _ := settings.Apply(s, delta)
	sp, _ := s.SessionProperties.Value()
	if !changed.Has("system_instruction") || sp.Instructions != "Be helpful." {
		t.Errorf("changed = %v, instructions = %q, want both updated", changed, sp.Instructions)
	}
}

func TestSPReplacesWholesale(t *testing.T) {
	s := store(events.SessionProperties{
		OutputModalities: []events.Modality{events.ModalityAudio, events.ModalityText},
		Instructions:     "Old instructions.",
	}, "Old instructions.")
	changed, _ := settings.Apply(s, &Settings{SessionProperties: settings.Set(events.SessionProperties{
		OutputModalities: []events.Modality{events.ModalityText},
	})})
	sp, _ := s.SessionProperties.Value()
	if !changed.Has("session_properties") || len(sp.OutputModalities) != 1 || sp.OutputModalities[0] != "text" {
		t.Errorf("properties = %+v, want them replaced", sp)
	}
	if sp.Model != "gpt-realtime-1.5" {
		t.Errorf("model = %q, want it synced from the top level", sp.Model)
	}
}

func TestSPModelSyncsToTopLevel(t *testing.T) {
	s := store(events.SessionProperties{}, "")
	changed, _ := settings.Apply(s, &Settings{SessionProperties: settings.Set(events.SessionProperties{
		Model: "gpt-realtime-2.0",
	})})
	sp, _ := s.SessionProperties.Value()
	if !changed.Has("model") || s.Model.Or("") != "gpt-realtime-2.0" || sp.Model != "gpt-realtime-2.0" {
		t.Errorf("changed = %v, model = %q, want the properties' model taken up", changed, s.Model.Or(""))
	}
}

func TestSPInstructionsSyncsToTopLevel(t *testing.T) {
	s := store(events.SessionProperties{}, "")
	changed, _ := settings.Apply(s, &Settings{SessionProperties: settings.Set(events.SessionProperties{
		Instructions: "New instructions.",
	})})
	if !changed.Has("system_instruction") || s.SystemInstruction.Or("") != "New instructions." {
		t.Errorf("changed = %v, instruction = %q, want the properties' taken up", changed, s.SystemInstruction.Or(""))
	}
}

func TestTopLevelTakesPrecedenceOverSP(t *testing.T) {
	s := store(events.SessionProperties{}, "")
	delta := &Settings{SessionProperties: settings.Set(events.SessionProperties{
		Model: "sp-model", Instructions: "sp instructions",
	})}
	delta.Model = settings.Set("top-model")
	delta.SystemInstruction = settings.Set("top instructions")
	_, _ = settings.Apply(s, delta)
	sp, _ := s.SessionProperties.Value()
	if s.Model.Or("") != "top-model" || sp.Model != "top-model" ||
		s.SystemInstruction.Or("") != "top instructions" || sp.Instructions != "top instructions" {
		t.Errorf("settings = %+v, properties = %+v, want the top level everywhere", s, sp)
	}
}

func TestNonSyncedFieldUpdateDoesNotAffectSP(t *testing.T) {
	s := store(events.SessionProperties{Instructions: "Keep me."}, "Keep me.")
	delta := &Settings{}
	delta.Temperature = settings.Set(0.5)
	changed, _ := settings.Apply(s, delta)
	sp, _ := s.SessionProperties.Value()
	if !changed.Has("temperature") || changed.Has("session_properties") || sp.Instructions != "Keep me." {
		t.Errorf("changed = %v, instructions = %q, want only the temperature changed", changed, sp.Instructions)
	}
}

// Ports of TestOpenAIRealtimeSessionProperties.

func TestRealtimeWhisperPromptIsOmitted(t *testing.T) {
	s := store(events.SessionProperties{Audio: &events.AudioConfiguration{Input: &events.AudioInput{
		Transcription: &events.InputAudioTranscription{Model: events.WhisperModel, Prompt: "Keywords: coffee"},
	}}}, "")
	// newSettings already omitted it; a prompt given again is omitted again.
	if sp, _ := s.SessionProperties.Value(); sp.InputTranscription().Prompt != "" {
		t.Errorf("prompt = %q, want it omitted", sp.InputTranscription().Prompt)
	}
	sp, _ := s.SessionProperties.Value()
	sp.InputTranscription().Prompt = "Keywords: coffee"
	s.SessionProperties = settings.Set(sp)
	if !omitUnsupportedTranscriptionPrompt(s) || sp.InputTranscription().Prompt != "" {
		t.Errorf("prompt = %q, want it omitted and reported", sp.InputTranscription().Prompt)
	}
}

func TestSupportedTranscriptionModelKeepsPrompt(t *testing.T) {
	s := store(events.SessionProperties{Audio: &events.AudioConfiguration{Input: &events.AudioInput{
		Transcription: &events.InputAudioTranscription{Model: "gpt-4o-transcribe", Prompt: "Keywords: coffee"},
	}}}, "")
	sp, _ := s.SessionProperties.Value()
	if omitUnsupportedTranscriptionPrompt(s) || sp.InputTranscription().Prompt != "Keywords: coffee" {
		t.Errorf("prompt = %q, want it kept", sp.InputTranscription().Prompt)
	}
}

// Ports of TestOpenAIRealtimeSettingsFromMapping.

func fromMap(t *testing.T, values map[string]any) *Settings {
	t.Helper()
	delta := &Settings{}
	if err := settings.FromMap(delta, values); err != nil {
		t.Fatalf("from map: %v", err)
	}
	return delta
}

func TestSPKeysRouteToSessionProperties(t *testing.T) {
	delta := fromMap(t, map[string]any{"instructions": "Be concise.", "output_modalities": []any{"text"}})
	sp, ok := delta.SessionProperties.Value()
	if !ok || sp.Instructions != "Be concise." || len(sp.OutputModalities) != 1 || sp.OutputModalities[0] != "text" {
		t.Errorf("properties = %+v (given %v), want both routed in", sp, ok)
	}
}

func TestModelRoutesToTopLevel(t *testing.T) {
	delta := fromMap(t, map[string]any{"model": "gpt-realtime-2.0"})
	if delta.Model.Or("") != "gpt-realtime-2.0" || delta.SessionProperties.IsGiven() {
		t.Errorf("delta = %+v, want the model alone", delta)
	}
}

func TestUnknownKeysGoToExtra(t *testing.T) {
	delta := fromMap(t, map[string]any{"unknown_param": 42})
	if delta.Model.IsGiven() || delta.SessionProperties.IsGiven() || delta.Extra["unknown_param"] != 42 {
		t.Errorf("delta = %+v, want the key in extra", delta)
	}
}

func TestMixedKeys(t *testing.T) {
	delta := fromMap(t, map[string]any{"model": "gpt-realtime-2.0", "instructions": "Be helpful.", "unknown": "val"})
	sp, _ := delta.SessionProperties.Value()
	if delta.Model.Or("") != "gpt-realtime-2.0" || sp.Instructions != "Be helpful." || delta.Extra["unknown"] != "val" {
		t.Errorf("delta = %+v, want each key routed", delta)
	}
}

func TestRoundtripFromMappingApplyUpdate(t *testing.T) {
	s := store(events.SessionProperties{}, "")
	delta := fromMap(t, map[string]any{"instructions": "Be concise.", "output_modalities": []any{"text"}})
	changed, err := settings.Apply(s, delta)
	if err != nil || !changed.Has("session_properties") {
		t.Fatalf("changed = %v, err = %v", changed, err)
	}
	sp, _ := s.SessionProperties.Value()
	if sp.Instructions != "Be concise." || sp.OutputModalities[0] != "text" ||
		s.SystemInstruction.Or("") != "Be concise." {
		t.Errorf("settings = %+v, properties = %+v, want the update applied and synced", s, sp)
	}
}

// Ports of the OpenAI binding of tests/test_realtime_service_tool_sync.py. A
// plain list of standard tools is the same Go type as a schema's, so the case
// normalizing one has no counterpart.

func sampleHandler(ctx context.Context, p llm.FunctionCallParams) error {
	return p.Result(ctx, "{}", nil)
}

func sampleTools() frames.ToolsSchema {
	return frames.ToolsSchema{Standard: []frames.Tool{{
		Name: "sample", Description: "A sample tool.", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: llm.FunctionCallHandler(sampleHandler),
	}}}
}

func TestServiceSchemaHandlerRegisters(t *testing.T) {
	s := New(Config{APIKey: "test-key", SessionProperties: events.SessionProperties{Tools: sampleTools()}})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.HasFunction("sample") {
		t.Error("the configured tool's handler was not registered")
	}
}

func TestServiceAsyncToolOptionHonored(t *testing.T) {
	keep := false
	tools := sampleTools()
	tools.Standard[0].CancelOnInterruption = &keep
	s := New(Config{APIKey: "test-key", SessionProperties: events.SessionProperties{Tools: tools}})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.FunctionIsAsync("sample") {
		t.Error("a configured tool not canceled on interruption should register as async")
	}
}

func TestNoServiceToolsIsSafe(t *testing.T) {
	s := New(Config{APIKey: "test-key"})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if s.HasFunction("sample") {
		t.Error("nothing should be registered")
	}
}

// toolNames are the names of the tools a session carries.
func toolNames(sp events.SessionProperties) []string {
	var out []string
	for _, t := range sp.WireTools {
		name, _ := t["name"].(string)
		out = append(out, name)
	}
	return out
}

func TestSendSessionUpdatePreservesServiceTools(t *testing.T) {
	s, sent := offlineService(t, Config{SessionProperties: events.SessionProperties{Tools: sampleTools()}})
	sendAndCapture(t, s, sent)
	// The stored schema survives, so the empty-conversation fallback still finds
	// and registers the handler.
	if len(s.sessionProperties().Tools.Standard) != 1 || s.sessionProperties().WireTools != nil {
		t.Errorf("stored properties = %+v, want the schema untouched", s.sessionProperties())
	}
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.HasFunction("sample") {
		t.Error("the configured tool's handler was not registered")
	}
}

func TestBuiltInToolsJoinTheServiceTools(t *testing.T) {
	s, sent := offlineService(t, Config{SessionProperties: events.SessionProperties{Tools: sampleTools()}})
	s.adapter.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "delegate"}})
	s.convo = frames.NewLLMContext("")
	got := toolNames(sendAndCapture(t, s, sent))
	if len(got) != 2 || got[0] != "sample" || got[1] != "delegate" {
		t.Errorf("tools = %v, want the configured one and the built-in", got)
	}
}

func TestBuiltInToolsAreSentWhenTheServiceHasNone(t *testing.T) {
	s, sent := offlineService(t, Config{})
	s.adapter.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "delegate"}})
	s.convo = frames.NewLLMContext("")
	if got := toolNames(sendAndCapture(t, s, sent)); len(got) != 1 || got[0] != "delegate" {
		t.Errorf("tools = %v, want the built-in alone", got)
	}
}

// Ports of tests/test_realtime_tool_sync.py. jargo has no direct functions: a
// tool carrying its handler is their counterpart.

func TestSetToolsFrameRegistersAndPrunesCarriedHandlers(t *testing.T) {
	s, _ := offlineService(t, Config{})
	ctx := context.Background()
	if err := s.ProcessFrame(ctx, frames.NewLLMSetToolsFrame(sampleTools().Standard), processor.Downstream); err != nil {
		t.Fatalf("set tools: %v", err)
	}
	if !s.HasFunction("sample") {
		t.Fatal("the advertised tool's handler was not registered")
	}
	if err := s.ProcessFrame(ctx, frames.NewLLMSetToolsFrame(nil), processor.Downstream); err != nil {
		t.Fatalf("clear tools: %v", err)
	}
	if s.HasFunction("sample") {
		t.Error("the handler of a tool no longer advertised was kept")
	}
}
