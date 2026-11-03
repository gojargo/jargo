package live

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

func sampleHandler(ctx context.Context, p llm.FunctionCallParams) error {
	return p.Result(ctx, "{}", nil)
}

// sampleTool advertises a tool named name carrying a handler.
func sampleTool(name string) frames.Tool {
	return frames.Tool{
		Name: name, Description: "A sample tool.", Parameters: json.RawMessage(`{"type":"object","properties":{}}`),
		Handler: llm.FunctionCallHandler(sampleHandler),
	}
}

func withTools(tools frames.ToolsSchema) *Service {
	return New(Config{APIKey: "test-key", Tools: tools})
}

// The ports of the upstream service-tool sync cases, bound to Gemini Live.

func TestServiceSchemaHandlerRegisters(t *testing.T) {
	s := withTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("sample")}})
	// An empty conversation falls back to the service tools.
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.HasFunction("sample") {
		t.Fatal("the configured tool's handler was not registered")
	}
}

func TestServiceAsyncToolOptionHonored(t *testing.T) {
	keep := false
	tool := sampleTool("sample")
	tool.CancelOnInterruption = &keep
	s := withTools(frames.ToolsSchema{Standard: []frames.Tool{tool}})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if !s.FunctionIsAsync("sample") {
		t.Fatal("a configured tool not canceled on interruption should register as async")
	}
}

func TestNoServiceToolsIsSafe(t *testing.T) {
	s := withTools(frames.ToolsSchema{})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if s.HasFunction("sample") {
		t.Fatal("nothing should be registered")
	}
}

// Gemini accepts tools in its own format, which carry no handler.
func TestRawServiceToolsRegisterNothing(t *testing.T) {
	s := withTools(frames.ToolsSchema{Custom: map[frames.AdapterType][]any{
		frames.AdapterTypeGemini: {map[string]any{"function_declarations": []any{}}},
	}})
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{})
	if s.HasFunction("function_declarations") {
		t.Fatal("tools in Gemini's format should register nothing")
	}
}

func TestContextToolsTakePrecedence(t *testing.T) {
	s := withTools(frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_init")}})
	// When the conversation advertises tools, the service tools aren't used.
	s.SyncToolHandlers(context.Background(), frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_context")}})
	if !s.HasFunction("from_context") {
		t.Error("the conversation's tool was not registered")
	}
	if s.HasFunction("from_init") {
		t.Error("the configured tool should not be used while the conversation advertises tools")
	}
}

// The tests below are not ports of upstream tests: they cover how the session
// settles its instruction and tools, which upstream does in _connect and
// _handle_context.

// declaredNames are the function names a setup body declares. It reads the
// body as it travels, so one built in memory and one received read the same.
func declaredNames(setup map[string]any) []string {
	raw, _ := json.Marshal(setup)
	var body struct {
		Tools []struct {
			FunctionDeclarations []struct {
				Name string `json:"name"`
			} `json:"functionDeclarations"` //nolint:tagliatelle // the Live API's field name
		} `json:"tools"`
	}
	_ = json.Unmarshal(raw, &body)
	var names []string
	for _, group := range body.Tools {
		for _, d := range group.FunctionDeclarations {
			names = append(names, d.Name)
		}
	}
	return names
}

// Before any conversation, the session opens with the configured tools and the
// built-in ones, and the configured instruction.
func TestASessionOpensWithTheConfiguredToolsAndInstruction(t *testing.T) {
	s := New(Config{
		APIKey: "k", Instructions: "be brief",
		Tools: frames.ToolsSchema{Standard: []frames.Tool{sampleTool("sample")}},
	})
	s.adapter.SetBuiltin(adapter.Builtin{Tool: frames.Tool{Name: "delegate"}})

	setup := setupOf(t, s)
	if got := declaredNames(setup); len(got) != 2 || got[0] != "sample" || got[1] != "delegate" {
		t.Errorf("tools = %v, want the configured one and the built-in", got)
	}
	system, _ := setup["systemInstruction"].(map[string]any)
	if got := systemText(system); got != "be brief" {
		t.Errorf("system instruction = %v, want the configured one", system)
	}
}

// systemText is the text of a system instruction, however it was decoded.
func systemText(system map[string]any) string {
	raw, _ := json.Marshal(system)
	var body struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	}
	_ = json.Unmarshal(raw, &body)
	if len(body.Parts) != 1 {
		return ""
	}
	return body.Parts[0].Text
}

// setups are the setup bodies the session received, in order.
func (r *recordingSession) setups() []map[string]any {
	return r.withKey("setup")
}

// waitSetups waits until the session has received n setup messages.
func waitSetups(t *testing.T, srv *recordingSession, n int) []map[string]any {
	t.Helper()
	for deadline := time.Now().Add(3 * time.Second); ; {
		if got := srv.setups(); len(got) >= n {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("received %d setups, want %d", len(srv.setups()), n)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// A first conversation bringing tools of its own reopens the session with
// them, in place of the configured ones.
func TestAConversationsToolsReplaceTheConfiguredOnes(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{
		APIKey: "k", BaseURL: wsURL(srv),
		Tools: frames.ToolsSchema{Standard: []frames.Tool{sampleTool("from_init")}},
	}))
	waitSetups(t, srv, 1)

	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{sampleTool("from_context")})
	s.handleContext(t.Context(), convo, 0)

	setups := waitSetups(t, srv, 2)
	if got := declaredNames(setups[1]); len(got) != 1 || got[0] != "from_context" {
		t.Errorf("tools = %v, want the conversation's alone", got)
	}
}

// A first conversation whose own prompt becomes the effective instruction,
// there being none configured, reopens the session with it.
func TestAConversationsPromptReopensWithoutAConfiguredInstruction(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv)}))
	waitSetups(t, srv, 1)

	convo := frames.NewLLMContext("be terse")
	convo.AddUserMessage("hi")
	s.handleContext(t.Context(), convo, 0)

	setups := waitSetups(t, srv, 2)
	system, _ := setups[1]["systemInstruction"].(map[string]any)
	if got := systemText(system); got != "be terse" {
		t.Errorf("system instruction = %v, want the conversation's prompt", system)
	}
}

// With an instruction configured, the conversation's prompt does not change
// the effective instruction, so the session is not reopened for it.
func TestAConfiguredInstructionKeepsTheSession(t *testing.T) {
	srv := newRecordingSession(t)
	s := startedServiceBuilt(t, New(Config{APIKey: "k", BaseURL: wsURL(srv), Instructions: "be brief"}))
	waitSetups(t, srv, 1)

	convo := frames.NewLLMContext("be terse")
	convo.AddUserMessage("hi")
	s.handleContext(t.Context(), convo, 0)
	settle()

	if got := srv.setups(); len(got) != 1 {
		t.Errorf("received %d setups, want the session left as it was", len(got))
	}
}
