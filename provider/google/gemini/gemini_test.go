package gemini

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/google/internal/wire"
	"github.com/gojargo/jargo/service/llm/adapter"
)

// adapterCapturedWarnings swaps the default logger for one writing into a buffer and
// returns a reader for what was logged, restoring the logger when the test ends.
func adapterCapturedWarnings(t *testing.T) func() string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return buf.String
}

// paramsOf converts a conversation and fails the test if the conversion did.
func paramsOf(t *testing.T, convo *frames.LLMContext, opts adapter.Options) Params {
	t.Helper()
	p, err := (&Adapter{}).LLMInvocationParams(convo, opts)
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	return p
}

// wantRoles fails unless the contents carry exactly the given roles in order.
func wantRoles(t *testing.T, contents []map[string]any, want ...string) {
	t.Helper()
	if len(contents) != len(want) {
		t.Fatalf("contents = %v, want %d of them with roles %v", contents, len(want), want)
	}
	for i := range want {
		if got := contents[i][adapterKeyRole]; got != want[i] {
			t.Errorf("content %d role = %v, want %q", i, got, want[i])
		}
	}
}

// textAt returns the text of a content's single text part.
func textAt(t *testing.T, c map[string]any) string {
	t.Helper()
	parts, ok := c[adapterKeyParts].([]map[string]any)
	if !ok || len(parts) == 0 {
		t.Fatalf("content %v carries no parts", c)
	}
	text, _ := parts[0][adapterKeyText].(string)
	return text
}

// wantPlaceholder is the placeholder signature as it travels, base64-encoded.
const wantPlaceholder = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="

// partsOf returns a content's parts, failing the test when it carries none.
func partsOf(t *testing.T, c map[string]any) []map[string]any {
	t.Helper()
	parts, ok := c[adapterKeyParts].([]map[string]any)
	if !ok || len(parts) == 0 {
		t.Fatalf("content %v carries no parts", c)
	}
	return parts
}

// mustContents converts messages and fails the test if the conversion did.
func mustContents(t *testing.T, msgs []frames.Message) []map[string]any {
	t.Helper()
	got, err := toContents(msgs)
	if err != nil {
		t.Fatalf("ToContents: %v", err)
	}
	return got
}

func TestIDForLLMSpecificMessages(t *testing.T) {
	if got := (&Adapter{}).IDForLLMSpecificMessages(); got != "google" {
		t.Errorf("id = %q, want %q", got, "google")
	}
}

// TestStandardMessagesConverted checks the conversation reaches the request with
// the assistant as the model and the instruction carried beside the contents.
func TestStandardMessagesConverted(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddAssistantMessage("Hi there")

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleUser, roleModel)
	if p.SystemInstruction != "You are helpful." {
		t.Errorf("instruction = %q, want the conversation's prompt", p.SystemInstruction)
	}
}

// TestSystemAndDeveloperMessagesBecomeUser checks that a system or developer
// message said partway through the conversation enters as the user, Gemini
// having neither input role.
func TestSystemAndDeveloperMessagesBecomeUser(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	convo.AddMessage(frames.Message{Role: frames.RoleSystem, Text: "Be concise."})
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "a tool reported late"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleUser, roleUser, roleUser)
	if got := textAt(t, p.Contents[1]); got != "Be concise." {
		t.Errorf("content 1 = %q, want what the system message said", got)
	}
	if got := textAt(t, p.Contents[2]); got != "a tool reported late" {
		t.Errorf("content 2 = %q, want what the developer message carried", got)
	}
}

// TestSystemOnlyConversationBecomesAUserMessage checks a conversation carrying
// nothing but a system prompt is sent as a user message rather than beside an
// empty content list, which Gemini rejects.
func TestSystemOnlyConversationBecomesAUserMessage(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")

	p := paramsOf(t, convo, adapter.Options{})
	if p.SystemInstruction != "" {
		t.Errorf("instruction = %q, want none: it went into the contents", p.SystemInstruction)
	}
	wantRoles(t, p.Contents, roleUser)
	if got := textAt(t, p.Contents[0]); got != "You are helpful." {
		t.Errorf("content = %q, want the prompt sent as a user message", got)
	}
}

// TestSystemInstructionReplacesTheContextPrompt checks that an instruction given
// for one call stands in place of the conversation's own: Gemini has one field
// for it, so the two cannot both be sent.
func TestSystemInstructionReplacesTheContextPrompt(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("Hello")
	warnings := adapterCapturedWarnings(t)

	p := paramsOf(t, convo, adapter.Options{SystemInstruction: "Be concise."})

	if p.SystemInstruction != "Be concise." {
		t.Errorf("instruction = %q, want the call's own", p.SystemInstruction)
	}
	if got := warnings(); !strings.Contains(got, "using the instruction") {
		t.Errorf("warnings = %q, want the conflict reported", got)
	}
}

// TestSystemRepeatedWhenOnlyToolTurns checks a conversation of nothing but tool
// turns is given the instruction again as a user message. Gemini reads the
// instruction as framing rather than as something to answer, so without it the
// model has no prose to reply to.
func TestSystemRepeatedWhenOnlyToolTurns(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "sunny"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleModel, roleUser, roleUser)
	if got := textAt(t, p.Contents[2]); got != "You are helpful." {
		t.Errorf("trailing content = %q, want the instruction said again", got)
	}
	if p.SystemInstruction != "You are helpful." {
		t.Errorf("instruction = %q, want it still carried beside the contents", p.SystemInstruction)
	}
}

// TestSystemNotRepeatedWhenProseIsPresent checks the repeat above is only for a
// conversation with nothing to answer.
func TestSystemNotRepeatedWhenProseIsPresent(t *testing.T) {
	convo := frames.NewLLMContext("You are helpful.")
	convo.AddUserMessage("weather?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "sunny"})

	p := paramsOf(t, convo, adapter.Options{})
	wantRoles(t, p.Contents, roleUser, roleModel, roleUser)
}

// TestEmptyContentsDropped checks a turn that carries nothing is left out:
// Gemini rejects a content with no parts.
func TestEmptyContentsDropped(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, ToolCalls: nil, Text: ""})

	p := paramsOf(t, convo, adapter.Options{})
	for _, c := range p.Contents {
		parts, ok := c[adapterKeyParts].([]map[string]any)
		if !ok || len(parts) == 0 {
			t.Errorf("content %v carries no parts, want it dropped", c)
		}
	}
}

// TestToolResultNamedAfterItsCall checks a result is named after the call it
// answers, which is what an asynchronous tool's late result carries only an id
// for.
func TestToolResultNamedAfterItsCall(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{{ID: "c1", Name: "get_weather"}}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "c1", Content: "sunny"}}},
	}
	b, err := json.Marshal(mustContents(t, msgs))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"name":"get_weather"`) {
		t.Errorf("contents = %s, want the result named after its call", b)
	}
}

// TestUnnamedToolResultGetsAPlaceholder checks a result whose call is nowhere in
// the conversation still carries a name, which Gemini requires.
func TestUnnamedToolResultGetsAPlaceholder(t *testing.T) {
	msgs := []frames.Message{
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "gone", Content: "sunny"}}},
	}
	b, err := json.Marshal(mustContents(t, msgs))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"name":"`+unnamedToolResult+`"`) {
		t.Errorf("contents = %s, want the placeholder name", b)
	}
}

func TestToContentsToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCall(frames.ToolCall{
		ID: "call_0", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`),
	})
	convo.AddToolResult(frames.ToolResult{ID: "call_0", Name: "get_weather", Content: "sunny"})

	b, err := json.Marshal(mustContents(t, convo.Messages()))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	// user turn + tool-result turn use role "user"; the assistant tool-call turn
	// uses role "model". Map keys marshal in alphabetical order.
	wants := []string{
		`"role":"user"`,
		`"role":"model"`,
		`"functionCall":{"args":{"location":"Paris"},"id":"call_0","name":"get_weather"}`,
		`"functionResponse":{"id":"call_0","name":"get_weather","response":{"value":"sunny"}}`,
	}
	for _, want := range wants {
		if !strings.Contains(got, want) {
			t.Errorf("contents missing %s\nin: %s", want, got)
		}
	}
}

func TestFunctionResponseDict(t *testing.T) {
	// A JSON object passes through unchanged.
	raw, ok := wire.FunctionResponseDict(`{"temp":20}`).(json.RawMessage)
	if !ok || string(raw) != `{"temp":20}` {
		t.Errorf("object should pass through, got %v", wire.FunctionResponseDict(`{"temp":20}`))
	}
	// A plain string is wrapped under "value".
	m, ok := wire.FunctionResponseDict("sunny").(map[string]any)
	if !ok || m["value"] != "sunny" {
		t.Errorf("non-object should wrap as {value}, got %v", wire.FunctionResponseDict("sunny"))
	}
}

// Gemini requires an object for a function response, so every result that is
// not one is wrapped, decoded when it is JSON. A function returning nothing
// yields null, which went out bare and was refused.
func TestFunctionResponseDictWrapsEveryNonObject(t *testing.T) {
	cases := map[string]string{
		`null`:                 `{"value":null}`,
		`[1,2]`:                `{"value":[1,2]}`,
		`42`:                   `{"value":42}`,
		`12345678901234567890`: `{"value":12345678901234567890}`,
		`"COMPLETED"`:          `{"value":"COMPLETED"}`,
		`COMPLETED`:            `{"value":"COMPLETED"}`,
		`{"temp":20}`:          `{"temp":20}`,
		`{"a":1} extra`:        `{"value":"{\"a\":1} extra"}`,
	}
	for in, want := range cases {
		b, err := json.Marshal(wire.FunctionResponseDict(in))
		if err != nil {
			t.Fatalf("%s: marshal: %v", in, err)
		}
		if string(b) != want {
			t.Errorf("%s -> %s, want %s", in, b, want)
		}
	}
}

// toolsJSON converts a tool set and decodes the result, for comparing with the
// expected declarations as data rather than as text.
func toolsJSON(t *testing.T, a *Adapter, tools ...frames.Tool) any {
	t.Helper()
	b, err := json.Marshal(a.ToProviderToolsFormat(frames.ToolsSchema{Standard: tools}))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return got
}

// wantJSON decodes the expected declarations.
func wantJSON(t *testing.T, s string) any {
	t.Helper()
	var want any
	if err := json.Unmarshal([]byte(s), &want); err != nil {
		t.Fatalf("expected JSON: %v", err)
	}
	return want
}

// Keys Gemini's validation rejects are dropped wherever they appear in a schema,
// a parameter whose name only looks like one of them excepted.
func TestToProviderToolsFormatStripsKeysGeminiRejects(t *testing.T) {
	adapterCapturedWarnings(t)
	got := toolsJSON(t, &Adapter{}, frames.Tool{
		Name:        "get_file_contents",
		Description: "Get the contents of a file",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"owner":{"type":"string","description":"Repository owner","x-mcp-header":"owner"},
			"x-api-key":{"type":"string","description":"Caller's API key"},
			"options":{"type":"object","additionalProperties":false,"properties":{"x-ref":{"type":"string"}}}
		},"required":["owner"]}`),
	})
	want := wantJSON(t, `[{"functionDeclarations":[{
		"name":"get_file_contents",
		"description":"Get the contents of a file",
		"parameters":{"type":"object","properties":{
			"owner":{"type":"string","description":"Repository owner"},
			"x-api-key":{"type":"string","description":"Caller's API key"},
			"options":{"type":"object","properties":{"x-ref":{"type":"string"}}}
		},"required":["owner"]}
	}]}]`)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v\nwant %v", got, want)
	}
}

// Constructs Gemini cannot express are rewritten: a union type becomes anyOf,
// and an enum of anything but strings is dropped with the type kept.
func TestToProviderToolsFormatTranslatesConstructsGeminiRejects(t *testing.T) {
	adapterCapturedWarnings(t)
	got := toolsJSON(t, &Adapter{}, frames.Tool{
		Name:        "issue_write",
		Description: "Set or clear issue fields",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"value":{"type":["string","number","boolean"],"description":"Value to set"},
			"delete":{"type":"boolean","enum":[true],"description":"Clear the field"},
			"mode":{"type":"string","enum":["create","update"]}
		},"required":["value"]}`),
	})
	want := wantJSON(t, `[{"functionDeclarations":[{
		"name":"issue_write",
		"description":"Set or clear issue fields",
		"parameters":{"type":"object","properties":{
			"value":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"}],"description":"Value to set"},
			"delete":{"type":"boolean","description":"Clear the field"},
			"mode":{"type":"string","enum":["create","update"]}
		},"required":["value"]}
	}]}]`)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v\nwant %v", got, want)
	}
}

// The adaptations are reported, once per tool: the same tools on a later
// inference do not warn again.
func TestToProviderToolsFormatWarnsOncePerAdaptedTool(t *testing.T) {
	logged := adapterCapturedWarnings(t)
	a := &Adapter{}
	tool := frames.Tool{
		Name: "issue_write",
		Parameters: json.RawMessage(
			`{"type":"object","properties":{"value":{"type":["string","number"],"x-mcp-header":"value"}}}`),
	}

	toolsJSON(t, a, tool)
	warned := logged()
	for _, want := range []string{"issue_write", "anyOf", "x-mcp-header"} {
		if !strings.Contains(warned, want) {
			t.Errorf("the warning does not mention %s: %s", want, warned)
		}
	}

	toolsJSON(t, a, tool)
	if again := strings.TrimPrefix(logged(), warned); again != "" {
		t.Errorf("the same tools warned again: %s", again)
	}
}

func TestToProviderToolsFormatStripsAdditionalProperties(t *testing.T) {
	out := (&Adapter{}).ToProviderToolsFormat(frames.ToolsSchema{Standard: []frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{"loc":{"type":"string"}}}`),
	}}})
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	if strings.Contains(got, "additionalProperties") {
		t.Errorf("additionalProperties should be stripped: %s", got)
	}
	if !strings.Contains(got, `"functionDeclarations"`) || !strings.Contains(got, `"name":"get_weather"`) {
		t.Errorf("declaration shape wrong: %s", got)
	}
}

// TestUnsignedToolCallCarriesThePlaceholderSignature covers the request shape
// the newer models validate: a model turn whose function calls carry no thought
// signature is refused. A call another provider made, or one the application
// wrote itself, has none, so it travels under the placeholder Google documents
// for exactly that.
func TestUnsignedToolCallCarriesThePlaceholderSignature(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("switch to Gemini")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "switch_llm"})

	parts := partsOf(t, mustContents(t, convo.Messages())[1])
	if got := parts[0][keyThoughtSignature]; got != wantPlaceholder {
		t.Errorf("thoughtSignature = %v, want the placeholder %q", got, wantPlaceholder)
	}
}

// TestASignedTurnIsLeftAsItStands covers a batch of parallel calls the model
// made itself: it signs the first and expects the rest unsigned, so a turn
// holding a signature is not one to add placeholders to.
func TestASignedTurnIsLeftAsItStands(t *testing.T) {
	contents := []map[string]any{{
		adapterKeyRole: roleModel,
		adapterKeyParts: []map[string]any{
			{keyFunctionCall: map[string]any{adapterKeyID: "c1"}, keyThoughtSignature: "sig-c1"},
			{keyFunctionCall: map[string]any{adapterKeyID: "c2"}},
		},
	}}
	addPlaceholderThoughtSignatures(contents)

	parts := partsOf(t, contents[0])
	if got := parts[0][keyThoughtSignature]; got != "sig-c1" {
		t.Errorf("the model's own signature = %v, want it untouched", got)
	}
	if got, ok := parts[1][keyThoughtSignature]; ok {
		t.Errorf("the second call of the batch carries %v, want it left unsigned", got)
	}
}

// TestThePlaceholderSkipsTextAndUserTurns checks it reaches function calls and
// nothing else: a signature on a text part or on a user turn is not a shape the
// API takes.
func TestThePlaceholderSkipsTextAndUserTurns(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("hi there")
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: "sunny"})

	for i, c := range mustContents(t, convo.Messages()) {
		for j, part := range partsOf(t, c) {
			if got, ok := part[keyThoughtSignature]; ok {
				t.Errorf("content %d part %d carries signature %v, want none", i, j, got)
			}
		}
	}
}

// TestTrailingModelTurnTakesANoOpUserTurn covers the request a model that
// cannot continue a trailing model turn refuses. The conversation ends on an
// assistant message whenever the model spoke last, so the shape is satisfied
// with a full stop that says nothing.
func TestTrailingModelTurnTakesANoOpUserTurn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("let me check on that")

	contents := paramsOf(t, convo, adapter.Options{EnsureLastMessageIsUser: true}).Contents
	wantRoles(t, contents, roleUser, roleModel, roleUser)
	if got := textAt(t, contents[2]); got != noOpUserTurn {
		t.Errorf("appended text = %q, want %q", got, noOpUserTurn)
	}
}

// TestFillerAfterAToolResultTakesANoOpUserTurn is the case that made this
// necessary: a tool call settles where it was made, and filler spoken while the
// tool ran leaves the model's own words last in the conversation.
func TestFillerAfterAToolResultTakesANoOpUserTurn(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("what's the weather?")
	convo.AddAssistantToolCall(frames.ToolCall{ID: "c1", Name: "get_weather"})
	convo.AddToolResult(frames.ToolResult{ID: "c1", Name: "get_weather", Content: `{"temperature":"75"}`})
	convo.AddAssistantMessage("let me check on that")

	contents := paramsOf(t, convo, adapter.Options{EnsureLastMessageIsUser: true}).Contents
	wantRoles(t, contents, roleUser, roleModel, roleUser, roleModel, roleUser)
	if got := textAt(t, contents[4]); got != noOpUserTurn {
		t.Errorf("appended text = %q, want %q", got, noOpUserTurn)
	}
}

// TestTrailingModelTurnIsKeptWhenNothingAsks checks the conversation is left
// alone for a model that does continue one, which is what the option is for.
func TestTrailingModelTurnIsKeptWhenNothingAsks(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("hi there")

	wantRoles(t, paramsOf(t, convo, adapter.Options{}).Contents, roleUser, roleModel)
}

// TestATrailingUserTurnIsNotDoubled checks nothing is appended to a
// conversation already ending the way the API wants it.
func TestATrailingUserTurnIsNotDoubled(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hello")
	convo.AddAssistantMessage("hi there")
	convo.AddUserMessage("how are you?")

	contents := paramsOf(t, convo, adapter.Options{EnsureLastMessageIsUser: true}).Contents
	wantRoles(t, contents, roleUser, roleModel, roleUser)
	if got := textAt(t, contents[2]); got != "how are you?" {
		t.Errorf("last content = %q, want the user's own words", got)
	}
}
