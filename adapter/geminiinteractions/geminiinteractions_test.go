package geminiinteractions

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

//nolint:gochecknoglobals // test fixtures
var (
	signatureBytes = []byte("\x12\x34signature")
	signatureB64   = base64.StdEncoding.EncodeToString(signatureBytes)
	placeholderB64 = base64.StdEncoding.EncodeToString([]byte("skip_thought_signature_validator"))
	weatherTool    = frames.Tool{
		Name:        "get_weather",
		Description: "Get the weather for a city.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
	weatherToolsOnly = frames.ToolsSchema{Standard: []frames.Tool{weatherTool}}
)

func user(text string) frames.Message { return frames.Message{Role: frames.RoleUser, Text: text} }

func assistant(text string) frames.Message {
	return frames.Message{Role: frames.RoleAssistant, Text: text}
}

func toolCall(callID, city string) frames.Message {
	return frames.Message{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{{
		ID: callID, Name: "get_weather", Args: json.RawMessage(`{"city":"` + city + `"}`),
	}}}
}

func toolResult(callID, content string) frames.Message {
	return frames.Message{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: callID, Content: content}}}
}

// signature is a thought signature stored the way the Gemini services store
// them.
func signature(bookmark map[string]any, value any) frames.Message {
	if value == nil {
		value = signatureBytes
	}
	return frames.NewLLMSpecificMessage("google", map[string]any{
		"type": "thought_signature", "signature": value, "bookmark": bookmark,
	})
}

func contextOf(messages ...frames.Message) *frames.LLMContext {
	convo := frames.NewLLMContext("")
	for _, m := range messages {
		convo.AddMessage(m)
	}
	return convo
}

func params(t *testing.T, a *Adapter, convo *frames.LLMContext, opts adapter.Options) Params {
	t.Helper()
	p, err := a.LLMInvocationParams(convo, opts)
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	return p
}

func inputOf(t *testing.T, messages ...frames.Message) []map[string]any {
	t.Helper()
	return params(t, &Adapter{}, contextOf(messages...), adapter.Options{}).Input
}

// assertJSON compares values by their JSON form.
func assertJSON(t *testing.T, got, want any) {
	t.Helper()
	g, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	w, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var gv, wv any
	_ = json.Unmarshal(g, &gv)
	_ = json.Unmarshal(w, &wv)
	if !reflect.DeepEqual(gv, wv) {
		t.Errorf("got  %s\nwant %s", g, w)
	}
}

func types(steps []map[string]any) []string {
	out := make([]string, 0, len(steps))
	for _, s := range steps {
		out = append(out, stepType(s))
	}
	return out
}

// typeAndIDs lists each step's type with its call id, if it has one.
func typeAndIDs(steps []map[string]any) [][2]string {
	out := make([][2]string, 0, len(steps))
	for _, s := range steps {
		id, _ := s[keyID].(string)
		if id == "" {
			id, _ = s[keyCallID].(string)
		}
		out = append(out, [2]string{stepType(s), id})
	}
	return out
}

func textItem(text string) map[string]any { return map[string]any{"type": "text", "text": text} }

// TestTextConversation checks user and assistant text becoming user_input and
// model_output steps.
func TestTextConversation(t *testing.T) {
	p := params(t, &Adapter{}, contextOf(user("Name a color."), assistant("Blue."), user("Another?")), adapter.Options{})
	if p.SystemInstruction != "" {
		t.Errorf("SystemInstruction = %q, want none", p.SystemInstruction)
	}
	assertJSON(t, p.Input, []map[string]any{
		{"type": "user_input", "content": []any{textItem("Name a color.")}},
		{"type": "model_output", "content": []any{textItem("Blue.")}},
		{"type": "user_input", "content": []any{textItem("Another?")}},
	})
}

// TestSystemInstructionSetting checks a service's system instruction applies
// when the conversation has none.
func TestSystemInstructionSetting(t *testing.T) {
	p := params(t, &Adapter{}, contextOf(user("Hi")), adapter.Options{SystemInstruction: "Be kind."})
	if p.SystemInstruction != "Be kind." {
		t.Errorf("SystemInstruction = %q, want the service's", p.SystemInstruction)
	}
}

// TestInputEndsOnTheUsersSide checks a conversation ending with an assistant
// message gains a trailing user turn.
func TestInputEndsOnTheUsersSide(t *testing.T) {
	input := inputOf(t, user("Count to three."), assistant("One, two,"))
	assertJSON(t, input[len(input)-1], map[string]any{"type": "user_input", "content": []any{textItem(".")}})
}

// TestSignedFunctionCall checks a stored signature becomes a thought step ahead
// of its call.
func TestSignedFunctionCall(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris?"),
		toolCall("call_1", "Paris"),
		signature(map[string]any{"function_call": "call_1"}, nil),
		toolResult("call_1", `{"temperature_f": 72}`),
	)
	assertJSON(t, input[1:], []map[string]any{
		{"type": "thought", "signature": signatureB64},
		{"type": "function_call", "id": "call_1", "name": "get_weather", "arguments": map[string]any{"city": "Paris"}},
		{
			"type": "function_result", "call_id": "call_1", "name": "get_weather",
			"result": map[string]any{"temperature_f": 72},
		},
	})
}

// TestStringSignaturePassesThrough checks a signature already in base64 text
// form is sent as it is.
func TestStringSignaturePassesThrough(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris?"),
		toolCall("call_1", "Paris"),
		signature(map[string]any{"function_call": "call_1"}, "already-base64"),
		toolResult("call_1", `{"temperature_f": 72}`),
	)
	assertJSON(t, input[1], map[string]any{"type": "thought", "signature": "already-base64"})
}

// TestUnsignedFunctionCallGetsPlaceholder checks a call with no signature, as
// from another LLM, gets the placeholder.
func TestUnsignedFunctionCallGetsPlaceholder(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris?"), toolCall("call_1", "Paris"), toolResult("call_1", `{"temperature_f": 72}`))
	assertJSON(t, input[1], map[string]any{"type": "thought", "signature": placeholderB64})
	if stepType(input[2]) != "function_call" {
		t.Errorf("step 2 = %v, want the call", input[2])
	}
}

// TestParallelCallsShareOneThought checks parallel calls signed on the first
// call become one thought, then the calls.
func TestParallelCallsShareOneThought(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris and Tokyo?"),
		toolCall("call_1", "Paris"),
		toolCall("call_2", "Tokyo"),
		signature(map[string]any{"function_call": "call_1"}, nil),
		toolResult("call_1", `{"temperature_f": 72}`),
		toolResult("call_2", `{"temperature_f": 60}`),
	)
	want := [][2]string{
		{"thought", ""},
		{"function_call", "call_1"},
		{"function_call", "call_2"},
		{"function_result", "call_1"},
		{"function_result", "call_2"},
	}
	if got := typeAndIDs(input[1:]); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
}

// TestInterleavedParallelCallsRegrouped checks a batch stored call-then-result
// per call is sent as calls, then results.
func TestInterleavedParallelCallsRegrouped(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris and Tokyo?"),
		toolCall("call_1", "Paris"),
		toolResult("call_1", `{"temperature_f": 72}`),
		toolCall("call_2", "Tokyo"),
		toolResult("call_2", `{"temperature_f": 60}`),
		signature(map[string]any{"function_call": "call_1"}, nil),
		user("And Rome?"),
		toolCall("call_3", "Rome"),
		signature(map[string]any{"function_call": "call_3"}, nil),
		toolResult("call_3", `{"temperature_f": 80}`),
	)
	want := [][2]string{
		{"user_input", ""},
		{"thought", ""},
		{"function_call", "call_1"},
		{"function_call", "call_2"},
		{"function_result", "call_1"},
		{"function_result", "call_2"},
		{"user_input", ""},
		{"thought", ""},
		{"function_call", "call_3"},
		{"function_result", "call_3"},
	}
	if got := typeAndIDs(input); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
}

// TestSequentialSignedCallsStayApart checks a call signed on its own starts a
// new turn rather than joining the last batch.
func TestSequentialSignedCallsStayApart(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris, then Tokyo?"),
		toolCall("call_1", "Paris"),
		toolResult("call_1", `{"temperature_f": 72}`),
		toolCall("call_2", "Tokyo"),
		toolResult("call_2", `{"temperature_f": 60}`),
		signature(map[string]any{"function_call": "call_1"}, nil),
		signature(map[string]any{"function_call": "call_2"}, nil),
	)
	want := []string{
		"user_input", "thought", "function_call", "function_result", "thought", "function_call", "function_result",
	}
	if got := types(input); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
}

// TestModelTurnStartsWithItsThought checks text stored apart from its signed
// calls still opens the turn with the thought.
func TestModelTurnStartsWithItsThought(t *testing.T) {
	input := inputOf(t,
		user("Weather in Paris?"),
		assistant("Let me check."),
		toolCall("call_1", "Paris"),
		signature(map[string]any{"function_call": "call_1"}, nil),
		toolResult("call_1", `{"temperature_f": 72}`),
	)
	want := []string{"user_input", "thought", "model_output", "function_call", "function_result"}
	if got := types(input); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
}

// TestSignedTextTurn checks a text reply's signature becomes a thought step
// ahead of the text.
func TestSignedTextTurn(t *testing.T) {
	input := inputOf(t,
		user("Hi"),
		assistant("Hello!"),
		signature(map[string]any{"text": "Hello!"}, nil),
		user("Name a color."),
	)
	want := []string{"user_input", "thought", "model_output", "user_input"}
	if got := types(input); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
}

// TestImageInput checks an inline image becomes an image content item with
// base64 data.
func TestImageInput(t *testing.T) {
	data := base64.StdEncoding.EncodeToString([]byte("\x89PNG fake"))
	input := inputOf(t, frames.Message{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartText, Text: "What is this?"},
		{Type: frames.ContentPartImageURL, ImageURL: "data:image/png;base64," + data},
	}})
	assertJSON(t, input, []map[string]any{{
		"type": "user_input",
		"content": []any{
			textItem("What is this?"),
			map[string]any{"type": "image", "mime_type": "image/png", "data": data},
		},
	}})
}

// TestImageURLInput checks an image given by URL is sent by its URI.
func TestImageURLInput(t *testing.T) {
	url := "https://example.com/cat.jpg"
	input := inputOf(t, frames.Message{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartImageURL, ImageURL: url},
	}})
	assertJSON(t, input[0]["content"], []any{map[string]any{"type": "image", "uri": url}})
}

// TestNonImageMediaTypes checks a data URL of another kind of media gets its own
// content type.
func TestNonImageMediaTypes(t *testing.T) {
	for _, tc := range []struct{ mime, want string }{
		{"audio/wav", "audio"}, {"video/mp4", "video"}, {"application/pdf", "document"},
	} {
		item := dataURLItem("data:"+tc.mime+";base64,AAAA", "")
		if item["type"] != tc.want || item["mime_type"] != tc.mime {
			t.Errorf("%s: item = %v, want type %s", tc.mime, item, tc.want)
		}
	}
}

// TestTools checks standard tools become flat function tools.
func TestTools(t *testing.T) {
	convo := contextOf(user("Hi"))
	convo.SetToolsSchema(weatherToolsOnly)
	p := params(t, &Adapter{}, convo, adapter.Options{})
	assertJSON(t, p.Tools, []map[string]any{{
		"type":        "function",
		"name":        "get_weather",
		"description": "Get the weather for a city.",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []any{"city"},
		},
	}})
}

// TestBuiltinToolsIncluded checks tools a service injects into the adapter are
// sent with the conversation's.
func TestBuiltinToolsIncluded(t *testing.T) {
	a := &Adapter{}
	a.SetBuiltin(adapter.Builtin{Tool: weatherTool})
	p := params(t, a, contextOf(user("Hi")), adapter.Options{})
	if len(p.Tools) != 1 || p.Tools[0]["name"] != "get_weather" {
		t.Errorf("tools = %v, want the built-in tool", p.Tools)
	}
}

// TestSystemInstructionSentOnce checks a conversation with no plain-text message
// does not repeat the system instruction.
func TestSystemInstructionSentOnce(t *testing.T) {
	url := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("png"))
	convo := contextOf(frames.Message{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartImageURL, ImageURL: url},
	}})
	p := params(t, &Adapter{}, convo, adapter.Options{SystemInstruction: "Be kind."})
	if p.SystemInstruction != "Be kind." {
		t.Errorf("SystemInstruction = %q, want the service's", p.SystemInstruction)
	}
	if got := types(p.Input); !reflect.DeepEqual(got, []string{"user_input"}) {
		t.Errorf("steps = %v, want the image alone", got)
	}
}

// TestOnePlaceholderPerUnsignedTurn checks unsigned parallel calls open their
// turn with a single placeholder.
func TestOnePlaceholderPerUnsignedTurn(t *testing.T) {
	input := inputOf(t,
		user("Weather in Rome and Oslo?"),
		toolCall("call_1", "Rome"),
		toolCall("call_2", "Oslo"),
		toolResult("call_1", `{"temperature_f": 70}`),
		toolResult("call_2", `{"temperature_f": 50}`),
	)
	want := []string{"user_input", "thought", "function_call", "function_call", "function_result", "function_result"}
	if got := types(input); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
	if input[1]["signature"] != placeholderB64 {
		t.Errorf("signature = %v, want the placeholder", input[1]["signature"])
	}
}

// TestAssistantTextAndCallsInOneMessage checks an assistant message carrying
// both text and calls keeps both.
func TestAssistantTextAndCallsInOneMessage(t *testing.T) {
	call := toolCall("call_1", "Paris")
	call.Text = "Let me check."
	input := inputOf(t,
		user("Weather in Paris?"),
		call,
		signature(map[string]any{"function_call": "call_1"}, nil),
		toolResult("call_1", `{"temperature_f": 72}`),
	)
	want := []string{"user_input", "thought", "model_output", "function_call", "function_result"}
	if got := types(input); !reflect.DeepEqual(got, want) {
		t.Errorf("steps = %v, want %v", got, want)
	}
}

// TestTextFunctionResult checks a tool result that is not a JSON object is sent
// as text.
func TestTextFunctionResult(t *testing.T) {
	input := inputOf(t, user("Weather in Paris?"), toolCall("call_1", "Paris"), toolResult("call_1", "COMPLETED"))
	last := input[len(input)-1]
	if last["result"] != "COMPLETED" {
		t.Errorf("result = %v, want the text", last["result"])
	}
	if last["name"] != "get_weather" {
		t.Errorf("name = %v, want the call's", last["name"])
	}
}

// TestSpecificStepPassesThrough checks an Interactions step stored as a
// Google-specific message is sent as it is.
func TestSpecificStepPassesThrough(t *testing.T) {
	step := map[string]any{"type": "user_input", "content": []any{textItem("Hi")}}
	input := inputOf(t, frames.NewLLMSpecificMessage("google", step))
	assertJSON(t, input, []map[string]any{step})
}

// capturedWarnings routes the default logger to a buffer for the test.
func capturedWarnings(t *testing.T) func() string {
	t.Helper()
	var mu sync.Mutex
	var buf bytes.Buffer
	prev := slog.Default()
	handler := slog.NewTextHandler(&lockedWriter{mu: &mu, w: &buf}, &slog.HandlerOptions{Level: slog.LevelWarn})
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return func() string {
		mu.Lock()
		defer mu.Unlock()
		return buf.String()
	}
}

type lockedWriter struct {
	mu *sync.Mutex
	w  *bytes.Buffer
}

func (l *lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

// TestUnsupportedSpecificMessageSkipped checks a Google-specific message that is
// not a step is left out, with a warning.
func TestUnsupportedSpecificMessageSkipped(t *testing.T) {
	logged := capturedWarnings(t)
	input := inputOf(t,
		frames.NewLLMSpecificMessage("google", map[string]any{"role": "user", "parts": []any{}}),
		user("Hi"),
	)
	if got := types(input); !reflect.DeepEqual(got, []string{"user_input"}) {
		t.Errorf("steps = %v, want the user's message alone", got)
	}
	if !strings.Contains(logged(), "Skipping Google-specific") {
		t.Errorf("logs = %q, want the skipped message reported", logged())
	}
}

// TestCustomGeminiTools checks built-in Gemini tools keyed by name become typed
// tools.
func TestCustomGeminiTools(t *testing.T) {
	schema := frames.ToolsSchema{Custom: map[frames.AdapterType][]any{
		frames.AdapterTypeGemini: {
			map[string]any{"google_search": map[string]any{}},
			map[string]any{"url_context": map[string]any{}},
		},
	}}
	assertJSON(t, (&Adapter{}).ToProviderToolsFormat(schema), []map[string]any{
		{"type": "google_search"}, {"type": "url_context"},
	})
}

// TestTypedToolsPassThrough checks tools already in the Interactions format pass
// through unchanged.
func TestTypedToolsPassThrough(t *testing.T) {
	tool := map[string]any{"type": "google_search", "search_types": []any{"web_search"}}
	assertJSON(t, ToInteractionsTools([]map[string]any{tool}), []map[string]any{tool})
}

// TestFunctionDeclarations checks each declaration of a function_declarations
// entry becomes its own tool.
func TestFunctionDeclarations(t *testing.T) {
	tools := []map[string]any{{"function_declarations": []any{
		map[string]any{"name": "a"}, map[string]any{"name": "b"},
	}}}
	assertJSON(t, ToInteractionsTools(tools), []map[string]any{
		{"type": "function", "name": "a"}, {"type": "function", "name": "b"},
	})
}

// TestMediaAndSignaturesRedacted checks the logging view hides media data and
// thought signatures, and leaves the conversation untouched.
func TestMediaAndSignaturesRedacted(t *testing.T) {
	url := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("png"))
	convo := contextOf(
		frames.Message{Role: frames.RoleUser, Content: []frames.ContentPart{
			{Type: frames.ContentPartText, Text: "Look"},
			{Type: frames.ContentPartImageURL, ImageURL: url},
		}},
		toolCall("call_1", "Paris"),
		signature(map[string]any{"function_call": "call_1"}, nil),
		toolResult("call_1", `{"temperature_f": 72}`),
	)

	steps := (&Adapter{}).MessagesForLogging(convo)

	if data := contentItems(steps[0])[1]["data"]; data != "..." {
		t.Errorf("image data = %v, want it elided", data)
	}
	assertJSON(t, steps[1], map[string]any{"type": "thought", "signature": "..."})
	if got := convo.Messages()[0].Content[1].ImageURL; got != url {
		t.Errorf("the conversation's image became %q", got)
	}
}
