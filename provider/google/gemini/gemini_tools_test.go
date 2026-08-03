package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// fakeSink collects the text and tool calls reported during a tool-capable
// generation.
type fakeSink struct {
	text  strings.Builder
	calls []frames.ToolCall
}

func (f *fakeSink) Text(t string) error          { f.text.WriteString(t); return nil }
func (f *fakeSink) Tool(c frames.ToolCall) error { f.calls = append(f.calls, c); return nil }

func TestToContentsToolTurn(t *testing.T) {
	convo := frames.NewLLMContext("be helpful")
	convo.AddUserMessage("weather in Paris?")
	convo.AddAssistantToolCalls("", []frames.ToolCall{
		{ID: "call_0", Name: "get_weather", Args: json.RawMessage(`{"location":"Paris"}`)},
	})
	convo.AddToolResults([]frames.ToolResult{
		{ID: "call_0", Name: "get_weather", Content: "sunny"},
	})

	b, err := json.Marshal(toContents(convo))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	// user turn + tool-result turn use role "user"; the assistant tool-call turn
	// uses role "model". Map keys marshal in alphabetical order.
	wants := []string{
		`"role":"user"`,
		`"role":"model"`,
		`"functionCall":{"args":{"location":"Paris"},"name":"get_weather"}`,
		`"functionResponse":{"name":"get_weather","response":{"value":"sunny"}}`,
	}
	for _, want := range wants {
		if !strings.Contains(got, want) {
			t.Errorf("contents missing %s\nin: %s", want, got)
		}
	}
}

func TestFunctionResponseDict(t *testing.T) {
	// A JSON object passes through unchanged.
	raw, ok := functionResponseDict(`{"temp":20}`).(json.RawMessage)
	if !ok || string(raw) != `{"temp":20}` {
		t.Errorf("object should pass through, got %v", functionResponseDict(`{"temp":20}`))
	}
	// A plain string is wrapped under "value".
	m, ok := functionResponseDict("sunny").(map[string]any)
	if !ok || m["value"] != "sunny" {
		t.Errorf("non-object should wrap as {value}, got %v", functionResponseDict("sunny"))
	}
}

func TestToToolsStripsAdditionalProperties(t *testing.T) {
	out := toTools([]frames.Tool{{
		Name:        "get_weather",
		Description: "Look up the weather",
		Parameters:  json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{"loc":{"type":"string"}}}`),
	}})
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

func TestGeminiToolStreamParsesParts(t *testing.T) {
	data := `{"candidates":[{"content":{"parts":[` +
		`{"text":"Checking. "},` +
		`{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}}` +
		`]}}]}`
	var chunk genChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	sink := &fakeSink{}
	ts := &geminiToolStream{sink: sink}
	if err := ts.consume(chunk); err != nil {
		t.Fatalf("consume: %v", err)
	}

	if sink.text.String() != "Checking. " {
		t.Errorf("text = %q", sink.text.String())
	}
	if len(sink.calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(sink.calls))
	}
	c := sink.calls[0]
	if c.Name != "get_weather" || string(c.Args) != `{"location":"Paris"}` {
		t.Errorf("call wrong: %+v (args %s)", c, c.Args)
	}
	if c.ID == "" {
		t.Error("expected a synthetic call id")
	}
}

// Safety filters are sent alongside the request rather than inside the
// generation config, which is where the API expects them.
func TestRequestBodyCarriesSafetySettings(t *testing.T) {
	safety := []SafetySetting{
		{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_LOW_AND_ABOVE"},
		{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_NONE", Method: "SEVERITY"},
	}
	s := NewLLM(Config{APIKey: "k", SafetySettings: safety})

	b, err := json.Marshal(s.requestBody(frames.NewLLMContext(""), false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(b)
	wants := []string{
		`"safetySettings":[`,
		`{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}`,
		`{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_NONE","method":"SEVERITY"}`,
	}
	for _, want := range wants {
		if !strings.Contains(got, want) {
			t.Errorf("request body is missing %s:\n%s", want, got)
		}
	}
}

// Configuring no filters sends none, leaving every category at the API default
// rather than pinning it to an empty list.
func TestRequestBodyOmitsSafetySettingsByDefault(t *testing.T) {
	s := NewLLM(Config{APIKey: "k"})

	b, err := json.Marshal(s.requestBody(frames.NewLLMContext(""), false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "safetySettings") {
		t.Errorf("request body carries safetySettings with none configured:\n%s", b)
	}
}

// A safety filter is checked before anything tries to use it, since the API
// rejects an unknown category or threshold outright.
func TestSafetySettingValidation(t *testing.T) {
	const harassment = "HARM_CATEGORY_HARASSMENT"
	cases := map[string]struct {
		setting SafetySetting
		wantOK  bool
	}{
		"category and threshold":  {SafetySetting{Category: harassment, Threshold: "BLOCK_ONLY_HIGH"}, true},
		"with a method":           {SafetySetting{Category: harassment, Threshold: "OFF", Method: "PROBABILITY"}, true},
		"unknown category":        {SafetySetting{Category: "HARM_CATEGORY_NONSENSE", Threshold: "BLOCK_ONLY_HIGH"}, false},
		"unknown threshold":       {SafetySetting{Category: harassment, Threshold: "BLOCK_SOMETIMES"}, false},
		"threshold without cause": {SafetySetting{Threshold: "BLOCK_ONLY_HIGH"}, false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := Config{APIKey: "k", SafetySettings: []SafetySetting{tc.setting}}.Validate()
			if tc.wantOK && err != nil {
				t.Errorf("Validate: %v", err)
			}
			if !tc.wantOK && err == nil {
				t.Error("Validate accepted a setting the API would reject")
			}
		})
	}
}
