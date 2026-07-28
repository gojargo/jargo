package realtime

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestSessionUpdateShape checks the session configuration xAI expects: the voice,
// the nested input/output audio format, and server-side turn detection. The
// model is deliberately absent, since xAI selects it on the handshake.
func TestSessionUpdateShape(t *testing.T) {
	s := New(Config{APIKey: "k", Instructions: "be brief"})
	session := s.sessionUpdate()

	if session.Type != "session.update" {
		t.Errorf("type = %q, want session.update", session.Type)
	}
	if got := session.Session["voice"]; got != defaultVoice {
		t.Errorf("voice = %v, want %q", got, defaultVoice)
	}
	if got := session.Session["instructions"]; got != "be brief" {
		t.Errorf("instructions = %v, want the configured prompt", got)
	}
	if _, ok := session.Session["model"]; ok {
		t.Error("session carries a model, but xAI selects it on the handshake")
	}

	turn, ok := session.Session["turn_detection"].(map[string]any)
	if !ok || turn[keyType] != "server_vad" {
		t.Errorf("turn_detection = %v, want server_vad by default", session.Session["turn_detection"])
	}

	audio, ok := session.Session["audio"].(map[string]any)
	if !ok {
		t.Fatalf("audio = %v, want the nested input/output object", session.Session["audio"])
	}
	for _, dir := range []string{"input", "output"} {
		side, ok := audio[dir].(map[string]any)
		if !ok {
			t.Fatalf("audio.%s = %v, want an object", dir, audio[dir])
		}
		format, ok := side["format"].(map[string]any)
		if !ok {
			t.Fatalf("audio.%s.format = %v, want an object", dir, side["format"])
		}
		if format[keyType] != pcmFormat {
			t.Errorf("audio.%s.format.type = %v, want %q", dir, format[keyType], pcmFormat)
		}
		if format["rate"] != defaultSampleRate {
			t.Errorf("audio.%s.format.rate = %v, want %d", dir, format["rate"], defaultSampleRate)
		}
	}
}

// TestSessionUpdateManualTurnDetection checks turning server VAD off sends an
// explicit null, which is how xAI is put into manual turn detection. Omitting
// the field would leave server VAD on.
func TestSessionUpdateManualTurnDetection(t *testing.T) {
	off := false
	s := New(Config{APIKey: "k", ServerVAD: &off})
	session := s.sessionUpdate()

	got, present := session.Session["turn_detection"]
	if !present {
		t.Fatal("turn_detection is absent, want an explicit null")
	}
	if got != nil {
		t.Errorf("turn_detection = %v, want null", got)
	}

	// The null has to survive marshaling, not just sit in the map.
	data, err := json.Marshal(session)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var wire struct {
		Session map[string]json.RawMessage `json:"session"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(wire.Session["turn_detection"]) != "null" {
		t.Errorf("marshaled turn_detection = %s, want null", wire.Session["turn_detection"])
	}
}

// TestSessionUpdateSampleRate checks a configured rate reaches both directions.
func TestSessionUpdateSampleRate(t *testing.T) {
	s := New(Config{APIKey: "k", SampleRate: 16000})
	audio, ok := s.sessionUpdate().Session["audio"].(map[string]any)
	if !ok {
		t.Fatal("audio is not an object")
	}
	for _, dir := range []string{"input", "output"} {
		side, _ := audio[dir].(map[string]any)
		format, _ := side["format"].(map[string]any)
		if format["rate"] != 16000 {
			t.Errorf("audio.%s.format.rate = %v, want 16000", dir, format["rate"])
		}
	}
}

// TestToolSpecs checks xAI's built-in search tools and the pipeline's function
// tools are rendered together, and that a session with neither omits the field.
func TestToolSpecs(t *testing.T) {
	t.Run("none", func(t *testing.T) {
		s := New(Config{APIKey: "k"})
		if got := s.toolSpecs(nil); got != nil {
			t.Errorf("toolSpecs(nil) = %v, want nil so the field is omitted", got)
		}
		if _, ok := s.sessionUpdate().Session["tools"]; ok {
			t.Error("session carries a tools field with no tools configured")
		}
	})

	t.Run("built-in and function tools", func(t *testing.T) {
		s := New(Config{
			APIKey:         "k",
			WebSearch:      true,
			XSearch:        true,
			XSearchHandles: []string{"xai"},
			FileSearch:     &FileSearch{VectorStoreIDs: []string{"col_1"}, MaxResults: 5},
		})
		specs := s.toolSpecs([]frames.Tool{{
			Name:        "get_weather",
			Description: "look up the weather",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		}})

		if len(specs) != 4 {
			t.Fatalf("got %d tools, want the three built-ins plus the function", len(specs))
		}
		if specs[0][keyType] != "web_search" {
			t.Errorf("tool 0 = %v, want web_search", specs[0])
		}
		if specs[1][keyType] != "x_search" {
			t.Errorf("tool 1 = %v, want x_search", specs[1])
		}
		if handles, ok := specs[1]["allowed_x_handles"].([]string); !ok || len(handles) != 1 || handles[0] != "xai" {
			t.Errorf("x_search handles = %v, want the configured handle", specs[1]["allowed_x_handles"])
		}
		if specs[2][keyType] != "file_search" {
			t.Errorf("tool 2 = %v, want file_search", specs[2])
		}
		if specs[2]["max_num_results"] != 5 {
			t.Errorf("file_search max_num_results = %v, want 5", specs[2]["max_num_results"])
		}
		if specs[3][keyType] != "function" || specs[3]["name"] != "get_weather" {
			t.Errorf("tool 3 = %v, want the get_weather function", specs[3])
		}
	})

	t.Run("x_search without handles omits the filter", func(t *testing.T) {
		s := New(Config{APIKey: "k", XSearch: true})
		specs := s.toolSpecs(nil)
		if len(specs) != 1 {
			t.Fatalf("got %d tools, want just x_search", len(specs))
		}
		if _, ok := specs[0]["allowed_x_handles"]; ok {
			t.Error("allowed_x_handles is present, want it omitted when unrestricted")
		}
	})
}

// TestConfigValidate pins which Config fields the provider requires.
func TestConfigValidate(t *testing.T) {
	cases := []struct {
		name  string
		cfg   Config
		valid bool
	}{
		{"missing API key", Config{}, false},
		{"API key only", Config{APIKey: "k"}, true},
		{"supported sample rate", Config{APIKey: "k", SampleRate: 48000}, true},
		{"unsupported sample rate", Config{APIKey: "k", SampleRate: 11025}, false},
		{"file search with a collection", Config{APIKey: "k", FileSearch: &FileSearch{VectorStoreIDs: []string{"c"}}}, true},
		{"file search without a collection", Config{APIKey: "k", FileSearch: &FileSearch{}}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.cfg.Validate()
			if c.valid && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
			if !c.valid && err == nil {
				t.Error("Validate() = nil, want an error")
			}
		})
	}
}

// TestResponseDoneUsageParsing checks both places xAI reports token accounting.
func TestResponseDoneUsageParsing(t *testing.T) {
	want := frames.LLMTokenUsage{PromptTokens: 150, CompletionTokens: 50, TotalTokens: 200}

	t.Run("nested in the response", func(t *testing.T) {
		raw := `{"type":"response.done","response":{"usage":{"total_tokens":200,"input_tokens":150,"output_tokens":50}}}`
		var ev serverEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if ev.Response == nil || ev.Response.Usage == nil {
			t.Fatal("response usage not parsed")
		}
		if got := ev.Response.Usage.tokenUsage(); got != want {
			t.Errorf("tokenUsage = %+v, want %+v", got, want)
		}
	})

	t.Run("at the top level", func(t *testing.T) {
		raw := `{"type":"response.done","usage":{"total_tokens":200,"input_tokens":150,"output_tokens":50}}`
		var ev serverEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if ev.Usage == nil {
			t.Fatal("top-level usage not parsed")
		}
		if got := ev.Usage.tokenUsage(); got != want {
			t.Errorf("tokenUsage = %+v, want %+v", got, want)
		}
	})
}
