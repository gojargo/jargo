package events_test

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

// session decodes the session of a session update as it travels.
func session(t *testing.T, sp events.SessionProperties) map[string]any {
	t.Helper()
	raw, err := json.Marshal(events.NewSessionUpdateEvent(sp))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var msg struct {
		EventID string         `json:"event_id"`
		Type    string         `json:"type"`
		Session map[string]any `json:"session"`
	}
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if msg.Type != "session.update" || msg.EventID == "" {
		t.Errorf("event = %s, want a session.update with an id", raw)
	}
	return msg.Session
}

// Ports of tests/test_openai_realtime_events.py.

// An event of a type with no model here parses to nothing.
func TestParseServerEventReturnsNoneForUnmodeledType(t *testing.T) {
	ev, err := events.ParseServerEvent([]byte(`{"event_id":"ev_1","type":"output_audio_buffer.cleared"}`))
	if err != nil || ev != nil {
		t.Errorf("event = %+v, err = %v, want neither", ev, err)
	}
}

// The server echoes transcription fields that may not be modeled here; parsing
// ignores them.
func TestParseSessionUpdatedWithUnmodeledTranscriptionFields(t *testing.T) {
	raw := `{"event_id":"ev_1","type":"session.updated","session":{"type":"realtime",` +
		`"audio":{"input":{"transcription":{"model":"gpt-live-transcribe","language":null,` +
		`"languages":["en","es"],"prompt":null,"a_field_openai_adds_later":true}}}}}`

	ev, err := events.ParseServerEvent([]byte(raw))
	if err != nil || ev == nil || ev.Session == nil {
		t.Fatalf("event = %+v, err = %v, want a session.updated", ev, err)
	}
	parsed := ev.Session.InputTranscription()
	if parsed == nil {
		t.Fatal("the session carries no transcription")
	}
	if parsed.Model != "gpt-live-transcribe" {
		t.Errorf("model = %q, want gpt-live-transcribe", parsed.Model)
	}
	if !slices.Equal(parsed.Languages, []string{"en", "es"}) {
		t.Errorf("languages = %q, want [en es]", parsed.Languages)
	}
}

// A response done carrying only its status parses, with no usage and no
// status details.
func TestParseResponseDoneWithOnlyStatus(t *testing.T) {
	ev, err := events.ParseServerEvent(
		[]byte(`{"event_id":"ev_1","type":"response.done","response":{"status":"completed"}}`))
	if err != nil || ev == nil || ev.Response == nil {
		t.Fatalf("event = %+v, err = %v, want a response.done", ev, err)
	}
	if ev.Response.Usage != nil || ev.Response.StatusDetails != nil {
		t.Errorf("response = %+v, want no usage and no status details", ev.Response)
	}
}

// Ports of the session-properties cases of tests/test_openai_realtime_reasoning.py.
// (The case coercing a dict into Reasoning is pydantic's own behavior, which a
// typed Go field has no counterpart for.)

func TestSessionPropertiesAcceptsReasoningObject(t *testing.T) {
	sp := events.SessionProperties{Reasoning: &events.Reasoning{Effort: "high"}}
	if sp.Reasoning == nil || sp.Reasoning.Effort != "high" {
		t.Errorf("reasoning = %+v, want high", sp.Reasoning)
	}
}

func TestReasoningAcceptsFutureEffortStrings(t *testing.T) {
	if r := (events.Reasoning{Effort: "ultra"}); r.Effort != "ultra" {
		t.Errorf("effort = %q, want ultra", r.Effort)
	}
}

func TestReasoningSerializesIntoSessionUpdate(t *testing.T) {
	got := session(t, events.SessionProperties{Reasoning: &events.Reasoning{Effort: "medium"}})
	reasoning, _ := got["reasoning"].(map[string]any)
	if reasoning["effort"] != "medium" || len(reasoning) != 1 {
		t.Errorf("reasoning = %v, want {effort: medium}", got["reasoning"])
	}
}

// The tests below are not ports of upstream tests: they cover the wire shape
// upstream's models produce when dumped without their unset fields.

// Unset properties are left out, the session type is always sent, and the
// configured tools travel only in the API's form.
func TestASessionUpdateSendsOnlyWhatIsSet(t *testing.T) {
	got := session(t, events.SessionProperties{})
	if len(got) != 1 || got["type"] != "realtime" {
		t.Errorf("session = %v, want the type alone", got)
	}
}

// Server-side turn detection turned off is sent as null; left unset, it is not
// sent at all.
func TestDisabledTurnDetectionIsSentAsNull(t *testing.T) {
	got := session(t, events.SessionProperties{Audio: &events.AudioConfiguration{
		Input: &events.AudioInput{TurnDetection: events.NoTurnDetection()},
	}})
	input, _ := got["audio"].(map[string]any)["input"].(map[string]any)
	if v, ok := input["turn_detection"]; !ok || v != nil {
		t.Errorf("input = %v, want turn_detection null", input)
	}

	got = session(t, events.SessionProperties{Audio: &events.AudioConfiguration{
		Input: &events.AudioInput{Transcription: &events.InputAudioTranscription{}},
	}})
	input, _ = got["audio"].(map[string]any)["input"].(map[string]any)
	if _, ok := input["turn_detection"]; ok {
		t.Errorf("input = %v, want no turn_detection", input)
	}
}

// The detection presets carry their documented defaults, and a transcription
// with no model uses the realtime whisper model.
func TestPresetsCarryTheirDefaults(t *testing.T) {
	got := session(t, events.SessionProperties{Audio: &events.AudioConfiguration{
		Input: &events.AudioInput{
			Format:        events.PCM(),
			Transcription: &events.InputAudioTranscription{},
			TurnDetection: events.ServerVAD(),
		},
	}})
	input, _ := got["audio"].(map[string]any)["input"].(map[string]any)
	td, _ := input["turn_detection"].(map[string]any)
	if td["type"] != "server_vad" || td["threshold"] != 0.5 || td["prefix_padding_ms"] != 300.0 ||
		td["silence_duration_ms"] != 500.0 {
		t.Errorf("turn_detection = %v, want server VAD at its defaults", td)
	}
	if tr, _ := input["transcription"].(map[string]any); tr["model"] != events.WhisperModel {
		t.Errorf("transcription = %v, want the whisper model", tr)
	}
	if f, _ := input["format"].(map[string]any); f["type"] != "audio/pcm" || f["rate"] != 24000.0 {
		t.Errorf("format = %v, want PCM at 24 kHz", f)
	}
}

// A response's token bound is a number or "inf", both ways.
func TestTokenLimit(t *testing.T) {
	for _, tc := range []struct {
		limit *events.TokenLimit
		want  string
	}{{events.Tokens(512), "512"}, {events.Unlimited(), `"inf"`}} {
		raw, err := json.Marshal(tc.limit)
		if err != nil || string(raw) != tc.want {
			t.Errorf("marshal = %s, %v, want %s", raw, err, tc.want)
		}
		var back events.TokenLimit
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Errorf("unmarshal %s: %v", raw, err)
		}
	}
}

// Copying properties leaves the stored ones untouched by changes to the copy.
func TestCopyIsIndependent(t *testing.T) {
	sp := events.SessionProperties{
		Instructions: "a",
		Audio: &events.AudioConfiguration{Input: &events.AudioInput{
			Transcription: &events.InputAudioTranscription{Prompt: "p"},
		}},
	}
	c := sp.Copy()
	c.Instructions = "b"
	c.Audio.Input.Transcription.Prompt = ""
	if sp.Instructions != "a" || sp.Audio.Input.Transcription.Prompt != "p" {
		t.Errorf("stored = %+v, want it unchanged", sp)
	}
}

// A conversation item gets a fresh id of its own.
func TestConversationItemsGetIDs(t *testing.T) {
	a, b := events.NewConversationItem(events.ItemMessage), events.NewConversationItem(events.ItemMessage)
	if a.ID == "" || a.ID == b.ID || strings.Contains(a.ID, "-") {
		t.Errorf("ids = %q, %q, want two distinct hex ids", a.ID, b.ID)
	}
}
