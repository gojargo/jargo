package bus_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Ported from upstream's bus serializer suite. What it guards is that a frame
// crossing a bus that leaves the process arrives as the same thing: the
// conversation, the toolset, and the tools written in one provider's own format
// in particular, which are the ones a round trip is most likely to flatten.

// standardTool is upstream's fixture tool.
func standardTool() frames.Tool {
	return frames.Tool{
		Name:        "get_weather",
		Description: "Get the weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}
}

// roundTrip sends a message through the serializer and back.
func roundTrip(t *testing.T, m bus.Message) bus.Message {
	t.Helper()
	s := bus.NewJSONSerializer(nil)
	raw, err := s.Serialize(m)
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	back, err := s.Deserialize(raw)
	if err != nil {
		t.Fatalf("Deserialize: %v", err)
	}
	return back
}

// contextRoundTrip sends a conversation across on a frame message and returns
// the one that arrived, which is how a conversation actually crosses a bus.
func contextRoundTrip(t *testing.T, c *frames.LLMContext) *frames.LLMContext {
	t.Helper()
	msg := &bus.FrameMessage{
		Frame:     frames.NewLLMContextFrame(c),
		Direction: processor.Downstream,
	}
	msg.From = "worker-a"

	back, ok := roundTrip(t, msg).(*bus.FrameMessage)
	if !ok {
		t.Fatalf("a frame message came back as something else")
	}
	cf, ok := back.Frame.(*frames.LLMContextFrame)
	if !ok {
		t.Fatalf("the frame came back as %T, want an LLMContextFrame", back.Frame)
	}
	return cf.Context
}

// TestToolsSchemaPreservesCustomTools is upstream's
// test_tools_schema_preserves_custom_tools: the tools written in one provider's
// own format have to survive the crossing. They are the reason to configure that
// provider, and nothing else carries them.
func TestToolsSchemaPreservesCustomTools(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetToolsSchema(frames.ToolsSchema{
		Standard: []frames.Tool{standardTool()},
		Custom: map[frames.AdapterType][]any{
			frames.AdapterTypeGemini: {map[string]any{"google_search": map[string]any{}}},
		},
	})

	got := contextRoundTrip(t, c).ToolsSchema()

	if len(got.Standard) != 1 || got.Standard[0].Name != "get_weather" {
		t.Fatalf("standard tools = %+v, want the one that was sent", got.Standard)
	}
	custom := got.CustomFor(frames.AdapterTypeGemini)
	if len(custom) != 1 {
		t.Fatalf("custom tools = %+v, want the one that was sent", got.Custom)
	}
	entry, ok := custom[0].(map[string]any)
	if !ok {
		t.Fatalf("the custom tool came back as %T, want the object that was sent", custom[0])
	}
	if _, present := entry["google_search"]; !present {
		t.Errorf("the custom tool came back as %+v, want it to carry google_search", entry)
	}
}

// TestToolsSchemaWithoutCustomToolsStaysEmpty is upstream's
// test_tools_schema_without_custom_tools_round_trips_to_none: a toolset that
// advertised nothing in a provider's own format must not arrive advertising an
// empty set, which reads differently to an adapter.
func TestToolsSchemaWithoutCustomToolsStaysEmpty(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetToolsSchema(frames.ToolsSchema{Standard: []frames.Tool{standardTool()}})

	if got := contextRoundTrip(t, c).ToolsSchema(); len(got.Custom) != 0 {
		t.Errorf("custom tools = %+v, want none", got.Custom)
	}
}

// TestConversationCrossesWhole checks the rest of a conversation survives with
// the toolset: what was said, the system prompt held apart from it, and whether
// the model must call a tool.
func TestConversationCrossesWhole(t *testing.T) {
	c := frames.NewLLMContext("You are helpful.")
	c.SetMessages([]frames.Message{
		{Role: frames.RoleUser, Text: "hi"},
		{Role: frames.RoleAssistant, Text: "hello"},
	})
	c.SetToolChoice(frames.ToolChoiceRequired)

	got := contextRoundTrip(t, c)

	if got.System() != "You are helpful." {
		t.Errorf("system = %q, want the prompt that was sent", got.System())
	}
	msgs := got.Messages()
	if len(msgs) != 2 || msgs[0].Text != "hi" || msgs[1].Role != frames.RoleAssistant {
		t.Errorf("messages = %+v, want the two that were sent", msgs)
	}
	if got.ToolChoice() != frames.ToolChoiceRequired {
		t.Errorf("tool choice = %q, want required", got.ToolChoice())
	}
}

// TestToolCallsAndResultsCross checks the parts of a turn that carry a tool
// exchange, which is what a worker running a job actually sends back.
func TestToolCallsAndResultsCross(t *testing.T) {
	c := frames.NewLLMContext("")
	c.SetMessages([]frames.Message{
		{
			Role:      frames.RoleAssistant,
			ToolCalls: []frames.ToolCall{{ID: "call-1", Name: "get_weather", Args: json.RawMessage(`{"location":"Lyon"}`)}},
		},
		{
			Role:        frames.RoleUser,
			ToolResults: []frames.ToolResult{{ID: "call-1", Name: "get_weather", Content: "sunny"}},
		},
	})

	msgs := contextRoundTrip(t, c).Messages()
	if len(msgs) != 2 {
		t.Fatalf("messages = %+v, want 2", msgs)
	}
	if len(msgs[0].ToolCalls) != 1 || msgs[0].ToolCalls[0].ID != "call-1" {
		t.Errorf("tool calls = %+v, want the one that was sent", msgs[0].ToolCalls)
	}
	if string(msgs[0].ToolCalls[0].Args) != `{"location":"Lyon"}` {
		t.Errorf("tool call args = %s, want the arguments that were sent", msgs[0].ToolCalls[0].Args)
	}
	if len(msgs[1].ToolResults) != 1 || msgs[1].ToolResults[0].Content != "sunny" {
		t.Errorf("tool results = %+v, want the one that was sent", msgs[1].ToolResults)
	}
}

// TestFrameMessageCrossesWithItsEnvelope is upstream's
// test_bytes_round_trip_through_serialize_deserialize: the whole message, not
// just the frame inside it, has to arrive intact.
func TestFrameMessageCrossesWithItsEnvelope(t *testing.T) {
	msg := &bus.FrameMessage{
		Frame:     frames.NewTranscriptionFrame("bonjour", "user-1", "ts"),
		Direction: processor.Upstream,
		Bridge:    "bridge-1",
	}
	msg.From = "worker-a"
	msg.To = "worker-b"

	back, ok := roundTrip(t, msg).(*bus.FrameMessage)
	if !ok {
		t.Fatal("the message came back as something else")
	}
	if back.Source() != "worker-a" || back.Target() != "worker-b" {
		t.Errorf("envelope = %q -> %q, want worker-a -> worker-b", back.Source(), back.Target())
	}
	if back.Direction != processor.Upstream || back.Bridge != "bridge-1" {
		t.Errorf("direction/bridge = %v/%q, want upstream/bridge-1", back.Direction, back.Bridge)
	}
	tf, ok := back.Frame.(*frames.TranscriptionFrame)
	if !ok {
		t.Fatalf("the frame came back as %T, want a TranscriptionFrame", back.Frame)
	}
	if tf.Text != "bonjour" || tf.UserID != "user-1" {
		t.Errorf("transcription = %+v, want the one that was sent", tf)
	}
}

// TestAudioCrossesAsBytes checks raw audio survives, which JSON cannot carry on
// its own and which every media frame is made of.
func TestAudioCrossesAsBytes(t *testing.T) {
	pcm := []byte{0, 1, 2, 3, 250, 251}
	msg := &bus.FrameMessage{
		Frame:     frames.NewInputAudioRawFrame(pcm, 16000, 1),
		Direction: processor.Downstream,
	}
	msg.From = "worker-a"

	back, ok := roundTrip(t, msg).(*bus.FrameMessage)
	if !ok {
		t.Fatal("the message came back as something else")
	}
	af, ok := back.Frame.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("the frame came back as %T, want an InputAudioRawFrame", back.Frame)
	}
	if string(af.Audio) != string(pcm) {
		t.Errorf("audio = %v, want the bytes that were sent", af.Audio)
	}
	if af.SampleRate != 16000 || af.NumChannels != 1 {
		t.Errorf("audio shape = %d Hz %d channels, want 16000/1", af.SampleRate, af.NumChannels)
	}
}

// TestUnknownTypeIsRefused checks a message naming a type this process does not
// have is reported rather than arriving as something else. A peer running a
// newer build is the ordinary case, and guessing would be worse than saying so.
func TestUnknownTypeIsRefused(t *testing.T) {
	s := bus.NewJSONSerializer(nil)
	_, err := s.Deserialize([]byte(`{"__type__":"NoSuchMessage","__data__":{}}`))
	if !errors.Is(err, bus.ErrUnknownType) {
		t.Errorf("Deserialize of an unknown type = %v, want ErrUnknownType", err)
	}
}

// TestNonMessageIsRefused checks bytes carrying something that is not a bus
// message are refused, rather than being handed on as one.
func TestNonMessageIsRefused(t *testing.T) {
	s := bus.NewJSONSerializer(nil)
	_, err := s.Deserialize([]byte(`{"__type__":"TranscriptionFrame","__data__":{"Text":"hi"}}`))
	if !errors.Is(err, bus.ErrNotAMessage) {
		t.Errorf("Deserialize of a bare frame = %v, want ErrNotAMessage", err)
	}
}
