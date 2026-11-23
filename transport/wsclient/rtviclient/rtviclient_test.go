package rtviclient_test

import (
	"encoding/base64"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient/rtviclient"
)

// server builds an RTVI server message as a bot sends it.
func server(t *testing.T, msgType string, data map[string]any) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"label": rtvi.MessageLabel, "type": msgType, "data": data})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// deserialize decodes one message with a fresh serializer.
func deserialize(t *testing.T, data []byte) frames.Frame {
	t.Helper()
	f, err := rtviclient.New().Deserialize(data)
	if err != nil {
		t.Fatalf("Deserialize(%s): %v", data, err)
	}
	return f
}

// raw asserts the bot's message of msgType with data stayed a raw message, and
// returns the message the frame carries, decoded.
func raw(t *testing.T, msgType string, data map[string]any) map[string]any {
	t.Helper()
	f, ok := deserialize(t, server(t, msgType, data)).(*frames.InputTransportMessageFrame)
	if !ok {
		t.Fatalf("%s: got %T, want a raw message", msgType, f)
	}
	var message map[string]any
	if err := json.Unmarshal(f.Message, &message); err != nil {
		t.Fatalf("%s: %v", f.Message, err)
	}
	if message["type"] != msgType {
		t.Fatalf("type = %v, want %s", message["type"], msgType)
	}
	// A message built without data carries a null one.
	var want any
	if data != nil {
		want = jsonRoundTrip(t, data)
	}
	if !reflect.DeepEqual(message["data"], want) {
		t.Fatalf("%s: data = %v, want %v", msgType, message["data"], want)
	}
	return message
}

// jsonRoundTrip is v as JSON decodes it.
func jsonRoundTrip(t *testing.T, v any) any {
	t.Helper()
	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(encoded, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// Reports about the bot's own LLM: this pipeline's LLM frames would be about
// its own.
func TestLLMReportsStayRawMessages(t *testing.T) {
	raw(t, "bot-llm-started", nil)
	raw(t, "bot-llm-text", map[string]any{"text": "Paris"})
	raw(t, "bot-llm-marker", map[string]any{"text": "◐", "kind": "short", "raw": "◐ wait"})
	raw(t, "bot-llm-stopped", nil)
}

func TestTTSTextStaysARawMessage(t *testing.T) {
	raw(t, "bot-tts-text", map[string]any{"text": "Hello there!"})
}

// Reports about the bot's own output: this pipeline's speaking frames and
// interruptions would be about its own.
func TestSpeakingAndInterruptionStayRawMessages(t *testing.T) {
	for _, msgType := range []string{"bot-started-speaking", "bot-stopped-speaking", "bot-interrupted"} {
		raw(t, msgType, nil)
	}
}

// The bot's transcription of this pipeline's speech: a TranscriptionFrame would
// pass it off as this pipeline's own.
func TestUserTranscriptionStaysARawMessage(t *testing.T) {
	raw(t, "user-transcription", map[string]any{"text": "hello", "user_id": "u", "timestamp": "t", "final": true})
	raw(t, "user-transcription", map[string]any{"text": "hel", "final": false})
}

// Reports about the bot's own function calls: this pipeline's function call
// frames would be about its own.
func TestFunctionCallsStayRawMessages(t *testing.T) {
	raw(t, "llm-function-call-in-progress", map[string]any{
		"function_name": "get_weather",
		"tool_call_id":  "c1",
		"arguments":     map[string]any{"city": "Paris"},
	})
	raw(t, "llm-function-call-stopped", map[string]any{
		"function_name": "get_weather", "tool_call_id": "c1", "cancelled": true, //nolint:misspell // the protocol's spelling
	})
}

func TestEveryRTVIMessageStaysRaw(t *testing.T) {
	// The last is a message type a later protocol version adds.
	for _, msgType := range []string{"bot-ready", "metrics", "server-message", "bot-something-new"} {
		raw(t, msgType, map[string]any{})
	}
}

// The frame carries the message as the bot sent it, a field the client does not
// know included.
func TestTheMessageIsCarriedAsSent(t *testing.T) {
	data := server(t, "bot-llm-text", map[string]any{"text": "Paris", "x_unknown": 1})
	f, ok := deserialize(t, data).(*frames.InputTransportMessageFrame)
	if !ok || string(f.Message) != string(data) {
		t.Fatalf("frame = %v, want the message as sent", f)
	}
}

func TestNonRTVIDropped(t *testing.T) {
	cases := map[string][]byte{
		"no label":      []byte(`{"type": "bot-llm-text"}`),
		"not json":      []byte("not json"),
		"not an object": []byte(`["rtvi-ai"]`),
	}
	for name, data := range cases {
		if f := deserialize(t, data); f != nil {
			t.Errorf("%s: got %T, want no frame", name, f)
		}
	}
}

func TestOutputAudioBecomesRawAudio(t *testing.T) {
	f := frames.NewOutputAudioRawFrame([]byte{0x01, 0x02, 0x03, 0x04}, 16000, 1)
	m, err := rtviclient.New().Serialize(f)
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Label string `json:"label"`
		Type  string `json:"type"`
		Data  struct {
			Base64Audio string `json:"base64Audio"` //nolint:tagliatelle // RTVI wire field
			SampleRate  int    `json:"sampleRate"`  //nolint:tagliatelle // RTVI wire field
			NumChannels int    `json:"numChannels"` //nolint:tagliatelle // RTVI wire field
		} `json:"data"`
	}
	if err := json.Unmarshal(m.Data, &out); err != nil {
		t.Fatalf("%s: %v", m.Data, err)
	}
	if out.Label != rtvi.MessageLabel || out.Type != "raw-audio" {
		t.Errorf("(label, type) = (%q, %q), want (%q, raw-audio)", out.Label, out.Type, rtvi.MessageLabel)
	}
	if out.Data.SampleRate != 16000 || out.Data.NumChannels != 1 {
		t.Errorf("(sampleRate, numChannels) = (%d, %d), want (16000, 1)", out.Data.SampleRate, out.Data.NumChannels)
	}
	pcm, err := base64.StdEncoding.DecodeString(out.Data.Base64Audio)
	if err != nil || string(pcm) != "\x01\x02\x03\x04" {
		t.Errorf("audio = %v (%v), want 01 02 03 04", pcm, err)
	}
}

func TestClientMessagePassthrough(t *testing.T) {
	msg := map[string]any{
		"label": rtvi.MessageLabel,
		"type":  "send-text",
		"id":    "1",
		"data":  map[string]any{"content": "hi", "options": map[string]any{"run_immediately": true}},
	}
	m, err := rtviclient.New().Serialize(frames.NewOutputTransportMessageFrame(msg))
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(m.Data, &out); err != nil {
		t.Fatalf("%s: %v", m.Data, err)
	}
	if !reflect.DeepEqual(out, msg) {
		t.Errorf("out = %v, want %v", out, msg)
	}
}

func TestNonRTVITransportMessageDropped(t *testing.T) {
	m, err := rtviclient.New().Serialize(frames.NewOutputTransportMessageFrame(map[string]any{"foo": "bar"}))
	if err != nil || !m.Empty() {
		t.Errorf("Serialize(non-RTVI) = %q, %v; want nothing", m.Data, err)
	}
}
