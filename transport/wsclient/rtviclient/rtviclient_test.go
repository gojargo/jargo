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

// as asserts f is a T and returns it.
func as[T frames.Frame](t *testing.T, f frames.Frame) T {
	t.Helper()
	got, ok := f.(T)
	if !ok {
		var want T
		t.Fatalf("got %T, want %T", f, want)
	}
	return got
}

func TestLLMLifecycle(t *testing.T) {
	as[*frames.LLMFullResponseStartFrame](t, deserialize(t, server(t, "bot-llm-started", nil)))
	text := as[*frames.LLMTextFrame](t, deserialize(t, server(t, "bot-llm-text", map[string]any{"text": "Paris"})))
	if text.Text != "Paris" {
		t.Errorf("text = %q, want Paris", text.Text)
	}
	as[*frames.LLMFullResponseEndFrame](t, deserialize(t, server(t, "bot-llm-stopped", nil)))
}

func TestLLMMarker(t *testing.T) {
	f := as[*frames.LLMMarkerResponseFrame](t, deserialize(t, server(t, "bot-llm-marker", map[string]any{
		"text": "◐", "kind": "short", "raw": "◐ wait", "markers": []string{"●", "◐", "○"},
	})))
	if f.Marker != "◐" || f.Kind != "short" || f.Raw != "◐ wait" {
		t.Errorf("(marker, kind, raw) = (%q, %q, %q), want (◐, short, ◐ wait)", f.Marker, f.Kind, f.Raw)
	}
	if want := []string{"●", "◐", "○"}; !reflect.DeepEqual(f.Markers, want) {
		t.Errorf("markers = %q, want %q", f.Markers, want)
	}

	// A response that carried no marker reports an empty text.
	f = as[*frames.LLMMarkerResponseFrame](t, deserialize(t, server(t, "bot-llm-marker", map[string]any{
		"text": "", "raw": "Hi",
	})))
	if f.Marker != "" || f.Kind != "" || f.Raw != "Hi" {
		t.Errorf("(marker, kind, raw) = (%q, %q, %q), want (none, none, Hi)", f.Marker, f.Kind, f.Raw)
	}
}

func TestTTSText(t *testing.T) {
	f := as[*frames.TTSTextFrame](t, deserialize(t, server(t, "bot-tts-text", map[string]any{"text": "Hello there!"})))
	if f.Text != "Hello there!" {
		t.Errorf("text = %q, want Hello there!", f.Text)
	}
}

func TestSpeakingAndInterruption(t *testing.T) {
	as[*frames.BotStartedSpeakingFrame](t, deserialize(t, server(t, "bot-started-speaking", nil)))
	as[*frames.BotStoppedSpeakingFrame](t, deserialize(t, server(t, "bot-stopped-speaking", nil)))
	as[*frames.InterruptionFrame](t, deserialize(t, server(t, "bot-interrupted", nil)))
}

func TestUserTranscriptionFinalVsInterim(t *testing.T) {
	final := as[*frames.TranscriptionFrame](t, deserialize(t, server(t, "user-transcription", map[string]any{
		"text": "hello", "user_id": "u", "timestamp": "t", "final": true,
	})))
	if final.Text != "hello" || final.UserID != "u" {
		t.Errorf("final = (%q, %q), want (hello, u)", final.Text, final.UserID)
	}

	interim := as[*frames.InterimTranscriptionFrame](t, deserialize(t, server(t, "user-transcription", map[string]any{
		"text": "hel", "final": false,
	})))
	if interim.Text != "hel" {
		t.Errorf("interim text = %q, want hel", interim.Text)
	}
}

func TestFunctionCall(t *testing.T) {
	f := as[*frames.FunctionCallInProgressFrame](t, deserialize(t, server(t, "llm-function-call-in-progress",
		map[string]any{
			"function_name": "get_weather",
			"tool_call_id":  "c1",
			"arguments":     map[string]any{"city": "Paris"},
		})))
	if f.ToolName != "get_weather" {
		t.Errorf("function name = %q, want get_weather", f.ToolName)
	}
	var args map[string]any
	if err := json.Unmarshal(f.Args, &args); err != nil {
		t.Fatalf("arguments %s: %v", f.Args, err)
	}
	if want := map[string]any{"city": "Paris"}; !reflect.DeepEqual(args, want) {
		t.Errorf("arguments = %v, want %v", args, want)
	}
}

func TestUnknownAndNonRTVIDropped(t *testing.T) {
	cases := map[string][]byte{
		"bot-ready": server(t, "bot-ready", map[string]any{"version": "2.0.0"}),
		"metrics":   server(t, "metrics", map[string]any{}),
		"no label":  []byte(`{"type": "bot-llm-text"}`),
		"not json":  []byte("not json"),
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
