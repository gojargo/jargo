package evalws_test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// encode is msg as the harness puts it on the wire.
func encode(t *testing.T, msg any) []byte {
	t.Helper()
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// deserialize runs a fresh serializer over msg.
func deserialize(t *testing.T, msg any) frames.Frame {
	t.Helper()
	f, err := evalws.NewSerializer().Deserialize(encode(t, msg))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// transportMessage asserts f wraps an RTVI message for the RTVI processor, and
// returns that message.
func transportMessage(t *testing.T, f frames.Frame) map[string]any {
	t.Helper()
	mf, ok := f.(*frames.InputTransportMessageFrame)
	if !ok {
		t.Fatalf("got %T, want an InputTransportMessageFrame", f)
	}
	var message map[string]any
	if err := json.Unmarshal(mf.Message, &message); err != nil {
		t.Fatal(err)
	}
	return message
}

// clientMessage is an RTVI client-message carrying d under t.
func clientMessage(id, msgType string, d any) map[string]any {
	return map[string]any{
		"label": rtvi.MessageLabel,
		"type":  rtvi.TypeClientMessage,
		"id":    id,
		"data":  map[string]any{"t": msgType, "d": d},
	}
}

func TestSendTextWrapsAsTransportMessage(t *testing.T) {
	msg := map[string]any{
		"label": rtvi.MessageLabel,
		"type":  "send-text",
		"id":    "1",
		"data":  map[string]any{"content": "hi"},
	}
	got := transportMessage(t, deserialize(t, msg))
	if !reflect.DeepEqual(got, msg) {
		t.Fatalf("message = %v, want %v", got, msg)
	}
}

// raw-audio is forwarded to the RTVI processor, which decodes it and pushes an
// InputAudioRawFrame downstream into the input transport.
func TestRawAudioWrapsAsTransportMessage(t *testing.T) {
	msg := map[string]any{
		"label": rtvi.MessageLabel,
		"type":  "raw-audio",
		"id":    "2",
		"data":  map[string]any{"base64Audio": "AAAA", "sampleRate": 16000, "numChannels": 1},
	}
	if got := transportMessage(t, deserialize(t, msg)); got["type"] != "raw-audio" {
		t.Fatalf("type = %v, want raw-audio", got["type"])
	}
}

func TestOtherRTVIMessageWrapsAsTransportMessage(t *testing.T) {
	msg := map[string]any{"label": rtvi.MessageLabel, "type": "client-ready", "id": "3", "data": map[string]any{}}
	transportMessage(t, deserialize(t, msg))
}

func TestEvalContextShortCircuitsToMessagesUpdate(t *testing.T) {
	// The messages are universal LLM messages, a role and its content.
	msg := clientMessage("4", evalws.ContextMessageType, map[string]any{
		"messages": []any{map[string]any{"role": "system", "content": "be terse"}},
	})
	f, ok := deserialize(t, msg).(*frames.LLMMessagesUpdateFrame)
	if !ok {
		t.Fatalf("got %T, want an LLMMessagesUpdateFrame", f)
	}
	want := []frames.Message{{Role: frames.RoleSystem, Text: "be terse"}}
	if !reflect.DeepEqual(f.Messages, want) {
		t.Fatalf("messages = %+v, want %+v", f.Messages, want)
	}
	if f.RunLLM {
		t.Fatal("run_llm is on, want the context seeded without a response")
	}
}

// The context's messages read as LLM messages: content as text or as parts, an
// assistant's tool calls, and a tool's result. One without a role is not a
// message, and the whole context is dropped.
func TestEvalContextReadsUniversalMessages(t *testing.T) {
	msg := clientMessage("5", evalws.ContextMessageType, map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "What is this?"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AA=="}},
		}},
		map[string]any{"role": "assistant", "content": "", "tool_calls": []any{map[string]any{
			"id": "call_1", "type": "function",
			"function": map[string]any{"name": "lookup", "arguments": `{"q":"x"}`},
		}}},
		map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "found"},
	}})
	f, ok := deserialize(t, msg).(*frames.LLMMessagesUpdateFrame)
	if !ok {
		t.Fatalf("got %T, want an LLMMessagesUpdateFrame", f)
	}
	want := []frames.Message{
		{Role: frames.RoleUser, Content: []frames.ContentPart{
			{Type: frames.ContentPartText, Text: "What is this?"},
			{Type: frames.ContentPartImageURL, ImageURL: "data:image/png;base64,AA=="},
		}},
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{
			{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"x"}`)},
		}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "call_1", Content: "found"}}},
	}
	if !reflect.DeepEqual(f.Messages, want) {
		t.Fatalf("messages = %+v, want %+v", f.Messages, want)
	}

	bad := clientMessage("6", evalws.ContextMessageType, map[string]any{
		"messages": []any{map[string]any{"content": "no role"}},
	})
	if f := deserialize(t, bad); f != nil {
		t.Fatalf("got %T, want the context dropped", f)
	}
}

func TestEvalConfigureShortCircuitsToObserverConfig(t *testing.T) {
	msg := clientMessage("6", evalws.ConfigureMessageType, map[string]any{
		"function_call_report_level": map[string]any{"*": "full"},
	})
	f, ok := deserialize(t, msg).(*rtvi.ConfigureObserverFrame)
	if !ok {
		t.Fatalf("got %T, want a ConfigureObserverFrame", f)
	}
	want := map[string]rtvi.FunctionCallReportLevel{"*": rtvi.ReportFull}
	if !reflect.DeepEqual(f.FunctionCallReportLevel, want) {
		t.Fatalf("report level = %v, want %v", f.FunctionCallReportLevel, want)
	}
}

func TestEvalConfigureEnablesVADUserSpeaking(t *testing.T) {
	msg := clientMessage("8", evalws.ConfigureMessageType, map[string]any{"vad_user_speaking": true})
	f, ok := deserialize(t, msg).(*rtvi.ConfigureObserverFrame)
	if !ok {
		t.Fatalf("got %T, want a ConfigureObserverFrame", f)
	}
	if f.VADUserSpeakingEnabled == nil || !*f.VADUserSpeakingEnabled {
		t.Fatalf("vad_user_speaking = %v, want on", f.VADUserSpeakingEnabled)
	}
	// An unset report level stays nil, so it is not disturbed.
	if f.FunctionCallReportLevel != nil {
		t.Fatalf("report level = %v, want it left alone", f.FunctionCallReportLevel)
	}
	if f.BotLLMMarkerEnabled != nil {
		t.Fatalf("llm markers = %v, want them left alone", *f.BotLLMMarkerEnabled)
	}
}

func TestEvalConfigureEnablesLLMMarkers(t *testing.T) {
	msg := clientMessage("9", evalws.ConfigureMessageType, map[string]any{"llm_markers": true})
	f, ok := deserialize(t, msg).(*rtvi.ConfigureObserverFrame)
	if !ok {
		t.Fatalf("got %T, want a ConfigureObserverFrame", f)
	}
	if f.BotLLMMarkerEnabled == nil || !*f.BotLLMMarkerEnabled {
		t.Fatalf("llm markers = %v, want on", f.BotLLMMarkerEnabled)
	}
	if f.VADUserSpeakingEnabled != nil {
		t.Fatalf("vad_user_speaking = %v, want it left alone", *f.VADUserSpeakingEnabled)
	}
}

func TestEvalImageStoredAndNotForwarded(t *testing.T) {
	img := []byte("\x89PNG-fake-bytes")
	msg := clientMessage("7", evalws.ImageMessageType, map[string]any{
		"image":  base64.StdEncoding.EncodeToString(img),
		"format": "image/png",
	})
	s := evalws.NewSerializer()
	// eval-image is consumed (not forwarded) and kept for a later image request.
	f, err := s.Deserialize(encode(t, msg))
	if err != nil || f != nil {
		t.Fatalf("Deserialize = %v, %v, want it consumed", f, err)
	}
	data, format, ok := s.UserImage()
	if !ok || !bytes.Equal(data, img) || format != "image/png" {
		t.Fatalf("user image = %q, %q, %v, want %q, image/png", data, format, ok, img)
	}
}

func TestEvalConnectConsumedAndNotForwarded(t *testing.T) {
	message := evalws.ConnectMessage(evalws.ConnectionFlags{SkipTTS: true})
	f, err := evalws.NewSerializer().Deserialize([]byte(message))
	if err != nil || f != nil {
		t.Fatalf("Deserialize = %v, %v, want it consumed", f, err)
	}
}

// DTMF is a first-class RTVI message handled by the RTVI processor, so the
// serializer forwards it like any other RTVI message.
// redPNG is a 4x3 PNG filled with red.
func redPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 4, 3))
	for y := range 3 {
		for x := range 4 {
			img.Set(x, y, color.RGBA{R: 255, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestUserImageFrameDecodesTheRegisteredImage(t *testing.T) {
	s := evalws.NewSerializer()
	msg := clientMessage("7", evalws.ImageMessageType, map[string]any{
		"image":  base64.StdEncoding.EncodeToString(redPNG(t)),
		"format": "image/png",
	})
	if _, err := s.Deserialize(encode(t, msg)); err != nil {
		t.Fatal(err)
	}
	request := frames.NewUserImageRequestFrame("user")
	request.Text = "what is this?"

	f, err := s.UserImageFrame(request)
	if err != nil {
		t.Fatal(err)
	}
	if f == nil {
		t.Fatal("UserImageFrame = nil, want the registered image")
	}
	if f.Width != 4 || f.Height != 3 {
		t.Fatalf("size = (%d, %d), want (4, 3)", f.Width, f.Height)
	}
	if f.Format != "RGB" {
		t.Fatalf("format = %q, want RGB", f.Format)
	}
	if !bytes.Equal(f.Image[:3], []byte{255, 0, 0}) {
		t.Fatalf("first pixel = %v, want [255 0 0]", f.Image[:3])
	}
	if f.Request != request {
		t.Fatal("the frame does not carry the request it answers")
	}
}

func TestUserImageFrameWithoutAnImageIsNone(t *testing.T) {
	f, err := evalws.NewSerializer().UserImageFrame(frames.NewUserImageRequestFrame("user"))
	if err != nil || f != nil {
		t.Fatalf("UserImageFrame = %v, %v, want nil", f, err)
	}
}

func TestDTMFMessageForwardedToProcessor(t *testing.T) {
	msg := map[string]any{
		"label": rtvi.MessageLabel,
		"type":  "dtmf",
		"id":    "9",
		"data":  map[string]any{"button": "#"},
	}
	if got := transportMessage(t, deserialize(t, msg)); got["type"] != "dtmf" {
		t.Fatalf("type = %v, want dtmf", got["type"])
	}
}

func TestNonContextClientMessageIsForwarded(t *testing.T) {
	transportMessage(t, deserialize(t, clientMessage("5", "something-else", map[string]any{})))
}

func TestNonRTVIMessageDropped(t *testing.T) {
	if f := deserialize(t, map[string]any{"type": "send-text"}); f != nil {
		t.Fatalf("got %T, want it dropped", f)
	}
}

func TestNonJSONDropped(t *testing.T) {
	f, err := evalws.NewSerializer().Deserialize([]byte("not json"))
	if err != nil || f != nil {
		t.Fatalf("Deserialize = %v, %v, want it dropped", f, err)
	}
}

func TestRTVIServerMessageSerializedToJSON(t *testing.T) {
	message := rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeBotLLMStarted}
	out, err := evalws.NewSerializer().Serialize(frames.NewOutputTransportMessageUrgentFrame(message))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out.Data, &got); err != nil {
		t.Fatalf("Serialize produced %q: %v", out.Data, err)
	}
	if got["type"] != "bot-llm-started" {
		t.Fatalf("type = %v, want bot-llm-started", got["type"])
	}
}

func TestNonRTVITransportMessageDropped(t *testing.T) {
	f := frames.NewOutputTransportMessageUrgentFrame(map[string]any{"label": "other", "type": "x"})
	assertDropped(t, evalws.NewSerializer(), f)
}

func TestAudioFrameDropped(t *testing.T) {
	assertDropped(t, evalws.NewSerializer(), frames.NewOutputAudioRawFrame([]byte{0, 0}, 16000, 1))
}

func assertDropped(t *testing.T, s *evalws.Serializer, f frames.Frame) {
	t.Helper()
	out, err := s.Serialize(f)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Empty() {
		t.Fatalf("Serialize(%T) = %q, want nothing", f, out.Data)
	}
}

var _ wsserver.Serializer = evalws.NewSerializer()

func TestConnectMessageRoundTripsItsFlags(t *testing.T) {
	flags := evalws.ConnectionFlags{SkipTTS: true, CaptureBotImages: true}
	var message any
	if err := json.Unmarshal([]byte(evalws.ConnectMessage(flags)), &message); err != nil {
		t.Fatal(err)
	}
	if parsed := evalws.ParseConnect(message); parsed == nil || *parsed != flags {
		t.Fatalf("parsed = %+v, want %+v", parsed, flags)
	}
}

func TestConnectMessageIsAnRTVIClientMessage(t *testing.T) {
	var message map[string]any
	if err := json.Unmarshal([]byte(evalws.ConnectMessage(evalws.ConnectionFlags{})), &message); err != nil {
		t.Fatal(err)
	}
	if message["label"] != rtvi.MessageLabel || message["type"] != "client-message" {
		t.Fatalf("message = %v, want an RTVI client-message", message)
	}
}

func TestConnectMessageMissingFlagsDefaultOff(t *testing.T) {
	var message map[string]any
	if err := json.Unmarshal([]byte(evalws.ConnectMessage(evalws.ConnectionFlags{})), &message); err != nil {
		t.Fatal(err)
	}
	data, _ := message["data"].(map[string]any)
	delete(data, "d")
	if parsed := evalws.ParseConnect(message); parsed == nil || *parsed != (evalws.ConnectionFlags{}) {
		t.Fatalf("parsed = %+v, want every flag off", parsed)
	}
}

func TestOtherMessagesAreNotEvalConnect(t *testing.T) {
	for name, message := range map[string]any{
		"client-ready": map[string]any{"label": rtvi.MessageLabel, "type": "client-ready"},
		"eval-context": map[string]any{
			"label": rtvi.MessageLabel,
			"type":  "client-message",
			"data":  map[string]any{"t": evalws.ContextMessageType, "d": map[string]any{}},
		},
		"a provider's":  map[string]any{"event": "start"},
		"not an object": "not a dict",
	} {
		if parsed := evalws.ParseConnect(message); parsed != nil {
			t.Errorf("%s message parsed as eval-connect: %+v", name, parsed)
		}
	}
}

// The bot reports its output images by size and format, only when asked.
func TestEvalSerializerBotImages(t *testing.T) {
	outputImage := func() *frames.OutputImageRawFrame {
		return frames.NewOutputImageRawFrame(make([]byte, 12), 2, 2, "RGB")
	}

	t.Run("image dropped by default", func(t *testing.T) {
		assertDropped(t, evalws.NewSerializer(), outputImage())
	})

	t.Run("image reported when captured", func(t *testing.T) {
		s := evalws.NewSerializer()
		s.SetCaptureImages(true)
		out, err := s.Serialize(outputImage())
		if err != nil {
			t.Fatal(err)
		}
		var message map[string]any
		if err := json.Unmarshal(out.Data, &message); err != nil {
			t.Fatal(err)
		}
		if message["type"] != evalws.BotImageType {
			t.Fatalf("type = %v, want %s", message["type"], evalws.BotImageType)
		}
		want := map[string]any{"width": 2.0, "height": 2.0, "format": "RGB"}
		if !reflect.DeepEqual(message["data"], want) {
			t.Fatalf("data = %v, want %v", message["data"], want)
		}
	})
}
