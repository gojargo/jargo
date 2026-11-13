package eval

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// clientMessageBytes is a client-message carrying payload under t, as the
// harness puts it on the wire.
func clientMessageBytes(t *testing.T, msgType string, payload any) []byte {
	t.Helper()
	data, err := json.Marshal(clientMessage(msgType, payload))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// An eval-configure asking for the raw VAD events turns them on, and leaves what
// it does not mention alone.
func TestEvalConfigureEnablesVADUserSpeaking(t *testing.T) {
	f, err := newSerializer().Deserialize(clientMessageBytes(t, evalConfigureMessage,
		map[string]any{"vad_user_speaking": true}))
	if err != nil {
		t.Fatal(err)
	}
	cf, ok := f.(*rtvi.ConfigureObserverFrame)
	if !ok {
		t.Fatalf("got %T, want a ConfigureObserverFrame", f)
	}
	if cf.VADUserSpeakingEnabled == nil || !*cf.VADUserSpeakingEnabled {
		t.Fatalf("vad_user_speaking = %v, want on", cf.VADUserSpeakingEnabled)
	}
	if cf.FunctionCallReportLevel != nil || cf.BotLLMMarkerEnabled != nil {
		t.Fatalf("frame = %v, want the report level and markers left alone", cf)
	}
}

// An eval-configure asking for the model's markers turns their reports on, under
// the llm_markers key the harness sends, and leaves the VAD events alone.
func TestEvalConfigureEnablesLLMMarkers(t *testing.T) {
	f, err := newSerializer().Deserialize(clientMessageBytes(t, evalConfigureMessage,
		map[string]any{"llm_markers": true}))
	if err != nil {
		t.Fatal(err)
	}
	cf, ok := f.(*rtvi.ConfigureObserverFrame)
	if !ok || cf.BotLLMMarkerEnabled == nil || !*cf.BotLLMMarkerEnabled || cf.VADUserSpeakingEnabled != nil {
		t.Fatalf("frame = %v, want the markers on and the VAD events left alone", f)
	}

	on := true
	sent, err := json.Marshal(configureMessage(nil, nil, &on))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(sent), `"llm_markers":true`) {
		t.Fatalf("configure message %s, want the markers asked for as llm_markers", sent)
	}
}

// An eval-context becomes the update seeding the bot's context, and an
// eval-configure the observer configuration it asks for; neither reaches the
// RTVI processor.
func TestEvalControlMessagesShortCircuit(t *testing.T) {
	s := newSerializer()
	f, err := s.Deserialize(clientMessageBytes(t, evalContextMessage, map[string]any{
		"messages": []map[string]any{{"role": "system", "text": "be terse"}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	update, ok := f.(*frames.LLMMessagesUpdateFrame)
	if !ok || len(update.Messages) != 1 || update.Messages[0].Text != "be terse" {
		t.Fatalf("frame = %v, want the context update", f)
	}

	f, err = s.Deserialize(clientMessageBytes(t, evalConfigureMessage, map[string]any{
		"function_call_report_level": map[string]any{"*": "full"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	cf, ok := f.(*rtvi.ConfigureObserverFrame)
	if !ok || cf.FunctionCallReportLevel["*"] != rtvi.ReportFull {
		t.Fatalf("frame = %v, want the full report level for every function", f)
	}
}

// Every other RTVI message goes to the bot's RTVI processor, and what is not
// RTVI is dropped.
func TestOtherMessagesAreForwardedOrDropped(t *testing.T) {
	s := newSerializer()
	for _, msgType := range []string{rtvi.TypeSendText, rtvi.TypeDTMF, rtvi.TypeClientReady} {
		data, _ := json.Marshal(rtvi.Message{Label: rtvi.MessageLabel, Type: msgType, Data: map[string]any{}})
		f, err := s.Deserialize(data)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := f.(*frames.InputTransportMessageFrame); !ok {
			t.Errorf("%s: got %T, want it forwarded as a transport message", msgType, f)
		}
	}
	for name, data := range map[string]string{
		"not RTVI": `{"type": "send-text"}`,
		"not JSON": "not json",
	} {
		if f, _ := s.Deserialize([]byte(data)); f != nil {
			t.Errorf("%s: got %T, want it dropped", name, f)
		}
	}
}

// A client-message the harness does not own goes to the bot's RTVI processor
// like any other RTVI message.
func TestANonEvalClientMessageIsForwarded(t *testing.T) {
	f, err := newSerializer().Deserialize(clientMessageBytes(t, "something-else", map[string]any{}))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := f.(*frames.InputTransportMessageFrame); !ok {
		t.Fatalf("got %T, want an InputTransportMessageFrame", f)
	}
}
