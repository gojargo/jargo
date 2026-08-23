package gladia

import (
	"testing"

	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/stt"
)

// speechMessage is one of the boundary messages Gladia sends when its own
// detection is switched on.
func speechMessage(kind string) message { return message{Type: kind} }

// TestSpeechBoundariesReachThePipeline covers Gladia's own detection driving the
// turn. Its speech messages become the boundaries the pipeline acts on.
//
// Whether the user starting to talk barges in on the bot is not decided here: it
// is carried by the turn strategies this service recommends, which is what
// TestMetadataDefersTheTurnWhenGladiaDetects covers.
func TestSpeechBoundariesReachThePipeline(t *testing.T) {
	s := &stream{vad: true}

	got, ok := s.result(speechMessage(msgSpeechStart))
	if !ok || got.Speech != stt.SpeechStarted {
		t.Fatalf("speech start mapped to %+v (kept %v), want a started boundary", got, ok)
	}

	got, ok = s.result(speechMessage(msgSpeechEnd))
	if !ok || got.Speech != stt.SpeechStopped {
		t.Fatalf("speech end mapped to %+v (kept %v), want a stopped boundary", got, ok)
	}
}

// TestSpeechBoundariesAreIgnoredWhenThePipelineDetects covers the default. The
// pipeline runs its own detection, so a boundary from here would compete with it
// and open turns twice over.
func TestSpeechBoundariesAreIgnoredWhenThePipelineDetects(t *testing.T) {
	s := &stream{vad: false}

	for _, kind := range []string{msgSpeechStart, msgSpeechEnd} {
		if got, ok := s.result(speechMessage(kind)); ok {
			t.Errorf("%s was acted on with the pipeline detecting: %+v", kind, got)
		}
	}
}

// TestTranscriptsAreUnaffectedByTheDetectionSetting covers the other messages:
// whichever side detects speech, the transcripts themselves are read the same.
func TestTranscriptsAreUnaffectedByTheDetectionSetting(t *testing.T) {
	var m message
	m.Type = msgTranscript
	m.Data.IsFinal = true
	m.Data.Utterance.Text = "hello there"
	m.Data.Utterance.Language = "en"

	for _, vad := range []bool{false, true} {
		s := &stream{vad: vad}
		got, ok := s.result(m)
		if !ok {
			t.Fatalf("the transcript was dropped with detection set to %v", vad)
		}
		if got.Text != "hello there" || !got.Final || !got.EndOfTurn || got.Language != "en" {
			t.Errorf("transcript mapped to %+v", got)
		}
		if got.Speech != stt.SpeechUnknown {
			t.Errorf("a transcript reported the speech boundary %v", got.Speech)
		}
	}
}

// TestMetadataDefersTheTurnWhenGladiaDetects covers what downstream is told.
// With Gladia's detection driving the turn the aggregator is asked to adopt
// external strategies, so it defers to the boundaries the service reports rather
// than running its own detection alongside them.
func TestMetadataDefersTheTurnWhenGladiaDetects(t *testing.T) {
	on := &connector{cfg: withDefaults(Config{APIKey: "k", EnableVAD: true})}
	got, ok := on.Metadata().UserTurnStrategies.(turns.UserTurnStrategies)
	if !ok {
		t.Fatalf("recommended turns = %T, want the external strategies", on.Metadata().UserTurnStrategies)
	}
	interrupts, external := got.ExternalInterruptions()
	if !external {
		t.Error("the recommended strategies are not the external ones")
	}
	if !interrupts {
		t.Error("the recommendation should barge in by default")
	}

	// A caller who wants Gladia's turn detection without the barge-in carries
	// that through to the strategies, which own the interruption.
	quiet := &connector{cfg: withDefaults(Config{APIKey: "k", EnableVAD: true, InterruptOnSpeech: new(bool)})}
	q, ok := quiet.Metadata().UserTurnStrategies.(turns.UserTurnStrategies)
	if !ok {
		t.Fatalf("recommended turns = %T, want the external strategies", quiet.Metadata().UserTurnStrategies)
	}
	if interrupts, _ := q.ExternalInterruptions(); interrupts {
		t.Error("the recommendation barges in with the barge-in turned off")
	}

	off := &connector{cfg: withDefaults(Config{APIKey: "k"})}
	if off.Metadata().UserTurnStrategies != nil {
		t.Error("external strategies were recommended with the pipeline detecting")
	}
}

// TestSettingsAskForSpeechEventsWhenGladiaDetects covers the session those
// boundaries have to arrive on. The default message filter asks only for
// transcripts, so switching detection on has to ask for the speech messages too
// or none of them are ever sent.
func TestSettingsAskForSpeechEventsWhenGladiaDetects(t *testing.T) {
	got := settingsOf(t, Config{APIKey: "k", EnableVAD: true}, 16000)

	mc, ok := got["messages_config"].(map[string]any)
	if !ok {
		t.Fatalf("messages_config = %v, want the filter", got["messages_config"])
	}
	if mc["receive_speech_events"] != true {
		t.Errorf("receive_speech_events = %v, want it requested: without it Gladia sends no boundaries",
			mc["receive_speech_events"])
	}
}

// TestSettingsLeaveACallersSpeechEventChoiceAlone covers a caller who set the
// field themselves. Their choice stands, including turning the messages off
// while still acting on any that arrive.
func TestSettingsLeaveACallersSpeechEventChoiceAlone(t *testing.T) {
	off := false
	got := settingsOf(t, Config{
		APIKey:         "k",
		EnableVAD:      true,
		MessagesConfig: &MessagesConfig{ReceiveSpeechEvents: &off},
	}, 16000)

	mc, ok := got["messages_config"].(map[string]any)
	if !ok {
		t.Fatalf("messages_config = %v, want the caller's filter", got["messages_config"])
	}
	if mc["receive_speech_events"] != false {
		t.Errorf("receive_speech_events = %v, want the caller's false", mc["receive_speech_events"])
	}
}

// TestSettingsOmitSpeechEventsWhenThePipelineDetects covers the default session:
// nothing acts on the boundaries, so there is no reason to ask for them.
func TestSettingsOmitSpeechEventsWhenThePipelineDetects(t *testing.T) {
	got := settingsOf(t, Config{APIKey: "k"}, 16000)

	mc, ok := got["messages_config"].(map[string]any)
	if !ok {
		t.Fatalf("messages_config = %v, want the default filter", got["messages_config"])
	}
	if _, present := mc["receive_speech_events"]; present {
		t.Errorf("receive_speech_events = %v, want it left out", mc["receive_speech_events"])
	}
}
