package rtvi

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Tests for the diarization speaker in the observer's transcription messages.
// With diarization on, the STT marks each transcription with who said it, and
// the user-input and stt-raw-text messages carry it; user_id is the transport's
// participant, not the speaker.

// speaker is the second speaker the diarization told apart.
//
//nolint:gochecknoglobals // a fixed value, read only
var speaker = &frames.TranscriptionSpeaker{Label: "B", Number: 2}

// transcriptionMessagesFor is the user-input and stt-raw-text messages the
// observer sends for f.
func transcriptionMessagesFor(f frames.Frame) []Message {
	o := rawTextObserver()
	o.push(f)
	var out []Message
	for _, m := range o.sent {
		if m.Type == TypeUserInput || m.Type == TypeSTTRawText {
			out = append(out, m)
		}
	}
	return out
}

// speakerOf is the speaker a user-input or stt-raw-text message carries.
func speakerOf(t *testing.T, m Message) *frames.TranscriptionSpeaker {
	t.Helper()
	switch d := m.Data.(type) {
	case UserInputData:
		return d.Speaker
	case STTRawTextData:
		return d.Speaker
	default:
		t.Fatalf("%s data = %T, want a transcription's", m.Type, m.Data)
		return nil
	}
}

// wireData is a message's data as it goes on the wire.
func wireData(t *testing.T, m Message) map[string]any {
	t.Helper()
	encoded, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Data map[string]any `json:"data"`
	}
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Data
}

func TestADiarizedTranscriptionCarriesItsSpeaker(t *testing.T) {
	f := frames.NewTranscriptionFrame("Mine is the banana.", "", "now")
	f.Speaker = speaker
	msgs := transcriptionMessagesFor(f)

	if len(msgs) != 2 {
		t.Fatalf("sent %+v, want a user-input and an stt-raw-text", msgs)
	}
	want := map[string]any{"label": "B", "number": float64(2)}
	for _, m := range msgs {
		if got := wireData(t, m)["speaker"]; !reflect.DeepEqual(got, want) {
			t.Errorf("%s speaker = %v, want %v", m.Type, got, want)
		}
	}
}

func TestAnInterimTranscriptionCarriesItsSpeaker(t *testing.T) {
	f := frames.NewInterimTranscriptionFrame("Mine is", "", "now")
	f.Speaker = speaker
	msgs := transcriptionMessagesFor(f)

	if len(msgs) != 2 {
		t.Fatalf("sent %+v, want a user-input and an stt-raw-text", msgs)
	}
	for _, m := range msgs {
		if final := wireData(t, m)["final"]; final != false {
			t.Errorf("%s final = %v, want false", m.Type, final)
		}
		if got := speakerOf(t, m); got != speaker {
			t.Errorf("%s speaker = %v, want %v", m.Type, got, speaker)
		}
	}
}

func TestATranscriptionWithoutASpeakerSendsNone(t *testing.T) {
	msgs := transcriptionMessagesFor(frames.NewTranscriptionFrame("Hello.", "", "now"))

	if len(msgs) != 2 {
		t.Fatalf("sent %+v, want a user-input and an stt-raw-text", msgs)
	}
	for _, m := range msgs {
		if got := speakerOf(t, m); got != nil {
			t.Errorf("%s speaker = %v, want none", m.Type, got)
		}
		if _, ok := wireData(t, m)["speaker"]; ok {
			t.Errorf("%s wire data carries a speaker, want it left out", m.Type)
		}
	}
}
