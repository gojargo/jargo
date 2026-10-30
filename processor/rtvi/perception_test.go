package rtvi_test

// The reading of how the user spoke in the observer's user-input messages.

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
)

func reading() []expressive.PerceptionSignal {
	rate := 4.1
	return []expressive.PerceptionSignal{
		{Name: "tone", Labels: []expressive.PerceptionLabel{
			{Label: "frustrated", Confidence: 0.86}, {Label: "neutral", Confidence: 0.10},
		}},
		{Name: "speech_rate", Value: &rate},
	}
}

// pushTo reports one push of f from source to destination.
func (h *pushHarness) pushTo(f frames.Frame, source, destination processor.Processor, firstPush bool) {
	h.observer.OnPushFrame(processor.FramePushed{
		Source:      source,
		Destination: destination,
		Frame:       f,
		Direction:   processor.Downstream,
		FirstPush:   firstPush,
	})
}

// userInputs is the data of the user-input messages sent so far, once they
// have settled.
func (h *pushHarness) userInputs() []rtvi.UserInputData {
	time.Sleep(100 * time.Millisecond)
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []rtvi.UserInputData
	for _, m := range h.msgs {
		if d, ok := m.Data.(rtvi.UserInputData); ok && m.Type == rtvi.TypeUserInput {
			out = append(out, d)
		}
	}
	return out
}

func TestAReadingTravelsWithItsTranscription(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	f := frames.NewInterimTranscriptionFrame("Can you tell me", "", "now")
	f.Perception = reading()
	h.push(f, true, nil)

	sent := h.userInputs()
	if len(sent) != 1 {
		t.Fatalf("sent %d user-input messages, want 1", len(sent))
	}
	if sent[0].Final {
		t.Error("an interim transcription was reported final")
	}
	got, err := json.Marshal(sent[0].Perception)
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"name":"tone","labels":[{"label":"frustrated","confidence":0.86},` +
		`{"label":"neutral","confidence":0.1}]},{"name":"speech_rate","labels":[],"value":4.1}]`
	if string(got) != want {
		t.Errorf("perception = %s, want %s", got, want)
	}
}

func TestATranscriptionWithoutAReadingHasNoPerception(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	h.push(frames.NewTranscriptionFrame("Where's my order?", "", "now"), true, nil)

	sent := h.userInputs()
	if len(sent) != 1 || sent[0].Perception != nil {
		t.Fatalf("sent %+v, want one user-input without perception", sent)
	}
	raw, err := json.Marshal(sent[0])
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	if _, ok := fields["perception"]; ok {
		t.Errorf("the message carries a perception field: %s", raw)
	}
}

// A transcription waits for a perceiver after the STT.

func announceReader(h *pushHarness, stt, reader, aggregator processor.Processor) {
	metadata := frames.NewPerceptionMetadataFrame("reader", expressive.PerceptionCapabilities{})
	h.pushTo(metadata, reader, aggregator, true)
	// Passed along, it does not make the next processor a perceiver.
	h.pushTo(metadata, aggregator, stt, false)
}

func TestATranscriptionIsReportedOnceRead(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	stt, reader, aggregator := processor.New("stt", nil), processor.New("reader", nil), processor.New("aggregator", nil)
	announceReader(h, stt, reader, aggregator)
	f := frames.NewTranscriptionFrame("Oh great, it's late.", "", "now")

	h.pushTo(f, stt, reader, true)
	if sent := h.userInputs(); len(sent) != 0 {
		t.Fatalf("reported %+v before the reader read it", sent)
	}

	f.Perception = reading()
	h.pushTo(f, reader, aggregator, false)
	h.pushTo(f, aggregator, stt, false)

	sent := h.userInputs()
	if len(sent) != 1 || sent[0].Text != "Oh great, it's late." {
		t.Fatalf("sent %+v, want the transcription once", sent)
	}
	if len(sent[0].Perception) != 2 || sent[0].Perception[0].Name != "tone" ||
		sent[0].Perception[1].Name != "speech_rate" {
		t.Errorf("perception = %+v, want the reader's", sent[0].Perception)
	}
}

func TestWithoutAPerceiverATranscriptionIsReportedWhenFirstPushed(t *testing.T) {
	h := newPushHarness(t, rtvi.DefaultObserverParams())
	stt, reader, aggregator := processor.New("stt", nil), processor.New("reader", nil), processor.New("aggregator", nil)
	f := frames.NewTranscriptionFrame("Where's my order?", "", "now")

	h.pushTo(f, stt, reader, true)
	h.pushTo(f, reader, aggregator, false)

	if sent := h.userInputs(); len(sent) != 1 {
		t.Fatalf("sent %d user-input messages, want 1", len(sent))
	}
}
