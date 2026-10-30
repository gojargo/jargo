package reader_test

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/internal/onnxrt"
	"github.com/gojargo/jargo/processor/tone/reader"
)

// referenceTags is the reader's tags as the reference implementation, in
// Python, writes them out.
type referenceTags struct {
	Emotion struct {
		Label         string             `json:"label"`
		Intensity     string             `json:"intensity"`
		Confidence    float64            `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"emotion"`
	Move struct {
		Label         string             `json:"label"`
		Confidence    float64            `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"move"`
	Sarcasm struct {
		Sarcastic   bool    `json:"sarcastic"`
		Probability float64 `json:"probability"`
	} `json:"sarcasm"`
	Backchannel struct {
		IsBackchannel bool    `json:"is_backchannel"`
		Probability   float64 `json:"probability"`
	} `json:"backchannel"`
}

func asReference(t reader.Tags) referenceTags {
	var r referenceTags
	r.Emotion.Label, r.Emotion.Intensity, r.Emotion.Confidence = t.Emotion.Label, t.Emotion.Intensity, t.Emotion.Confidence
	r.Emotion.Probabilities = map[string]float64{}
	for _, p := range t.Emotion.Probabilities {
		r.Emotion.Probabilities[p.Label] = p.Probability
	}
	r.Move.Label, r.Move.Confidence = t.Move.Label, t.Move.Confidence
	r.Move.Probabilities = map[string]float64{}
	for _, p := range t.Move.Probabilities {
		r.Move.Probabilities[p.Label] = p.Probability
	}
	r.Sarcasm.Sarcastic, r.Sarcasm.Probability = t.Sarcasm.Sarcastic, t.Sarcasm.Probability
	r.Backchannel.IsBackchannel, r.Backchannel.Probability = t.Backchannel.IsBackchannel, t.Backchannel.Probability
	return r
}

// TestTheModelReadsAsTheReferenceDoes reads the turns the reference read, with
// the released model, and expects the same tags. The model is too large to
// keep here, so the test runs with JARGO_TONE_READER_DIR set to a release's
// "reader" folder, and the ONNX runtime the reference ran with (1.30.0).
func TestTheModelReadsAsTheReferenceDoes(t *testing.T) {
	dir := os.Getenv("JARGO_TONE_READER_DIR")
	if dir == "" {
		t.Skip("set JARGO_TONE_READER_DIR to a reader release's folder")
	}
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	for _, precision := range []reader.Precision{reader.FP16, reader.FP32} {
		t.Run(string(precision), func(t *testing.T) {
			raw, err := os.ReadFile("testdata/reference_" + string(precision) + ".json")
			if err != nil {
				t.Fatal(err)
			}
			var cases []struct {
				Text     string        `json:"text"`
				Previous string        `json:"previous"`
				Tags     referenceTags `json:"tags"`
			}
			if err := json.Unmarshal(raw, &cases); err != nil {
				t.Fatal(err)
			}
			m, err := reader.LoadModel(dir, precision, 1)
			if err != nil {
				t.Fatal(err)
			}
			for _, c := range cases {
				tags, err := m.Tag(c.Text, c.Previous)
				if err != nil {
					t.Fatalf("%q: %v", c.Text, err)
				}
				if got := asReference(tags); !reflect.DeepEqual(got, c.Tags) {
					t.Errorf("%.40q:\n got %+v\nwant %+v", c.Text, got, c.Tags)
				}
			}
		})
	}
}
