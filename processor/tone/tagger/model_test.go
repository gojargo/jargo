package tagger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/adapter/tts/elevenlabs"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/internal/onnxrt"
)

// referenceLabels is a sentence's labels as the reference implementation, in
// Python, writes them out.
type referenceLabels struct {
	Delivery    *string             `json:"delivery"`
	Spans       [][]json.RawMessage `json:"spans"`
	Emphasis    [][2]int            `json:"emphasis"`
	SoundBefore *string             `json:"sound_before"`
	SoundAfter  *string             `json:"sound_after"`
}

func (r referenceLabels) labels(t *testing.T) SentenceLabels {
	t.Helper()
	text := func(s *string) string {
		if s == nil {
			return ""
		}
		return *s
	}
	l := SentenceLabels{Delivery: text(r.Delivery), SoundBefore: text(r.SoundBefore), SoundAfter: text(r.SoundAfter)}
	for _, raw := range r.Spans {
		var s Span
		if len(raw) != 3 || json.Unmarshal(raw[0], &s.Start) != nil || json.Unmarshal(raw[1], &s.End) != nil ||
			json.Unmarshal(raw[2], &s.Delivery) != nil {
			t.Fatalf("span %s", raw)
		}
		l.Spans = append(l.Spans, s)
	}
	if len(r.Emphasis) > 0 {
		l.Emphasis = r.Emphasis
	}
	return l
}

// referenceCase is a turn and a sentence the reference read: what it wrote for
// the model, and the labels each size chose.
type referenceCase struct {
	Name     string `json:"name"`
	Messages []struct {
		Role  string `json:"role"`
		Text  string `json:"text"`
		Heard []struct {
			Name   string   `json:"name"`
			Labels [][2]any `json:"labels"`
			Value  *float64 `json:"value"`
		} `json:"heard"`
		Speculation bool `json:"speculation"`
	} `json:"messages"`
	Sentence    string              `json:"sentence"`
	Earlier     []string            `json:"earlier"`
	TTS         string              `json:"tts"`
	ServiceName string              `json:"service_name"`
	Guidance    []map[string]string `json:"guidance"`
	Delivery    []string            `json:"delivery"`
	Sounds      []string            `json:"sounds"`
	Vocabulary  map[string][]string `json:"vocabulary"`
	Previous    []referenceLabels   `json:"previous"`
	Layout      json.RawMessage     `json:"layout"`
	Encoded     struct {
		Call  string `json:"call"`
		Text  string `json:"text"`
		Start int    `json:"start"`
	} `json:"encoded"`
	IDs           []int64                    `json:"ids"`
	AttentionMask []int64                    `json:"attention_mask"`
	SentenceMask  []int64                    `json:"sentence_mask"`
	Offsets       []*[2]int                  `json:"offsets"`
	Labels        map[string]referenceLabels `json:"labels"`
}

func referenceCases(t *testing.T) []referenceCase {
	t.Helper()
	raw, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []referenceCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

// turn is the case's turn, built as the reference built it.
func (c referenceCase) turn(t *testing.T) (TurnContext, layout) {
	t.Helper()
	presets := map[string]expressive.Guidance{
		"MATCH_STYLE":      expressive.GuidanceMatchStyle,
		"SLOW_FOR_DETAILS": expressive.GuidanceSlowForDetails,
		"BREATHE":          expressive.GuidanceBreathe,
	}
	var policy expressive.ExpressionPolicy
	for _, g := range c.Guidance {
		if preset, ok := g["preset"]; ok {
			policy.Guidance = append(policy.Guidance, presets[preset])
		} else {
			policy.Guidance = append(policy.Guidance, expressive.GuidanceLine(g["line"]))
		}
	}
	if c.Delivery != nil {
		policy.Delivery = []expressive.ExpressionDelivery{}
		for _, d := range c.Delivery {
			policy.Delivery = append(policy.Delivery, expressive.ExpressionDelivery(d))
		}
	}
	if c.Sounds != nil {
		policy.Sounds = []expressive.ExpressionSound{}
		for _, s := range c.Sounds {
			policy.Sounds = append(policy.Sounds, expressive.ExpressionSound(s))
		}
	}
	capabilities := map[string]*expressive.ExpressionCapabilities{
		"elevenlabs": elevenlabs.ElevenLabsAudioTagExpressionAdapter{}.Capabilities(),
		"cartesia":   cartesia.CartesiaExpressionAdapter{}.Capabilities(),
	}[c.TTS]
	turn := TurnContext{Space: ResolveLabelSpace(capabilities, policy, c.ServiceName, c.Vocabulary), Policy: policy}
	for _, m := range c.Messages {
		var heard []expressive.PerceptionSignal
		if m.Heard != nil {
			heard = []expressive.PerceptionSignal{}
			for _, s := range m.Heard {
				signal := expressive.PerceptionSignal{Name: s.Name, Value: s.Value}
				for _, l := range s.Labels {
					label, _ := l[0].(string)
					confidence, _ := l[1].(float64)
					signal.Labels = append(signal.Labels, expressive.PerceptionLabel{Label: label, Confidence: confidence})
				}
				heard = append(heard, signal)
			}
			heard = fold(heard)
		}
		turn.Messages = append(turn.Messages, Message{
			Role: m.Role, Text: m.Text, Heard: heard, Speculation: m.Speculation,
		})
	}
	if c.Previous != nil {
		turn.Previous = []SentenceLabels{}
		for _, p := range c.Previous {
			turn.Previous = append(turn.Previous, p.labels(t))
		}
	}
	l, err := parseLayout(c.Layout)
	if err != nil {
		t.Fatal(err)
	}
	return turn, l
}

// TestASentenceIsEncodedAsTheReferenceEncodesIt writes the text pair for each
// turn the reference read, with the layout it read it with, and expects the
// reference's text: the call, the header with its state hints, the reply so
// far and the sentence, and where the sentence starts.
func TestASentenceIsEncodedAsTheReferenceEncodesIt(t *testing.T) {
	for _, c := range referenceCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			turn, l := c.turn(t)
			call, text, start := encode(turn, c.Earlier, c.Sentence, l)
			if call != c.Encoded.Call {
				t.Errorf("call\n got %q\nwant %q", call, c.Encoded.Call)
			}
			if text != c.Encoded.Text {
				t.Errorf("text\n got %q\nwant %q", text, c.Encoded.Text)
			}
			if start != c.Encoded.Start {
				t.Errorf("start %d, want %d", start, c.Encoded.Start)
			}
		})
	}
}

// TestTheModelTagsAsTheReferenceDoes labels the sentences the reference
// labeled, with the released models of both sizes, and expects the same
// tokens and the same labels. The cases with a layout that cuts the call first
// read the call that way, as the reference did. The models are too large to
// keep here, so the test runs with JARGO_TONE_TAGGER_DIR set to a release's
// "tagger" folder, holding 17m and 32m, and the ONNX runtime the reference ran
// with (1.30.0).
func TestTheModelTagsAsTheReferenceDoes(t *testing.T) {
	dir := os.Getenv("JARGO_TONE_TAGGER_DIR")
	if dir == "" {
		t.Skip("set JARGO_TONE_TAGGER_DIR to a tagger release's folder")
	}
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	cases := referenceCases(t)
	for _, size := range []Size{Size17M, Size32M} {
		t.Run(string(size), func(t *testing.T) {
			m, err := LoadModel(filepath.Join(dir, string(size)), 1)
			if err != nil {
				t.Fatal(err)
			}
			released := defaultLayout()
			released.PreviousTags = true
			if m.layout != released {
				t.Fatalf("layout %+v, want tagger-rc1's %+v", m.layout, released)
			}
			for _, c := range cases {
				if c.Labels == nil {
					continue
				}
				turn, l := c.turn(t)
				m.layout = l
				ids, _, mask, _, err := m.encode(turn, c.Earlier, c.Sentence)
				if err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				if !slices.Equal(ids, c.IDs) || !slices.Equal(mask, c.SentenceMask) {
					t.Errorf("%s: tokens\n got %v %v\nwant %v %v", c.Name, ids, mask, c.IDs, c.SentenceMask)
					continue
				}
				got, err := m.Tag(turn, c.Earlier, c.Sentence)
				if err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				if want := c.Labels[string(size)].labels(t); !reflect.DeepEqual(got, want) {
					t.Errorf("%s:\n got %+v\nwant %+v", c.Name, got, want)
				}
			}
		})
	}
}

// TestTheModelLabelsTheReferencesTokensAsTheReferenceDoes runs the released
// models of both sizes on the tokens the reference tokenized each sentence
// into, and expects the reference's labels: the session, the heads and their
// decoding, without the tokenizer. It runs with JARGO_TONE_TAGGER_DIR set, as
// TestTheModelTagsAsTheReferenceDoes does.
func TestTheModelLabelsTheReferencesTokensAsTheReferenceDoes(t *testing.T) {
	dir := os.Getenv("JARGO_TONE_TAGGER_DIR")
	if dir == "" {
		t.Skip("set JARGO_TONE_TAGGER_DIR to a tagger release's folder")
	}
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	cases := referenceCases(t)
	for _, size := range []Size{Size17M, Size32M} {
		t.Run(string(size), func(t *testing.T) {
			m := sessionOnly(t, filepath.Join(dir, string(size)))
			for _, c := range cases {
				if c.Labels == nil {
					continue
				}
				turn, _ := c.turn(t)
				offsets := make([]offset, len(c.Offsets))
				for i, o := range c.Offsets {
					if o != nil {
						offsets[i] = offset{o[0], o[1], true}
					}
				}
				shape := []int64{1, int64(len(c.IDs))}
				results, err := m.session.Run([]onnxrt.Tensor{
					onnxrt.Int64(shape, c.IDs), onnxrt.Int64(shape, c.AttentionMask), onnxrt.Int64(shape, c.SentenceMask),
				})
				if err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				h, err := readHeads(results)
				if err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				got, err := m.decode(h, offsets, c.Sentence, turn.Space)
				if err != nil {
					t.Fatalf("%s: %v", c.Name, err)
				}
				if want := c.Labels[string(size)].labels(t); !reflect.DeepEqual(got, want) {
					t.Errorf("%s:\n got %+v\nwant %+v", c.Name, got, want)
				}
			}
		})
	}
}

// sessionOnly is the model in dir with its config and session, and no
// tokenizer.
func sessionOnly(t *testing.T, dir string) *Model {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "config.json")) //nolint:gosec // the test's own folder
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{}
	if err := json.Unmarshal(raw, &m.config); err != nil {
		t.Fatal(err)
	}
	weights, err := os.ReadFile(filepath.Join(dir, m.config.ONNX)) //nolint:gosec // the test's own folder
	if err != nil {
		t.Fatal(err)
	}
	m.session, err = onnxrt.NewWithOptions(weights, inputNames, outputNames, onnxrt.Options{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	return m
}
