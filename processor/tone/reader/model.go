package reader

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/gojargo/jargo/internal/hfhub"
	"github.com/gojargo/jargo/internal/onnxrt"
	"github.com/gojargo/jargo/internal/tokenizers"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

const (
	// Repo is the Hugging Face repository the tone models are released in.
	Repo = "pipecat-ai/tone-models"
	// Revision is the reader's release the reader downloads by default.
	Revision = "reader-rc2"

	folder = "reader"
)

// Precision is the weights a model is loaded with.
type Precision string

const (
	// FP16 is the weights in 16 bits, which give the same tags at half the size.
	FP16 Precision = "fp16"
	// FP32 is the weights in 32 bits.
	FP32 Precision = "fp32"
)

//nolint:gochecknoglobals // the file each precision is released as
var modelFiles = map[Precision]string{FP16: "model_fp16.onnx", FP32: "model.opt.onnx"}

// ErrPrecision is returned for a precision the model is not released in.
var ErrPrecision = errors.New("reader: unknown precision")

// modelConfig is the part of the model's config.json the reader reads.
type modelConfig struct {
	MaxLen      int      `json:"max_len"`
	Outputs     []string `json:"outputs"`
	Emotions    []string `json:"emotions"`
	Intensities []string `json:"intensities"`
	Moves       []string `json:"moves"`
	Calibration *struct {
		EmotionTemperature *float64 `json:"emotion_temperature"`
		MoveTemperature    *float64 `json:"move_temperature"`
		Sarcasm            *platt   `json:"sarcasm"`
	} `json:"calibration"`
	SarcasmTauRaw *float64 `json:"sarcasm_tau_raw"`
}

// platt is a calibration curve, sigmoid(a * logit(p) + b).
type platt struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

// Model is the reader's model, on the CPU.
type Model struct {
	config             modelConfig
	tokenizer          *tokenizers.Tokenizer
	truncation         *tokenizers.Truncation
	session            *onnxrt.Session
	emotionTemperature float64
	moveTemperature    float64
	sarcasmCurve       *platt
	sarcasmThreshold   float64
}

// LoadModel loads the model from a folder holding its config.json,
// tokenizer.json and ONNX file, using threads CPU threads for one read.
func LoadModel(dir string, precision Precision, threads int) (*Model, error) {
	file, ok := modelFiles[precision]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrPrecision, precision)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json")) //nolint:gosec // the model's own folder
	if err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	m := &Model{emotionTemperature: 1, moveTemperature: 1, sarcasmThreshold: 0.5}
	if err = json.Unmarshal(raw, &m.config); err != nil {
		return nil, fmt.Errorf("reader: config.json: %w", err)
	}
	if m.tokenizer, err = tokenizers.FromFile(filepath.Join(dir, "tokenizer.json")); err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	// Truncated at the model's length, from the end, as the tokenizer's
	// defaults do.
	m.truncation = &tokenizers.Truncation{MaxLength: m.config.MaxLen}
	weights, err := os.ReadFile(filepath.Join(dir, file)) //nolint:gosec // the model's own folder
	if err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	m.session, err = onnxrt.NewWithOptions(weights, []string{"input_ids", "attention_mask"}, m.config.Outputs,
		onnxrt.Options{IntraOpThreads: threads})
	if err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	if c := m.config.Calibration; c != nil {
		if c.EmotionTemperature != nil {
			m.emotionTemperature = *c.EmotionTemperature
		}
		if c.MoveTemperature != nil {
			m.moveTemperature = *c.MoveTemperature
		}
		m.sarcasmCurve = c.Sarcasm
	}
	if m.config.SarcasmTauRaw != nil {
		m.sarcasmThreshold = *m.config.SarcasmTauRaw
	}
	return m, nil
}

// DownloadModel loads the model's release from Hugging Face, downloading its
// files into cacheDir the first time. An empty cacheDir is Hugging Face's own
// cache.
func DownloadModel(
	ctx context.Context, revision string, precision Precision, threads int, cacheDir string,
) (*Model, error) {
	file, ok := modelFiles[precision]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrPrecision, precision)
	}
	root, err := download(ctx, hfhub.Config{
		Repo:     Repo,
		Revision: revision,
		Files:    []string{folder + "/config.json", folder + "/tokenizer.json", folder + "/" + file},
		CacheDir: cacheDir,
	})
	if err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	return LoadModel(filepath.Join(root, folder), precision, threads)
}

// download fetches a release's files. A variable so tests can see what is
// asked for.
//
//nolint:gochecknoglobals // a seam for tests
var download = hfhub.Download

// Emotions is the emotions the model reads.
func (m *Model) Emotions() []string { return m.config.Emotions }

// Probability is one label's probability.
type Probability struct {
	Label       string
	Probability float64
}

// Tags is what the model reads of one turn.
type Tags struct {
	Emotion     EmotionTag
	Move        MoveTag
	Sarcasm     SarcasmTag
	Backchannel BackchannelTag
}

// EmotionTag is the feeling the turn's words show.
type EmotionTag struct {
	Label     string
	Intensity string
	// Confidence is the likeliest label's probability.
	Confidence float64
	// Probabilities is every emotion's probability, in the model's order.
	Probabilities []Probability
}

// MoveTag is what the turn does, such as asking or answering.
type MoveTag struct {
	Label         string
	Confidence    float64
	Probabilities []Probability
}

// SarcasmTag is whether the turn is sarcastic.
type SarcasmTag struct {
	Sarcastic bool
	// Probability is the calibrated chance it is sarcastic.
	Probability float64
}

// BackchannelTag is whether the turn is only a listening sound.
type BackchannelTag struct {
	IsBackchannel bool
	Probability   float64
}

// Tag reads one turn, with the turn before it when there is one.
func (m *Model) Tag(text, previous string) (Tags, error) {
	rendered := `"` + shared.Strip(text) + `"`
	if p := shared.Strip(previous); p != "" {
		rendered += ` Before: "` + p + `"`
	}
	enc, err := m.tokenizer.Encode(rendered, m.truncation)
	if err != nil {
		return Tags{}, fmt.Errorf("reader: %w", err)
	}
	shape := []int64{1, int64(len(enc.IDs))}
	results, err := m.session.Run([]onnxrt.Tensor{onnxrt.Int64(shape, enc.IDs), onnxrt.Int64(shape, enc.AttentionMask)})
	if err != nil {
		return Tags{}, fmt.Errorf("reader: %w", err)
	}
	outputs := make(map[string][]float32, len(results))
	for i, r := range results {
		outputs[m.config.Outputs[i]] = r.F32
	}

	emotions := softmax(outputs["emotion"], m.emotionTemperature)
	moves := softmax(outputs["move"], m.moveTemperature)
	emotion := m.config.Emotions[argmax(emotions)]
	intensity := m.config.Intensities[argmax(softmax(outputs["emotion_intensity"], 1))]
	if emotion == "neutral" {
		intensity = "low"
	}
	rawSarcasm := softmax(outputs["sarcasm"], 1)[1]
	backchannel := softmax(outputs["backchannel"], 1)[1]
	return Tags{
		Emotion: EmotionTag{
			Label:         emotion,
			Intensity:     intensity,
			Confidence:    shared.Round(emotions[argmax(emotions)], 4),
			Probabilities: probabilities(m.config.Emotions, emotions),
		},
		Move: MoveTag{
			Label:         m.config.Moves[argmax(moves)],
			Confidence:    shared.Round(moves[argmax(moves)], 4),
			Probabilities: probabilities(m.config.Moves, moves),
		},
		Sarcasm: SarcasmTag{
			Sarcastic:   rawSarcasm >= m.sarcasmThreshold,
			Probability: shared.Round(calibrated(rawSarcasm, m.sarcasmCurve), 4),
		},
		Backchannel: BackchannelTag{
			IsBackchannel: backchannel >= 0.5,
			Probability:   shared.Round(backchannel, 4),
		},
	}, nil
}

// softmax is the probabilities of the logits at a temperature, worked out in
// 64 bits.
func softmax(logits []float32, temperature float64) []float64 {
	z := make([]float64, len(logits))
	top := math.Inf(-1)
	for i, l := range logits {
		z[i] = float64(l) / temperature
		top = max(top, z[i])
	}
	var sum float64
	for i := range z {
		z[i] = math.Exp(z[i] - top)
		sum += z[i]
	}
	for i := range z {
		z[i] /= sum
	}
	return z
}

// argmax is the index of the largest value, the first of several equal ones.
func argmax(values []float64) int {
	best := 0
	for i, v := range values {
		if v > values[best] {
			best = i
		}
	}
	return best
}

func probabilities(labels []string, probs []float64) []Probability {
	out := make([]Probability, 0, min(len(labels), len(probs)))
	for i := range min(len(labels), len(probs)) {
		out = append(out, Probability{Label: labels[i], Probability: shared.Round(probs[i], 4)})
	}
	return out
}

// calibrated is the chance the curve gives for p, or p without a curve.
func calibrated(p float64, curve *platt) float64 {
	if curve == nil {
		return p
	}
	p = min(max(p, 1e-6), 1-1e-6)
	z := curve.A*math.Log(p/(1-p)) + curve.B
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	return math.Exp(z) / (1 + math.Exp(z))
}
