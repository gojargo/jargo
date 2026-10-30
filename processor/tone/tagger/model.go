package tagger

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/internal/hfhub"
	"github.com/gojargo/jargo/internal/onnxrt"
	"github.com/gojargo/jargo/internal/tokenizers"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
)

const (
	// Repo is the Hugging Face repository the tone models are released in.
	Repo = "pipecat-ai/tone-models"
	// Revision is the tagger's release the tagger downloads by default.
	Revision = "tagger-rc1"
)

// Size is a size the model is released in.
type Size string

const (
	// Size17M is the smaller model, the default.
	Size17M Size = "17m"
	// Size32M is the larger model, which agrees with its teacher a little more
	// often and takes nearly three times as long a sentence.
	Size32M Size = "32m"
)

//nolint:gochecknoglobals // the model's inputs and outputs, in its order
var (
	inputNames  = []string{"input_ids", "attention_mask", "sentence_mask"}
	outputNames = []string{"delivery", "sound_before", "sound_after", "spans", "emphasis", "breath"}
)

// A span or stressed word's tokens, as the model classes them.
const (
	outside = iota
	begin
	inside
)

var (
	errConfig  = errors.New("tagger: config.json")
	errOutputs = errors.New("tagger: the model's outputs are not the shape it was released with")
	errOffset  = errors.New("tagger: a token's offsets are outside the sentence")
)

// modelConfig is the part of the model's config.json the tagger reads.
type modelConfig struct {
	Layout          json.RawMessage   `json:"layout"`
	Deliveries      []string          `json:"deliveries"`
	Sounds          []string          `json:"sounds"`
	ONNX            string            `json:"onnx"`
	BreathsPerReply *int              `json:"breaths_per_reply"`
	BreathKinds     breathKinds       `json:"breath_kinds"`
	SoundBias       map[string][]step `json:"sound_bias"`
}

// BreathKind is a kind a breath is drawn as, with its share.
type BreathKind struct {
	Kind  string
	Share float64
}

// breathKinds is the kinds in the order config.json lists them, the order they
// are drawn in.
type breathKinds []BreathKind

func (b *breathKinds) UnmarshalJSON(raw []byte) error {
	*b = nil
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil || tok == nil {
		return err
	}
	if tok != json.Delim('{') {
		return fmt.Errorf("%w: breath_kinds is not an object", errConfig)
	}
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return err
		}
		var share float64
		if err := dec.Decode(&share); err != nil {
			return err
		}
		kind, _ := key.(string)
		// A key given twice keeps its first place and its last share, as a
		// JSON object read into a dict does.
		if i := slices.IndexFunc(*b, func(k BreathKind) bool { return k.Kind == kind }); i >= 0 {
			(*b)[i].Share = share
			continue
		}
		*b = append(*b, BreathKind{kind, share})
	}
	return nil
}

// step is a sound's calibration for sentences of at most most words, or of
// any number where most is nil.
type step struct {
	most  *float64
	shift float64
}

func (s *step) UnmarshalJSON(raw []byte) error {
	var pair []json.RawMessage
	if err := json.Unmarshal(raw, &pair); err != nil {
		return err
	}
	if len(pair) != 2 {
		return fmt.Errorf("%w: a sound_bias step is not [most words, shift]", errConfig)
	}
	if err := json.Unmarshal(pair[0], &s.most); err != nil {
		return err
	}
	return json.Unmarshal(pair[1], &s.shift)
}

// Model is the tagger's model, on the CPU. One model can serve every call.
type Model struct {
	config    modelConfig
	layout    layout
	tokenizer *tokenizers.Tokenizer
	session   *onnxrt.Session
}

// LoadModel loads the model from a folder holding its config.json,
// tokenizer.json and ONNX file, using threads CPU threads for one sentence.
func LoadModel(dir string, threads int) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json")) //nolint:gosec // the model's own folder
	if err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	m := &Model{}
	if err = json.Unmarshal(raw, &m.config); err != nil {
		return nil, fmt.Errorf("%w: %w", errConfig, err)
	}
	if m.config.Layout == nil {
		return nil, fmt.Errorf("%w: no layout", errConfig)
	}
	if m.layout, err = parseLayout(m.config.Layout); err != nil {
		return nil, fmt.Errorf("%w: %w", errConfig, err)
	}
	if m.tokenizer, err = tokenizers.FromFile(filepath.Join(dir, "tokenizer.json")); err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	if m.config.ONNX == "" {
		return nil, fmt.Errorf("%w: no onnx file", errConfig)
	}
	weights, err := os.ReadFile(filepath.Join(dir, m.config.ONNX)) //nolint:gosec // the model's own folder
	if err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	m.session, err = onnxrt.NewWithOptions(weights, inputNames, outputNames, onnxrt.Options{IntraOpThreads: threads})
	if err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	return m, nil
}

// DownloadModel loads the model of a size from a release on Hugging Face,
// downloading the files of its folder, "tagger/<size>", into cacheDir the
// first time. An empty cacheDir is Hugging Face's own cache.
//
// The folder holds config.json, tokenizer.json and the ONNX file config.json
// names, so config.json is fetched first.
func DownloadModel(ctx context.Context, size Size, revision string, threads int, cacheDir string) (*Model, error) {
	folder := "tagger/" + string(size)
	cfg := hfhub.Config{Repo: Repo, Revision: revision, Files: []string{folder + "/config.json"}, CacheDir: cacheDir}
	root, err := download(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(folder), "config.json")) //nolint:gosec // the cache
	if err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	var config struct {
		ONNX string `json:"onnx"`
	}
	if err = json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("%w: %w", errConfig, err)
	}
	if config.ONNX == "" {
		return nil, fmt.Errorf("%w: no onnx file", errConfig)
	}
	cfg.Files = append(cfg.Files, folder+"/tokenizer.json", folder+"/"+config.ONNX)
	if root, err = download(ctx, cfg); err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	return loadFolder(filepath.Join(root, filepath.FromSlash(folder)), threads)
}

//nolint:gochecknoglobals // seams for tests
var (
	// download fetches a release's files, and loadFolder opens a downloaded
	// folder. Variables so tests can see what is asked for.
	download   = hfhub.Download
	loadFolder = LoadModel
)

// BreathsPerReply is the most breaths a reply takes, and false for no limit.
func (m *Model) BreathsPerReply() (int, bool) {
	if m.config.BreathsPerReply == nil {
		return 0, false
	}
	return *m.config.BreathsPerReply, true
}

// BreathKinds is the kinds a breath is drawn as, with their shares, in order;
// empty for a plain breath.
func (m *Model) BreathKinds() []BreathKind { return slices.Clone(m.config.BreathKinds) }

// offset is a token's code-point offsets in the sentence, where it is one of
// the sentence's tokens.
type offset struct {
	start, end int
	ok         bool
}

// heads is the model's outputs for one sentence.
type heads struct {
	delivery, before, after []float32
	// spans and emphasis are each token's three classes.
	spans, emphasis [][]float32
	breath          []float32
}

// Tag labels one sentence of a reply, from the labels the turn's space holds.
// earlier is the reply's earlier sentences and sentence the sentence, each
// without tags.
func (m *Model) Tag(turn TurnContext, earlier []string, sentence string) (SentenceLabels, error) {
	ids, attention, sentenceMask, offsets, err := m.encode(turn, earlier, sentence)
	if err != nil {
		return SentenceLabels{}, err
	}
	shape := []int64{1, int64(len(ids))}
	results, err := m.session.Run([]onnxrt.Tensor{
		onnxrt.Int64(shape, ids), onnxrt.Int64(shape, attention), onnxrt.Int64(shape, sentenceMask),
	})
	if err != nil {
		return SentenceLabels{}, fmt.Errorf("tagger: %w", err)
	}
	h, err := readHeads(results)
	if err != nil {
		return SentenceLabels{}, err
	}
	return m.decode(h, offsets, sentence, turn.Space)
}

// readHeads is the first row of each output: a vector for a sentence's
// heads, three classes a token for its tokens' heads.
func readHeads(results []onnxrt.Tensor) (heads, error) {
	if len(results) != len(outputNames) {
		return heads{}, errOutputs
	}
	vectors := make([][]float32, len(results))
	for _, i := range []int{0, 1, 2, 5} {
		t := results[i]
		if len(t.Shape) != 2 || t.Shape[1] < 0 || int64(len(t.F32)) < t.Shape[1] {
			return heads{}, errOutputs
		}
		vectors[i] = t.F32[:t.Shape[1]]
	}
	tokens := make([][][]float32, len(results))
	for _, i := range []int{3, 4} {
		t := results[i]
		if len(t.Shape) != 3 || t.Shape[1] < 0 || t.Shape[2] <= 0 || int64(len(t.F32)) < t.Shape[1]*t.Shape[2] {
			return heads{}, errOutputs
		}
		width := int(t.Shape[2])
		rows := make([][]float32, t.Shape[1])
		for j := range rows {
			rows[j] = t.F32[j*width : (j+1)*width]
		}
		tokens[i] = rows
	}
	return heads{
		delivery: vectors[0], before: vectors[1], after: vectors[2],
		spans: tokens[3], emphasis: tokens[4], breath: vectors[5],
	}, nil
}

// decode is the labels from one sentence's heads, from those the space holds.
func (m *Model) decode(h heads, offsets []offset, sentence string, space LabelSpace) (SentenceLabels, error) {
	if m.config.Deliveries == nil || m.config.Sounds == nil {
		return SentenceLabels{}, fmt.Errorf("%w: no deliveries or sounds", errConfig)
	}
	shift, err := m.shifts(sentence)
	if err != nil {
		return SentenceLabels{}, err
	}
	labels, err := m.sentenceLabels(h, shift, space)
	if err != nil {
		return SentenceLabels{}, err
	}
	text := []rune(sentence)
	n := len(offsets)
	if space.Deliveries.Has(expressive.SlowDelivery) {
		slow, rangeErr := ranges(argmaxRows(h.spans[:min(n, len(h.spans))]), offsets, text)
		if rangeErr != nil {
			return SentenceLabels{}, rangeErr
		}
		for _, r := range slow {
			labels.Spans = append(labels.Spans, Span{r[0], r[1], expressive.SlowDelivery})
		}
	}
	if space.Emphasis {
		if labels.Emphasis, err = ranges(argmaxRows(h.emphasis[:min(n, len(h.emphasis))]), offsets, text); err != nil {
			return SentenceLabels{}, err
		}
	}
	return labels, nil
}

// shifts is each calibrated sound's shift for the sentence. Calibration
// shifts a sound's logit by the sentence's words: [[most words, shift], ...,
// [null, shift]].
func (m *Model) shifts(sentence string) (map[string]float64, error) {
	words := 0
	for _, token := range shared.Fields(sentence) {
		if slices.ContainsFunc([]rune(token), shared.IsAlnum) {
			words++
		}
	}
	shift := make(map[string]float64, len(m.config.SoundBias))
	for sound, steps := range m.config.SoundBias {
		s, err := shiftFor(steps, words)
		if err != nil {
			return nil, err
		}
		shift[sound] = s
	}
	return shift, nil
}

// sentenceLabels is the sentence's delivery and its sounds.
func (m *Model) sentenceLabels(h heads, shift map[string]float64, space LabelSpace) (SentenceLabels, error) {
	sounds := m.config.Sounds
	soundShift := make([]float64, len(sounds))
	for i, sound := range sounds {
		soundShift[i] = shift[sound]
	}
	// A breath has a head of its own, so the sound heads never choose one.
	var others []string
	for _, sound := range space.Sounds.Labels() {
		if sound != breath {
			others = append(others, sound)
		}
	}
	allowed := func(sound string) bool { return slices.Contains(others, sound) }

	delivery, err := choose(widen(h.delivery), m.config.Deliveries, space.Deliveries.Has)
	if err != nil {
		return SentenceLabels{}, err
	}
	labels := SentenceLabels{Delivery: delivery}
	for _, sound := range []struct {
		logits []float32
		label  *string
	}{{h.before, &labels.SoundBefore}, {h.after, &labels.SoundAfter}} {
		logits, err := shifted(sound.logits, soundShift)
		if err != nil {
			return SentenceLabels{}, err
		}
		if *sound.label, err = choose(logits, sounds, allowed); err != nil {
			return SentenceLabels{}, err
		}
	}
	// A breath goes before the sentence, where its logit, shifted, is above
	// zero and no other sound starts the sentence. The logit is 32-bit, and the
	// shift is added to it in 32 bits.
	if labels.SoundBefore == "" && space.Sounds.Has(breath) {
		if len(h.breath) == 0 {
			return SentenceLabels{}, errOutputs
		}
		if h.breath[0]+float32(shift[breath]) > 0 {
			labels.SoundBefore = breath
		}
	}
	return labels, nil
}

// encode is the model's inputs for one sentence, and each token's offsets in
// the sentence.
func (m *Model) encode(
	turn TurnContext, earlier []string, sentence string,
) (ids, attention, sentenceMask []int64, offsets []offset, err error) {
	first, second, start := encode(turn, earlier, sentence, m.layout)
	enc, err := m.encodePair(first, second)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("tagger: %w", err)
	}
	count := min(len(enc.SequenceIDs), len(enc.SpecialTokensMask), len(enc.Offsets))
	sentenceMask = make([]int64, count)
	offsets = make([]offset, count)
	for i := range count {
		a, b := enc.Offsets[i].Start, enc.Offsets[i].End
		if enc.SequenceIDs[i] == 1 && enc.SpecialTokensMask[i] == 0 && a >= start && b > a {
			sentenceMask[i] = 1
			offsets[i] = offset{a - start, b - start, true}
		}
	}
	return enc.IDs, enc.AttentionMask, sentenceMask, offsets, nil
}

// encodePair tokenizes the text pair within the model's token budget, cutting
// from the left.
func (m *Model) encodePair(first, second string) (tokenizers.Encoding, error) {
	l := m.layout
	if l.CutCallFirst {
		// The tokenizer won't cut a text to nothing, so a call too long to keep
		// any of is left out.
		trunc := &tokenizers.Truncation{
			MaxLength: l.MaxTokens, Strategy: tokenizers.OnlyFirst, Direction: tokenizers.Left,
		}
		for _, call := range []string{first, ""} {
			if enc, err := m.tokenizer.EncodePair(call, second, trunc); err == nil {
				return enc, nil
			}
		}
	}
	return m.tokenizer.EncodePair(first, second, &tokenizers.Truncation{
		MaxLength: l.MaxTokens, Strategy: tokenizers.LongestFirst, Direction: tokenizers.Left,
	})
}

// shiftFor is the shift of the first step that holds a sentence of words.
func shiftFor(steps []step, words int) (float64, error) {
	for _, s := range steps {
		if s.most == nil || float64(words) <= *s.most {
			return s.shift, nil
		}
	}
	return 0, fmt.Errorf("%w: no sound_bias step for %d words", errConfig, words)
}

func widen(logits []float32) []float64 {
	out := make([]float64, len(logits))
	for i, l := range logits {
		out[i] = float64(l)
	}
	return out
}

// shifted is the logits with the shifts added, in 64 bits.
func shifted(logits []float32, shift []float64) ([]float64, error) {
	if len(logits) != len(shift) {
		return nil, errOutputs
	}
	out := widen(logits)
	for i := range out {
		out[i] += shift[i]
	}
	return out, nil
}

// choose is the likeliest class the space allows, or empty for no label.
func choose(logits []float64, classes []string, allowed func(string) bool) (string, error) {
	if len(logits) != len(classes) {
		return "", errOutputs
	}
	masked := make([]float64, len(logits))
	for i, c := range classes {
		masked[i] = math.Inf(-1)
		if c == none || allowed(c) {
			masked[i] = logits[i]
		}
	}
	if best := classes[argmax(masked)]; best != none {
		return best, nil
	}
	return "", nil
}

// argmax is the index of the largest value, the first of several equal ones,
// or of the first NaN, as numpy's argmax.
func argmax[F float32 | float64](values []F) int {
	best := 0
	for i, v := range values {
		if v != v { //nolint:gocritic // NaN
			return i
		}
		if v > values[best] {
			best = i
		}
	}
	return best
}

func argmaxRows(rows [][]float32) []int {
	out := make([]int, len(rows))
	for i, r := range rows {
		out[i] = argmax(r)
	}
	return out
}

// ranges is code-point ranges from begin, inside and outside token classes,
// widened to whole words.
func ranges(classes []int, offsets []offset, sentence []rune) ([][2]int, error) {
	var words [][2]int
	for _, r := range tokenRanges(classes, offsets) {
		w, ok, err := wholeWords(r, sentence)
		if err != nil {
			return nil, err
		}
		if ok {
			words = append(words, w)
		}
	}
	return words, nil
}

// tokenRanges is the ranges the token classes mark, from the first offset of
// each range's first token to the last of its last.
func tokenRanges(classes []int, offsets []offset) [][2]int {
	var found [][2]int
	var current *[2]int
	for i, cls := range classes[:min(len(classes), len(offsets))] {
		o := offsets[i]
		if !o.ok {
			continue
		}
		switch {
		case cls == begin || (cls == inside && current == nil):
			if current != nil {
				found = append(found, *current)
			}
			current = &[2]int{o.start, o.end}
		case cls == inside:
			current[1] = o.end
		case current != nil:
			found = append(found, *current)
			current = nil
		}
	}
	if current != nil {
		found = append(found, *current)
	}
	return found
}

// wholeWords is the range without the spaces at its ends, widened to whole
// words, and false when it is only spaces. A range running past the sentence
// is an error: walking it in from both ends reaches past the sentence's last
// character whatever it holds.
func wholeWords(r [2]int, sentence []rune) ([2]int, bool, error) {
	a, b := r[0], r[1]
	if a < 0 || b > len(sentence) {
		return [2]int{}, false, errOffset
	}
	// A subword token's offsets can include the space before it.
	for a < b && shared.IsSpace(sentence[a]) {
		a++
	}
	for b > a && shared.IsSpace(sentence[b-1]) {
		b--
	}
	if a == b {
		return [2]int{}, false, nil
	}
	for a > 0 && shared.IsAlnum(sentence[a-1]) {
		a--
	}
	for b < len(sentence) && shared.IsAlnum(sentence[b]) {
		b++
	}
	return [2]int{a, b}, true, nil
}
