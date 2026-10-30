// Package tagger writes expression tags into an LLM's plain replies, one
// sentence at a time, with the tone tagger model (alpha).
//
// The model (https://huggingface.co/pipecat-ai/tone-models) labels each
// sentence of a reply as it streams in: a delivery, slow spans for details,
// stressed words, a sound before or after and, with
// expressive.GuidanceBreathe, a breath before it, drawn as a light breath, a
// deep breath or an exhale. It reads the call so far and how the user sounded,
// and runs on the CPU in about 15 milliseconds a sentence, through the ONNX
// runtime. The labels are expressive.CoreDelivery, expressive.CoreSounds and
// the kinds of breath it draws (BreathKinds), written as tags in the names the
// TTS service performs them by.
//
// Part of expressive conversation (alpha). These types are likely to change
// between releases.
package tagger

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/expressive/formatter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/text"
)

// ErrVocabulary is returned for a vocabulary that names a label the tagger
// does not write, or names one twice.
var ErrVocabulary = errors.New("tagger: vocabulary")

// Config configures a Tagger. The zero value downloads the default release's
// 17m model and tags with one thread.
type Config struct {
	// Name is the processor's name. Empty is "ToneTagger".
	Name string
	// Size is the model's size: Size17M, or Size32M, which agrees with its
	// teacher a little more often and takes nearly three times as long a
	// sentence. Empty is Size17M.
	Size Size `validate:"omitempty,oneof=17m 32m"`
	// Revision is the model's release on Hugging Face, such as "tagger-rc1".
	// Empty is Revision.
	Revision string
	// ModelDir is a folder to load the model from instead of downloading one,
	// laid out as a release's "tagger/17m", such as for a model you trained or
	// keep yourself. Size and Revision are then unused.
	ModelDir string
	// Threads is the CPU threads for one sentence. Zero is 1.
	Threads int `validate:"min=0"`
	// Vocabulary is the names to write first for a delivery, sound or kind of
	// breath (expressive.CoreDelivery, expressive.CoreSounds, BreathKinds),
	// best first, such as {expressive.DeliveryWarm: {"cheerful"}}. A name is
	// written only where the TTS service performs it and the policy offers it;
	// elsewhere the label's universal name stands. A name written for
	// expressive.SoundBreath is written for every breath, rather than its
	// kinds' names. New refuses a label the tagger does not write.
	Vocabulary map[expressive.UniversalName][]string
	// CacheDir is where a downloaded model is kept. Empty is Hugging Face's own
	// cache, ~/.cache/huggingface/hub.
	CacheDir string
}

// model is what the tagger needs of its model. *Model implements it.
type model interface {
	Tag(turn TurnContext, earlier []string, sentence string) (SentenceLabels, error)
	BreathsPerReply() (int, bool)
	BreathKinds() []BreathKind
}

// modelKey is what identifies a loaded model and its settings.
type modelKey struct {
	dir      string
	size     Size
	revision string
	threads  int
}

//nolint:gochecknoglobals // the models are loaded once a process and shared
var (
	models = &shared.Cache[modelKey, model]{}
	// loadDir and downloadRelease open the model. Variables so tests can stand
	// a model in.
	loadDir = func(dir string, threads int) (model, error) {
		return LoadModel(dir, threads)
	}
	downloadRelease = func(
		ctx context.Context, size Size, revision string, threads int, cacheDir string,
	) (model, error) {
		return DownloadModel(ctx, size, revision, threads, cacheDir)
	}
)

// Tagger tags an LLM's plain replies with the tone tagger, a sentence at a
// time.
//
// It goes between the LLM and the TTS service. Each sentence is labeled with
// the call so far, how the user sounded and the pipeline's expression policy,
// and sent on with the tags the TTS service performs. The sentence waits for
// the labeling, about 15 milliseconds.
//
// The model loads while the pipeline is set up, once a process, and every
// tagger shares it. The first load downloads it from Hugging Face, which holds
// the pipeline's start, so download it ahead, such as when building the image.
// At start, the tagger broadcasts a frames.ExpressionMetadataFrame, so the LLM
// is taught no tags and writes plain words, and the context keeps them plain.
// A model that cannot be loaded is reported as an error that leaves the tagger
// unusable: the LLM writes the tags itself and replies pass through as
// written.
//
// The policy's guidance presets steer the tagger, its Delivery and Sounds
// narrow the names it writes, and a breath is written only with
// expressive.GuidanceBreathe. The developer's own lines are for the LLM.
//
// Turn expressive conversation on with
// pipeline.Params.EnableExpressiveConversation. Add a tone reader
// (processor/tone/reader) for the tagger to hear how the user sounded:
//
//	pipeline.New(..., userAggregator, llm, toneTagger, tts, ...)
type Tagger struct {
	*aggregators.LLMText
	cfg        Config
	vocabulary map[string][]string
	model      model
	policy     expressive.ExpressionPolicy
	space      *LabelSpace
	messages   []Message
	turn       *TurnContext
	renderer   *replyRenderer
	sentences  []string
	// labels is the labels of the reply being tagged, which the next reply's
	// model reads; labeled is false before the first reply.
	labels  []SentenceLabels
	labeled bool
	// draw draws each breath's kind, seeded by the call so far, so a call
	// replays alike.
	draw *draw
}

// New builds a Tagger. It fails on a configuration it cannot use, and with
// ErrVocabulary on a vocabulary naming a label the tagger does not write.
func New(cfg Config) (*Tagger, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("tagger: %w", err)
	}
	vocabulary, err := vocabularyByLabel(cfg.Vocabulary)
	if err != nil {
		return nil, err
	}
	if cfg.Name == "" {
		cfg.Name = "ToneTagger"
	}
	if cfg.Size == "" {
		cfg.Size = Size17M
	}
	if cfg.Revision == "" {
		cfg.Revision = Revision
	}
	if cfg.Threads == 0 {
		cfg.Threads = 1
	}
	t := &Tagger{cfg: cfg, vocabulary: vocabulary}
	t.LLMText = aggregators.NewLLMTextFor(cfg.Name, t,
		text.NewSkipTagsAggregator(frames.TextTypeSentence, "", formatter.ExpressionTags()))
	t.AddTextTransformer(tts.TextTransformer{TextType: frames.TextTypeSentence, Transform: t.tag})
	return t, nil
}

// vocabularyByLabel is the vocabulary keyed by the labels' names, refusing a
// label the tagger does not write.
func vocabularyByLabel(vocabulary map[expressive.UniversalName][]string) (map[string][]string, error) {
	var labels []string
	for _, d := range expressive.CoreDelivery() {
		labels = append(labels, string(d))
	}
	for _, s := range slices.Concat(expressive.CoreSounds(), BreathKinds()) {
		labels = append(labels, string(s))
	}
	byLabel := make(map[string][]string, len(vocabulary))
	var unknown []string
	for name, names := range vocabulary {
		if name == nil {
			continue
		}
		label := name.String()
		if !slices.Contains(labels, label) {
			unknown = append(unknown, label)
			continue
		}
		if _, twice := byLabel[label]; twice {
			return nil, fmt.Errorf("%w: names %q twice", ErrVocabulary, label)
		}
		byLabel[label] = slices.Clone(names)
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		return nil, fmt.Errorf("%w: the tagger writes none of %q; it writes %q", ErrVocabulary, unknown, labels)
	}
	return byLabel, nil
}

// Setup sets the processor up with the pipeline's expression policy, and loads
// its model.
func (t *Tagger) Setup(ctx context.Context, s processor.Setup) error {
	if err := t.LLMText.Setup(ctx, s); err != nil {
		return err
	}
	t.policy = expressive.ExpressionPolicy{}
	if s.ExpressionPolicy != nil {
		t.policy = *s.ExpressionPolicy
	}
	if !s.EnableExpressiveConversation {
		slog.WarnContext(ctx, t.Name()+": expressive conversation is off, so replies pass through untagged; "+
			"turn it on with pipeline.Params.EnableExpressiveConversation")
		return nil
	}
	key := modelKey{dir: t.cfg.ModelDir, threads: t.cfg.Threads}
	if key.dir == "" {
		key.size, key.revision = t.cfg.Size, t.cfg.Revision
	}
	m, err := models.Load(key, func() (model, error) {
		if t.cfg.ModelDir != "" {
			return loadDir(t.cfg.ModelDir, t.cfg.Threads)
		}
		return downloadRelease(ctx, t.cfg.Size, t.cfg.Revision, t.cfg.Threads, t.cfg.CacheDir)
	})
	if err != nil {
		t.PushError(ctx, fmt.Sprintf("%s: couldn't load its model: %v", t.Name(), err), err, false,
			processor.ForceTreatAsPermanent())
		return nil
	}
	t.model = m
	return nil
}

// ProcessFrame tags the LLM's sentences, and keeps the call and the TTS's names
// to tag them by.
func (t *Tagger) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if t.model == nil {
		// Replies pass through as the LLM wrote them.
		return t.LLMText.ProcessFrame(ctx, f, dir)
	}

	switch fr := f.(type) {
	case *frames.TTSMetadataFrame:
		t.onTTSMetadata(ctx, fr)
	case *frames.LLMContextUserTurnFrame:
		t.onUserTurn(fr)
	case *rtvi.SendTextFrame:
		// Text the user typed is appended to the context with no user turn
		// frame, and with no reading.
		t.messages = append(t.messages, Message{Role: roleUser, Text: fr.Text})
	case *frames.LLMContextAssistantTurnFrame:
		t.messages = append(t.messages, Message{Role: "assistant", Text: fr.Text})
	case *frames.UserTurnResponseCancelFrame:
		// The turn an early reply was to answer is not over yet.
		t.messages = slices.DeleteFunc(t.messages, func(m Message) bool { return m.Speculation })
	case *frames.InterruptionFrame:
		t.turn = nil
	}

	if err := t.LLMText.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch f.(type) {
	case *frames.StartFrame:
		return t.Broadcast(ctx, func() frames.Frame { return frames.NewExpressionMetadataFrame(t.Name()) })
	case *frames.LLMFullResponseStartFrame:
		t.startReply()
	}
	return nil
}

func (t *Tagger) onTTSMetadata(ctx context.Context, f *frames.TTSMetadataFrame) {
	if f.Expression == nil {
		slog.WarnContext(ctx, fmt.Sprintf("%s: %s performs no expression, so replies pass through untagged",
			t.Name(), f.ServiceName))
		t.space = nil
		return
	}
	space := ResolveLabelSpace(f.Expression, t.policy, f.ServiceName, t.vocabulary)
	t.space = &space
}

func (t *Tagger) onUserTurn(f *frames.LLMContextUserTurnFrame) {
	m := Message{Role: roleUser, Text: f.Text, Heard: fold(f.Perception), Speculation: f.Speculation}
	// A turn replaces the speculation before it, where it stands: the early
	// reply that answered it may already follow it.
	for i := len(t.messages) - 1; i >= 0; i-- {
		if t.messages[i].Speculation {
			t.messages[i] = m
			return
		}
	}
	t.messages = append(t.messages, m)
}

func (t *Tagger) startReply() {
	// Text left from a reply cut short is not part of this one.
	t.Reset()
	t.sentences = nil
	if t.space == nil {
		t.turn = nil
		return
	}
	var previous []SentenceLabels
	if t.labeled {
		previous = t.labels
	}
	t.turn = &TurnContext{Messages: slices.Clone(t.messages), Space: *t.space, Policy: t.policy, Previous: previous}
	t.renderer = newReplyRenderer(*t.space)
	t.labels, t.labeled = []SentenceLabels{}, true
	texts := make([]string, len(t.turn.Messages))
	for i, m := range t.turn.Messages {
		texts[i] = m.Text
	}
	t.draw = newDraw(strings.Join(texts, "\n"))
}

// tag tags one sentence of the reply; text that is not spoken stays as
// written.
func (t *Tagger) tag(ctx context.Context, sentence string, _ frames.TextType) (string, error) {
	if t.turn == nil || t.renderer == nil || t.model == nil || !t.labeled || t.SkippingTTS() {
		return sentence, nil
	}
	// The labels are offsets into the sentence as the model reads it.
	plain := strings.Join(shared.Fields(sentence), " ")
	earlier := slices.Clone(t.sentences)
	t.sentences = append(t.sentences, plain)
	started := time.Now()
	labels, tagged, err := t.label(earlier, plain)
	if err != nil {
		// A transform that fails drops the sentence, so it is spoken untagged.
		t.PushError(ctx, fmt.Sprintf("%s: couldn't tag a sentence, spoken untagged: %v", t.Name(), err), err, false)
		return plain, nil
	}
	slog.DebugContext(ctx, fmt.Sprintf("%s: tagged in %.1f ms: %s",
		t.Name(), float64(time.Since(started).Microseconds())/1000, tagged))
	// The next reply's model reads a breath as a breath, whatever kind it was
	// drawn as.
	t.labels = append(t.labels, labels)
	return tagged, nil
}

// label labels the sentence and writes it with its tags.
func (t *Tagger) label(earlier []string, plain string) (SentenceLabels, string, error) {
	labels, err := t.model.Tag(*t.turn, earlier, plain)
	if err != nil {
		return SentenceLabels{}, "", err
	}
	labels = t.cappedBreaths(labels)
	drawn, err := t.drawn(labels)
	if err != nil {
		return SentenceLabels{}, "", err
	}
	return labels, t.renderer.render(plain, drawn), nil
}

// drawn is the labels with a breath drawn as a kind the service performs, at
// the model's shares.
//
// A breath stays plain when the model or the service has no kinds, or when it
// is written as a name the developer gave for every breath.
func (t *Tagger) drawn(labels SentenceLabels) (SentenceLabels, error) {
	if labels.SoundBefore != breath || t.model == nil || t.turn == nil {
		return labels, nil
	}
	if name, ok := t.turn.Space.Sounds.Name(breath); ok &&
		slices.Contains(t.vocabulary[breath], name) {
		return labels, nil
	}
	var kinds []string
	var shares []float64
	for _, k := range t.model.BreathKinds() {
		if t.turn.Space.Breaths.Has(k.Kind) {
			kinds = append(kinds, k.Kind)
			shares = append(shares, k.Share)
		}
	}
	if len(kinds) == 0 {
		return labels, nil
	}
	kind, err := t.draw.choice(kinds, shares)
	if err != nil {
		return SentenceLabels{}, err
	}
	labels.SoundBefore = kind
	return labels, nil
}

// cappedBreaths is the labels without a breath past the model's limit for a
// reply. The model breathes only before a sentence.
func (t *Tagger) cappedBreaths(labels SentenceLabels) SentenceLabels {
	limit, ok := t.model.BreathsPerReply()
	if !ok || labels.SoundBefore != breath {
		return labels
	}
	taken := 0
	for _, s := range t.labels {
		if s.SoundBefore == breath {
			taken++
		}
	}
	if taken >= limit {
		labels.SoundBefore = ""
	}
	return labels
}
