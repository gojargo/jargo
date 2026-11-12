// Package tone is a classifier backed by the tone reader model, a local model of
// how one turn of text was meant (alpha).
//
// The model (https://huggingface.co/pipecat-ai/tone-models) reads one turn of
// text, with the turn before it, and returns the speaker's emotion, whether the
// turn is sarcastic, the move it makes and whether it is only a listening
// sound, each with a calibrated confidence. It runs on the CPU in a few
// milliseconds, through the ONNX runtime.
//
// These types are likely to change between releases.
package tone

import (
	"context"
	"fmt"
	"reflect"
	"sync"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/modelcache"
	"github.com/gojargo/jargo/internal/validate"
)

// The names of the questions the model answers.
const (
	Emotion     = "emotion"
	Move        = "move"
	Sarcasm     = "sarcasm"
	Backchannel = "backchannel"
)

// Config configures a Classifier. The zero value downloads the default release
// at FP16 and reads with four threads.
type Config struct {
	// Name names the classifier in its metrics; empty uses
	// "ToneClassifier#<n>".
	Name string
	// Revision is the model's release on Hugging Face, such as "reader-rc2".
	// Empty is Revision.
	Revision string
	// ModelDir is a folder to load the model from instead of downloading one,
	// laid out as a release's "reader" folder. Revision is then unused.
	ModelDir string
	// Precision is the weights to load. Empty is FP16.
	Precision Precision `validate:"omitempty,oneof=fp16 fp32"`
	// Threads is the CPU threads for one read. Zero is 4.
	Threads int `validate:"min=0"`
	// CacheDir is where a downloaded model is kept. Empty is Hugging Face's own
	// cache, ~/.cache/huggingface/hub.
	CacheDir string
}

// model is what the classifier needs of its model. *Model implements it.
type model interface {
	Emotions() []string
	Moves() []string
	Tag(text, previous string) (Tags, error)
}

// modelKey is what identifies a loaded model and its settings.
type modelKey struct {
	dir, revision string
	precision     Precision
	threads       int
}

//nolint:gochecknoglobals // the models are loaded once a process and shared
var (
	models = &modelcache.Cache[modelKey, model]{}
	// loadDir and downloadRelease open the model. Variables so tests can stand
	// a model in.
	loadDir = func(dir string, precision Precision, threads int) (model, error) {
		return LoadModel(dir, precision, threads)
	}
	downloadRelease = func(
		ctx context.Context, revision string, precision Precision, threads int, cacheDir string,
	) (model, error) {
		return DownloadModel(ctx, revision, precision, threads, cacheDir)
	}
)

// Classifier answers the tone reader model's own questions about one turn of
// text.
//
// It answers only the questions built into the model, the ones Questions
// returns: the turn's "emotion" and "move", whether it is sarcastic
// ("sarcasm"), and whether it is only a listening sound such as "mhm"
// ("backchannel"). Any other question fails with an error wrapping
// classifier.ErrClassifier. The questions take their options from the model,
// which Setup loads.
//
// The state is the turn's text, or a map with "user" and optionally
// "assistant" to read it with the bot's words before it.
//
// The model loads in Setup, once a process, and every classifier shares it. The
// first load downloads it from Hugging Face.
//
//	c, err := tone.New(tone.Config{})
//	if err := c.Setup(ctx); err != nil { ... }
//	questions, _ := c.Questions()
//	results, err := c.Ask(ctx, map[string]any{
//		"assistant": "Your table is ready in forty minutes.", "user": "Oh great.",
//	}, questions)
//	sarcastic := results[tone.Sarcasm].(classifier.YesNoResult).Probability
type Classifier struct {
	*classifier.Base
	cfg Config

	mu        sync.RWMutex
	model     model
	questions []classifier.Named[classifier.Question]
}

// New builds a tone classifier. It fails on a configuration it cannot use.
func New(cfg Config) (*Classifier, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("tone: %w", err)
	}
	if cfg.Revision == "" {
		cfg.Revision = Revision
	}
	if cfg.Precision == "" {
		cfg.Precision = FP16
	}
	if cfg.Threads == 0 {
		cfg.Threads = 4
	}
	c := &Classifier{cfg: cfg}
	c.Base = classifier.NewBase("ToneClassifier", cfg.Name, c)
	return c, nil
}

// Model is the model's folder, or its release on Hugging Face.
func (c *Classifier) Model() string {
	if c.cfg.ModelDir != "" {
		return c.cfg.ModelDir
	}
	return Repo + "@" + c.cfg.Revision
}

// Questions is the questions the model answers, by name. It fails, with an
// error wrapping classifier.ErrClassifier, before the model is loaded.
func (c *Classifier) Questions() ([]classifier.Named[classifier.Question], error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.model == nil {
		return nil, fmt.Errorf("%w: %s: the questions come from the model, loaded in Setup",
			classifier.ErrClassifier, c.Name())
	}
	return append([]classifier.Named[classifier.Question](nil), c.questions...), nil
}

// Setup loads the model, or shares the one already loaded. It fails, with an
// error wrapping classifier.ErrClassifier, when the model cannot be loaded.
func (c *Classifier) Setup(ctx context.Context) error {
	if err := c.Base.Setup(ctx); err != nil {
		return err
	}
	key := modelKey{dir: c.cfg.ModelDir, precision: c.cfg.Precision, threads: c.cfg.Threads}
	if key.dir == "" {
		key.revision = c.cfg.Revision
	}
	m, err := models.Load(key, func() (model, error) {
		if c.cfg.ModelDir != "" {
			return loadDir(c.cfg.ModelDir, c.cfg.Precision, c.cfg.Threads)
		}
		return downloadRelease(ctx, c.cfg.Revision, c.cfg.Precision, c.cfg.Threads, c.cfg.CacheDir)
	})
	if err != nil {
		return fmt.Errorf("%w: couldn't load the tone model: %w", classifier.ErrClassifier, err)
	}
	c.mu.Lock()
	c.model, c.questions = m, modelQuestions(m)
	c.mu.Unlock()
	return nil
}

// AskQuestions implements classifier.Asker, answering the questions with one
// read of the turn. The call reports no token usage.
func (c *Classifier) AskQuestions(
	_ context.Context, state any, questions []classifier.Named[classifier.Question], _ []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	c.mu.RLock()
	m, own := c.model, c.questions
	c.mu.RUnlock()
	if m == nil {
		return nil, nil, fmt.Errorf("%w: %s: the model isn't loaded, see Setup", classifier.ErrClassifier, c.Name())
	}
	heads := make(map[string]string, len(questions))
	for _, q := range questions {
		head, ok := headOf(own, q.Question)
		if !ok {
			return nil, nil, fmt.Errorf("%w: %s answers only its own questions, from Questions",
				classifier.ErrClassifier, c.Name())
		}
		heads[q.Name] = head
	}
	text, previous, err := c.turn(state)
	if err != nil {
		return nil, nil, err
	}
	tags, err := m.Tag(text, previous)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: couldn't read the turn: %w", classifier.ErrClassifier, err)
	}
	results := make(map[string]classifier.Result, len(questions))
	for name, head := range heads {
		results[name] = resultOf(head, tags)
	}
	return results, nil, nil
}

// modelQuestions is the questions the model answers, with its own options.
func modelQuestions(m model) []classifier.Named[classifier.Question] {
	options := func(names []string) []classifier.Option {
		out := make([]classifier.Option, 0, len(names))
		for _, n := range names {
			out = append(out, classifier.Option{Name: n})
		}
		return out
	}
	return []classifier.Named[classifier.Question]{
		{Name: Emotion, Question: classifier.ChoiceQuestion{
			Instructions: "Which feeling the turn's words show.",
			Options:      options(m.Emotions()),
		}},
		{Name: Move, Question: classifier.ChoiceQuestion{
			Instructions: "What the turn does in the conversation.",
			Options:      options(m.Moves()),
		}},
		{Name: Sarcasm, Question: classifier.YesNoQuestion{Instructions: "Is the turn sarcastic?"}},
		{Name: Backchannel, Question: classifier.YesNoQuestion{
			Instructions: `Is the turn only a listening sound, such as "mhm", that acknowledges ` +
				"the speaker without taking the turn?",
		}},
	}
}

// headOf is which of the model's questions this is, by name, and false if none.
func headOf(own []classifier.Named[classifier.Question], q classifier.Question) (string, bool) {
	for _, o := range own {
		if reflect.DeepEqual(o.Question, q) {
			return o.Name, true
		}
	}
	return "", false
}

// turn is the turn's text and the bot's words before it.
func (c *Classifier) turn(state any) (text, previous string, err error) {
	switch s := state.(type) {
	case string:
		return s, "", nil
	case map[string]string:
		if user, ok := s["user"]; ok {
			return user, s["assistant"], nil
		}
	case map[string]any:
		user, ok := s["user"].(string)
		if !ok {
			break
		}
		switch assistant := s["assistant"].(type) {
		case nil:
			return user, "", nil
		case string:
			return user, assistant, nil
		}
	}
	return "", "", fmt.Errorf("%w: %s reads a turn: text, or a map with 'user' and optionally 'assistant'",
		classifier.ErrClassifier, c.Name())
}

// resultOf is the answer to one of the model's questions, from its tags.
func resultOf(head string, tags Tags) classifier.Result {
	choice := func(label string, confidence float64, probabilities []Probability) classifier.ChoiceResult {
		r := classifier.ChoiceResult{
			Choice: label, Confidence: confidence, Probabilities: make(map[string]float64, len(probabilities)),
		}
		for _, p := range probabilities {
			r.Probabilities[p.Label] = p.Probability
		}
		return r
	}
	switch head {
	case Emotion:
		return choice(tags.Emotion.Label, tags.Emotion.Confidence, tags.Emotion.Probabilities)
	case Move:
		return choice(tags.Move.Label, tags.Move.Confidence, tags.Move.Probabilities)
	case Sarcasm:
		return classifier.YesNoResult{Probability: tags.Sarcasm.Probability}
	default:
		return classifier.YesNoResult{Probability: tags.Backchannel.Probability}
	}
}

var _ classifier.Classifier = (*Classifier)(nil)
