// Package reader reads how each user turn was meant, from its words, with the
// tone reader model (alpha).
//
// The model (https://huggingface.co/pipecat-ai/tone-models) reads one turn of
// text, with the turn before it, and returns the speaker's emotion, whether the
// turn is sarcastic, the move it makes and whether it is only a listening
// sound, each with a calibrated confidence. It runs on the CPU in a few
// milliseconds, through the ONNX runtime. The classifier/tone package answers
// with it; this one reads each turn with that classifier.
//
// Part of expressive conversation (alpha). These types are likely to change
// between releases.
package reader

import (
	"fmt"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/tone"
	"github.com/gojargo/jargo/processor/perception"
)

const (
	emotionDescription = "The feeling the words show. Read from the words alone, so neutral means they show no " +
		"strong feeling, not that the user feels none: go by what they say."
	sarcasmDescription = "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: " +
		"a sarcastic turn can read happy."
)

// toneClassifier is what a ToneSource asks: a tone classifier, whose questions
// come from its model. *tone.Classifier implements it.
type toneClassifier interface {
	classifier.Classifier
	Questions() ([]classifier.Named[classifier.Question], error)
}

// newTone builds a source's tone classifier. A variable so tests can stand one
// in.
//
//nolint:gochecknoglobals // a seam for tests
var newTone = func(cfg tone.Config) (toneClassifier, error) { return tone.New(cfg) }

// SourceConfig configures a ToneSource. The zero value downloads the default
// release at FP16, reads with four threads and waits 0.5 s for a reading.
type SourceConfig struct {
	// Revision is the model's release on Hugging Face, such as "reader-rc2".
	// Empty is tone.Revision.
	Revision string
	// ModelDir is a folder to load the model from instead of downloading one,
	// laid out as a release's "reader" folder. Revision is then unused.
	ModelDir string
	// Precision is the weights to load. Empty is tone.FP16.
	Precision tone.Precision
	// Threads is the CPU threads for one read. Zero is 4.
	Threads int
	// CacheDir is where a downloaded model is kept. Empty is Hugging Face's own
	// cache, ~/.cache/huggingface/hub.
	CacheDir string
	// Timeout is how long a transcription waits for its reading. Nil is
	// perception.DefaultTextTimeout, 0.5 s.
	Timeout *time.Duration
}

// ToneSource is the tone classifier as one of a perception.Reader's sources.
//
// It reads each final transcription, with the bot's reply before it, and
// reports the turn's "emotion", its two likeliest labels with their
// confidences, and its "sarcasm", the chance it is sarcastic. Give it to a
// reader beside the sources of other classifiers, such as one that hears the
// user's audio. Another source's signal needs a name other than these two.
//
//	toneSource, err := reader.NewSource(reader.SourceConfig{})
//	r, err := perception.New(perception.Config{Sources: []perception.Source{toneSource, hearing}})
type ToneSource struct {
	*perception.BaseSource
	tone toneClassifier
}

// NewSource builds a ToneSource. It fails on a configuration it cannot use.
func NewSource(cfg SourceConfig) (*ToneSource, error) {
	c, err := newTone(tone.Config{
		Revision: cfg.Revision, ModelDir: cfg.ModelDir, Precision: cfg.Precision,
		Threads: cfg.Threads, CacheDir: cfg.CacheDir,
	})
	if err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	// The tone classifier's questions come from its model once it is loaded:
	// see Questions.
	base, err := perception.NewSource(perception.SourceConfig{Classifier: c, Timeout: cfg.Timeout})
	if err != nil {
		return nil, fmt.Errorf("reader: %w", err)
	}
	return &ToneSource{BaseSource: base, tone: c}, nil
}

// Questions is the tone classifier's emotion and sarcasm questions, with the
// loaded model's options. It fails, with an error wrapping
// classifier.ErrClassifier, before the model is loaded.
func (s *ToneSource) Questions() ([]perception.NamedQuestion, error) {
	own, err := s.tone.Questions()
	if err != nil {
		return nil, err
	}
	asked := map[string]classifier.Question{}
	for _, q := range own {
		asked[q.Name] = q.Question
	}
	return []perception.NamedQuestion{
		{Name: tone.Emotion, Question: perception.Question{Question: asked[tone.Emotion], Description: emotionDescription}},
		{Name: tone.Sarcasm, Question: perception.Question{Question: asked[tone.Sarcasm], Description: sarcasmDescription}},
	}, nil
}

// Names is the tone classifier's signals, "emotion" and "sarcasm", named before
// the model loads.
func (s *ToneSource) Names() []string { return []string{tone.Emotion, tone.Sarcasm} }

// Config configures a tone reader. The zero value downloads the default release
// at FP16, reads with four threads and waits 0.5 s for a reading.
type Config struct {
	// Name is the processor's name. Empty is "ToneReader".
	Name string
	// Revision is the model's release on Hugging Face, such as "reader-rc2".
	// Empty is tone.Revision.
	Revision string
	// ModelDir is a folder to load the model from instead of downloading one,
	// laid out as a release's "reader" folder. Revision is then unused.
	ModelDir string
	// Precision is the weights to load. Empty is tone.FP16.
	Precision tone.Precision
	// Threads is the CPU threads for one read. Zero is 4.
	Threads int
	// CacheDir is where a downloaded model is kept. Empty is Hugging Face's own
	// cache, ~/.cache/huggingface/hub.
	CacheDir string
	// Timeout is how long a transcription waits for its reading. Nil is
	// perception.DefaultTextTimeout, 0.5 s.
	Timeout *time.Duration
	// ReplacePerception makes the reading replace the one a transcription
	// carries, such as the STT service's, instead of being added to it.
	ReplacePerception bool
}

// Reader reads how each user turn was meant: a perception.Reader whose one
// source is a ToneSource.
//
// It goes between the STT service and the user aggregator. Each final
// transcription is read, with the bot's reply before it, and the reading is
// added to its Perception: the turn's "emotion", its two likeliest labels with
// their confidences, and its "sarcasm", the chance it is sarcastic. The
// transcription waits for the read, a few milliseconds, up to Config.Timeout.
//
// The model loads while the pipeline is set up, once a process, and every
// reader shares it. The first load downloads it from Hugging Face, which holds
// the pipeline's start, so download it ahead, such as when building the image.
// A model that cannot be loaded is reported as an error that leaves the reader
// unusable, and transcriptions pass through unread.
//
// Turn expressive conversation on with
// pipeline.Params.EnableExpressiveConversation for the LLM to read the turns:
//
//	pipeline.New(transport.Input(), stt, toneReader, userAggregator, llm, ...)
type Reader = perception.Reader

// New builds a tone reader. It fails on a configuration it cannot use.
func New(cfg Config) (*Reader, error) {
	source, err := NewSource(SourceConfig{
		Revision: cfg.Revision, ModelDir: cfg.ModelDir, Precision: cfg.Precision,
		Threads: cfg.Threads, CacheDir: cfg.CacheDir, Timeout: cfg.Timeout,
	})
	if err != nil {
		return nil, err
	}
	if cfg.Name == "" {
		cfg.Name = "ToneReader"
	}
	return perception.New(perception.Config{
		Name: cfg.Name, Sources: []perception.Source{source}, ReplacePerception: cfg.ReplacePerception,
	})
}

var _ perception.Source = (*ToneSource)(nil)
