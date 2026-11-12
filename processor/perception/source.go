package perception

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/internal/pytext"
)

// ErrInvalidConfig reports a reader or a source that cannot be built as given.
var ErrInvalidConfig = errors.New("perception: invalid configuration")

const (
	// DefaultTextTimeout is how long a transcription waits for a text
	// classifier's answers by default, counted from when it arrives.
	DefaultTextTimeout = 500 * time.Millisecond
	// DefaultAudioTimeout is how long a transcription waits for the answers of
	// a classifier that hears by default, counted from when the user stops
	// speaking.
	DefaultAudioTimeout = 2 * time.Second
	// defaultMaxLabels is the most options a choice reports by default.
	defaultMaxLabels = 2
)

// Question is one signal a Reader reports, and the question it asks for it.
type Question struct {
	// Question is what the classifier is asked. A choice is reported as its
	// likeliest options, a yes/no as its probability and a score as its
	// position on the scale.
	Question classifier.Question
	// Description is a short line telling the LLM how to read the signal, such
	// as its range, or empty.
	Description string
	// MaxLabels is the most options a choice reports, likeliest first. Zero is
	// 2; it is at least 1.
	MaxLabels int
}

// maxLabels is the most options the question's choice reports.
func (q Question) maxLabels() int {
	if q.MaxLabels <= 0 {
		return defaultMaxLabels
	}
	return q.MaxLabels
}

// NamedQuestion is a question with the name of the signal it reports.
type NamedQuestion struct {
	// Name is the signal's name, as written in a reading: no whitespace, "=",
	// ",", "[", "]", "(" or ")".
	Name     string
	Question Question
}

// Source is one classifier a Reader asks, and the questions it is asked.
//
// A reader with several sources asks each one about the same turn and writes
// their answers in one reading, so a signal's name belongs to one source. The
// classifier says how its source reads: a text classifier reads each final
// transcription, and one that can hear, as its SupportsAudio says, hears the
// user's audio.
//
// [NewSource] builds a source whose questions are known up front. A classifier
// whose questions are not known until it is set up gets a source of its own
// that embeds a BaseSource and overrides Questions, and Names to name its
// signals before then.
type Source interface {
	// Classifier is what answers the questions about each turn.
	Classifier() classifier.Classifier
	// Questions is the questions asked about each turn, in the order their
	// signals are reported. It fails, with an error wrapping
	// classifier.ErrClassifier, while they are not known yet.
	Questions() ([]NamedQuestion, error)
	// Names is the names of the signals the source reports, known before it is
	// set up.
	Names() []string
	// Timeout is how long a transcription waits for the classifier's answers.
	Timeout() time.Duration
}

// SourceConfig configures a BaseSource.
type SourceConfig struct {
	// Classifier is what answers the questions about each turn. Required.
	Classifier classifier.Classifier `validate:"required"`
	// Questions is the questions, each with the name of the signal it reports.
	Questions []NamedQuestion
	// Timeout is how long a transcription waits for the classifier's answers.
	// Nil is DefaultTextTimeout for a text classifier, counted from when the
	// transcription arrives, and DefaultAudioTimeout for one that hears,
	// counted from when the user stops speaking.
	Timeout *time.Duration
}

// BaseSource is a source whose questions are known when it is built.
//
//	hume, err := perception.NewSource(perception.SourceConfig{
//		Classifier: audioClassifier,
//		Questions: []perception.NamedQuestion{{
//			Name: "hesitation",
//			Question: perception.Question{
//				Question:    classifier.YesNoQuestion{Instructions: "Does the speaker sound unsure?"},
//				Description: "The chance the user sounded unsure, from 0 to 1.",
//			},
//		}},
//	})
type BaseSource struct {
	classifier classifier.Classifier
	questions  []NamedQuestion
	timeout    time.Duration
}

// NewSource builds a source. It fails with ErrInvalidConfig when a name cannot
// be written in a reading.
func NewSource(cfg SourceConfig) (*BaseSource, error) {
	if cfg.Classifier == nil {
		return nil, fmt.Errorf("%w: a source needs a classifier", ErrInvalidConfig)
	}
	for _, q := range cfg.Questions {
		if !validSignalName(q.Name) {
			return nil, fmt.Errorf("%w: %q can't name a signal: it has a space or one of =,[]()", ErrInvalidConfig, q.Name)
		}
		if q.Question.Question == nil {
			return nil, fmt.Errorf("%w: %q has no question", ErrInvalidConfig, q.Name)
		}
		if q.Question.MaxLabels < 0 {
			return nil, fmt.Errorf("%w: %q reports at most %d labels, want at least 1",
				ErrInvalidConfig, q.Name, q.Question.MaxLabels)
		}
	}
	timeout := DefaultTextTimeout
	if cfg.Classifier.SupportsAudio() {
		timeout = DefaultAudioTimeout
	}
	if cfg.Timeout != nil {
		timeout = *cfg.Timeout
	}
	return &BaseSource{
		classifier: cfg.Classifier,
		questions:  append([]NamedQuestion(nil), cfg.Questions...),
		timeout:    timeout,
	}, nil
}

// Classifier is the classifier that answers the questions.
func (s *BaseSource) Classifier() classifier.Classifier { return s.classifier }

// Questions is the questions asked about each turn, by the name of the signal
// each reports.
func (s *BaseSource) Questions() ([]NamedQuestion, error) {
	return append([]NamedQuestion(nil), s.questions...), nil
}

// Names is the names of the signals the source reports.
func (s *BaseSource) Names() []string {
	names := make([]string, 0, len(s.questions))
	for _, q := range s.questions {
		names = append(names, q.Name)
	}
	return names
}

// Timeout is how long a transcription waits for the classifier's answers.
func (s *BaseSource) Timeout() time.Duration { return s.timeout }

// validSignalName reports whether name can be written in a reading: not empty,
// with no whitespace, "=", ",", "[", "]", "(" or ")".
func validSignalName(name string) bool {
	return name != "" && !strings.ContainsFunc(name, func(r rune) bool {
		return pytext.IsSpace(r) || strings.ContainsRune("=,[]()", r)
	})
}

var _ Source = (*BaseSource)(nil)
