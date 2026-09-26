// Package classifier answers typed questions about some state.
//
// A classifier is a plain object. Whoever needs answers creates one, keeps it,
// and calls it. It does not sit in a pipeline and no frames flow into it. A
// question is a [YesNoQuestion], a [ChoiceQuestion] or a [ScoreQuestion].
// Questions are asked by name, several about one state at once: Ask takes any
// mix of kinds, and YesNo, Choice and Score take questions of one kind and
// return typed results.
//
// The state a question is about is plain text, or structured data such as a
// transcript with speaker labels or a trimmed screen snapshot: a string, or a
// value that encodes to a JSON object or array. The instructions, the meanings
// and the options of a question take the same.
//
// The implementations live in the subpackages: classifier/llm answers through
// any LLM service that runs a one-shot inference, and classifier/jev through
// Jev, TypeSafe's hosted classification model.
package classifier

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/events"
)

// EventMetrics is raised after every call with its metrics, a
// []frames.MetricsData: the time the call took and, when the classifier knows
// it, the tokens it used. A classifier cannot push frames, so the owner is the
// one to put them in a MetricsFrame.
const EventMetrics = "on_metrics"

// ErrClassifier is wrapped by every error a classifier returns because it could
// not answer a question. Test for it with errors.Is.
var ErrClassifier = errors.New("classifier")

// ErrInvalidQuestion reports a question that cannot be asked as it stands.
var ErrInvalidQuestion = errors.New("classifier: invalid question")

// ErrNoSuchLevel reports a level a scale does not have.
var ErrNoSuchLevel = errors.New("classifier: not a level of this scale")

// Question is one of YesNoQuestion, ChoiceQuestion or ScoreQuestion.
type Question interface {
	isQuestion()
}

// YesNoQuestion asks whether the state meets a condition.
type YesNoQuestion struct {
	// Instructions is what is being checked for, as a yes or no question. Text,
	// or structured data holding the question in one field and what it refers
	// to in others.
	Instructions any
	// Yes is what counts as a yes, when the question alone leaves it open; nil
	// when it does not.
	Yes any
	// No is what counts as a no; nil when the question alone says.
	No any
}

// ChoiceQuestion asks which of several options fits the state.
type ChoiceQuestion struct {
	// Instructions is what is being decided.
	Instructions any
	// Options are the options to choose from, in order.
	Options []Option
}

// Option is one option of a ChoiceQuestion.
type Option struct {
	// Name is the option, as the answer names it.
	Name string
	// Description is when the option applies, or nil when its name says enough.
	Description any
}

// ScoreQuestion asks where the state falls on an ordered scale.
type ScoreQuestion struct {
	// Instructions is what is being rated.
	Instructions any
	// Levels are the levels of the scale in order, lowest first, each described
	// in a few words or as structured data. At least two.
	Levels []any
}

func (YesNoQuestion) isQuestion()  {}
func (ChoiceQuestion) isQuestion() {}
func (ScoreQuestion) isQuestion()  {}

// Has reports whether the question offers an option named name.
func (q ChoiceQuestion) Has(name string) bool {
	for _, o := range q.Options {
		if o.Name == name {
			return true
		}
	}
	return false
}

// OptionNames is the names of the options, in order.
func (q ChoiceQuestion) OptionNames() []string {
	names := make([]string, 0, len(q.Options))
	for _, o := range q.Options {
		names = append(names, o.Name)
	}
	return names
}

// Named is a question with the name its answer is returned under.
type Named[Q Question] struct {
	Name     string
	Question Q
}

// Result is one of YesNoResult, ChoiceResult or ScoreResult.
type Result interface {
	isResult()
}

// YesNoResult is the answer to a YesNoQuestion.
type YesNoResult struct {
	// Probability is how likely the answer is yes, from 0 to 1.
	Probability float64
}

// IsYes reports whether yes is the likelier answer. A caller that needs more
// certainty than that compares Probability with a threshold of its own.
func (r YesNoResult) IsYes() bool { return r.Probability >= 0.5 }

// ChoiceResult is the answer to a ChoiceQuestion.
type ChoiceResult struct {
	// Choice is the option that fits best.
	Choice string
	// Probabilities is how likely each option is, keyed by option.
	Probabilities map[string]float64
	// Confidence is how sure the classifier is of Choice, from 0 to 1.
	Confidence float64
}

// ScoreLevel is one level of a ScoreQuestion's scale and how likely it is.
type ScoreLevel struct {
	// Level is the level as the question gave it.
	Level any
	// Probability is how likely the state is at this level, from 0 to 1.
	Probability float64
}

// ScoreResult is the answer to a ScoreQuestion.
type ScoreResult struct {
	// Score is where the state falls on the scale, as a position from 0 (the
	// first level) to one less than the number of levels. It is the
	// probability-weighted position, so it may fall between two levels.
	Score float64
	// Levels is how likely each level is, in the question's order.
	Levels []ScoreLevel
	// Confidence is how sure the classifier is of Score, from 0 to 1.
	Confidence float64
}

// Probability is how likely one level is, given as the question gave it. It
// reports ErrNoSuchLevel for a level the scale does not have.
func (r ScoreResult) Probability(level any) (float64, error) {
	for _, l := range r.Levels {
		if reflect.DeepEqual(l.Level, level) {
			return l.Probability, nil
		}
	}
	return 0, fmt.Errorf("%w: %v", ErrNoSuchLevel, level)
}

func (YesNoResult) isResult()  {}
func (ChoiceResult) isResult() {}
func (ScoreResult) isResult()  {}

// Asker is what a classifier implements: answering the questions about a state
// in one call, and saying what tokens the call used when that is known.
type Asker interface {
	// AskQuestions answers the questions, one result per name, each of the type
	// its question calls for. usage is nil when the call's tokens are unknown.
	// It answers or fails within a bound of its own, so a caller waiting on it
	// is never left hanging, and its errors wrap ErrClassifier.
	AskQuestions(
		ctx context.Context, state any, questions []Named[Question],
	) (results map[string]Result, usage *frames.LLMTokenUsage, err error)
	// Model is the model that answers, named in the metrics, or "".
	Model() string
}

// Classifier answers typed questions about a state. Base implements everything
// but Model, which the classifier embedding it provides.
type Classifier interface {
	// Name is the classifier's name, which its metrics are reported under.
	Name() string
	// Model is the model that answers, named in the metrics, or "".
	Model() string
	// Events is the classifier's event registry, raising EventMetrics.
	Events() *events.Registry
	// Setup prepares the classifier ahead of the first question.
	Setup(ctx context.Context) error
	// Cleanup releases what the classifier holds.
	Cleanup(ctx context.Context) error
	// Ask answers several questions of any kinds about one state.
	Ask(ctx context.Context, state any, questions []Named[Question]) (map[string]Result, error)
	// YesNo asks whether the state meets each condition.
	YesNo(ctx context.Context, state any, questions []Named[YesNoQuestion]) (map[string]YesNoResult, error)
	// Choice asks which option fits the state, for each question.
	Choice(ctx context.Context, state any, questions []Named[ChoiceQuestion]) (map[string]ChoiceResult, error)
	// Score asks where the state falls on each scale.
	Score(ctx context.Context, state any, questions []Named[ScoreQuestion]) (map[string]ScoreResult, error)
}

// nameCounter numbers classifiers that were not given a name.
//
//nolint:gochecknoglobals // process-wide instance counter
var nameCounter atomic.Uint64

// Base holds what every classifier shares: its name, its events, and the typed
// ways of asking, built on the Asker it is given. A classifier embeds it.
//
// An owner calls Setup before the first question and Cleanup when it is done.
// A classifier raises EventMetrics after every call; its owner puts the data in
// a MetricsFrame:
//
//	events.On(c.Events(), classifier.EventMetrics, func(ctx context.Context, data []frames.MetricsData) {
//		_ = p.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream)
//	})
type Base struct {
	registry events.Registry
	name     string
	asker    Asker
}

// NewBase builds the base of a classifier of the named type, answering through
// asker. name names this instance; empty uses "<typeName>#<n>".
func NewBase(typeName, name string, asker Asker) *Base {
	if name == "" {
		name = fmt.Sprintf("%s#%d", typeName, nameCounter.Add(1))
	}
	b := &Base{name: name, asker: asker}
	b.registry.Register(EventMetrics, false)
	return b
}

// Name is the classifier's name, which its metrics are reported under.
func (b *Base) Name() string { return b.name }

// String returns the classifier's name.
func (b *Base) String() string { return b.name }

// Events is the classifier's event registry, raising EventMetrics.
func (b *Base) Events() *events.Registry { return &b.registry }

// Setup prepares the classifier ahead of the first question. The base has
// nothing to prepare.
func (b *Base) Setup(context.Context) error { return nil }

// Cleanup waits for the event handlers still running.
func (b *Base) Cleanup(ctx context.Context) error {
	b.registry.Cleanup(ctx)
	return nil
}

// Ask answers several questions about one state, one result per name, each of
// the type its question calls for. It fails with an error wrapping
// ErrClassifier when the answers could not be produced, or not in time.
func (b *Base) Ask(ctx context.Context, state any, questions []Named[Question]) (map[string]Result, error) {
	for _, q := range questions {
		if s, ok := q.Question.(ScoreQuestion); ok && len(s.Levels) < 2 {
			return nil, fmt.Errorf("%w: %q has %d levels, want at least two",
				ErrInvalidQuestion, q.Name, len(s.Levels))
		}
	}
	started := time.Now()
	results, usage, err := b.asker.AskQuestions(ctx, state, questions)
	if err != nil {
		return nil, err
	}
	b.registry.Call(ctx, EventMetrics, b.asker, b.metrics(time.Since(started), usage))
	return results, nil
}

// YesNo asks whether the state meets each condition, and returns how likely each
// answer is yes, by the same names.
func (b *Base) YesNo(
	ctx context.Context, state any, questions []Named[YesNoQuestion],
) (map[string]YesNoResult, error) {
	results, err := b.Ask(ctx, state, widen(questions))
	if err != nil {
		return nil, err
	}
	return typed[YesNoResult](results)
}

// Choice asks which option fits the state, for each question, and returns the
// option that fits and how likely each one is, by the same names.
func (b *Base) Choice(
	ctx context.Context, state any, questions []Named[ChoiceQuestion],
) (map[string]ChoiceResult, error) {
	results, err := b.Ask(ctx, state, widen(questions))
	if err != nil {
		return nil, err
	}
	return typed[ChoiceResult](results)
}

// Score asks where the state falls on each scale, and returns the position on
// each scale and how likely each level is, by the same names.
func (b *Base) Score(
	ctx context.Context, state any, questions []Named[ScoreQuestion],
) (map[string]ScoreResult, error) {
	results, err := b.Ask(ctx, state, widen(questions))
	if err != nil {
		return nil, err
	}
	return typed[ScoreResult](results)
}

// metrics is what a call reports: the time it took and, when known, the tokens
// it used.
func (b *Base) metrics(took time.Duration, usage *frames.LLMTokenUsage) []frames.MetricsData {
	base := frames.BaseMetricsData{Processor: b.name, Model: b.asker.Model()}
	data := []frames.MetricsData{frames.ProcessingMetricsData{BaseMetricsData: base, Value: took}}
	if usage != nil {
		data = append(data, frames.LLMUsageMetricsData{BaseMetricsData: base, Value: *usage})
	}
	return data
}

// widen turns questions of one kind into questions of any kind.
func widen[Q Question](questions []Named[Q]) []Named[Question] {
	out := make([]Named[Question], 0, len(questions))
	for _, q := range questions {
		out = append(out, Named[Question]{Name: q.Name, Question: q.Question})
	}
	return out
}

// typed checks every result is of the type asked for.
func typed[R Result](results map[string]Result) (map[string]R, error) {
	out := make(map[string]R, len(results))
	for name, result := range results {
		r, ok := result.(R)
		if !ok {
			var want R
			return nil, fmt.Errorf("%w: expected a %T for %q, got %T", ErrClassifier, want, name, result)
		}
		out[name] = r
	}
	return out, nil
}
