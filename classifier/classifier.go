// Package classifier answers typed questions about some state.
//
// A classifier is a plain object. Whoever needs answers creates one, keeps it,
// and calls it. It does not sit in a pipeline and no frames flow into it. A
// question is a [YesNoQuestion], a [ChoiceQuestion] or a [ScoreQuestion].
// Questions are asked by name, several about one state at once: Ask takes any
// mix of kinds, and YesNo, Choice and Score take questions of one kind and
// return typed results. Questions can come with inputs alongside the state, of
// the kinds the classifier takes: images ([Image]), clips of audio ([Audio])
// and files such as PDFs or videos ([File]).
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
	"slices"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/audio/resample"
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
	//
	// inputs holds only the kinds the classifier takes, in the order they were
	// given, each clip of an accepted length and at its AudioSampleRate. It is
	// empty, never nil, when the questions came with none.
	AskQuestions(
		ctx context.Context, state any, questions []Named[Question], inputs []Input,
	) (results map[string]Result, usage *frames.LLMTokenUsage, err error)
	// Model is the model that answers, named in the metrics, or "".
	Model() string
}

// Capabilities is what a classifier says about the inputs it takes. Base
// implements each with its default; a classifier that takes more overrides the
// ones that differ, and Base reads them through the classifier it was given.
type Capabilities interface {
	// SupportsImages reports whether questions can come with images. False by
	// default.
	SupportsImages() bool
	// SupportsAudio reports whether questions can come with a clip of audio.
	// False by default.
	SupportsAudio() bool
	// MinAudio is the shortest clip the classifier accepts. Zero by default.
	MinAudio() time.Duration
	// MaxAudio is the longest clip the classifier accepts, or zero for no
	// limit, the default.
	MaxAudio() time.Duration
	// PreferredAudio is how long a clip the classifier answers best: its
	// MinAudio by default.
	PreferredAudio() time.Duration
	// AudioSampleRate is the sample rate clips are resampled to, or zero to keep
	// their own, the default.
	AudioSampleRate() int
	// FileTypes is the MIME types of the files questions can come with; none by
	// default.
	FileTypes() []string
}

// Classifier answers typed questions about a state. Base implements everything
// but Model, which the classifier embedding it provides.
//
// Questions can come with inputs: images, clips of audio and files. A
// classifier says which it takes through its Capabilities and refuses the
// rest. One that hears declares the clips it accepts: from MinAudio to MaxAudio
// long, resampled to AudioSampleRate before AskQuestions gets them. A clip of
// another length is refused. It also says how long a clip it answers best, in
// PreferredAudio.
type Classifier interface {
	Capabilities
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
	// Ask answers several questions of any kinds about one state, with the
	// inputs that come with it.
	Ask(ctx context.Context, state any, questions []Named[Question], inputs ...Input) (map[string]Result, error)
	// YesNo asks whether the state meets each condition.
	YesNo(
		ctx context.Context, state any, questions []Named[YesNoQuestion], inputs ...Input,
	) (map[string]YesNoResult, error)
	// Choice asks which option fits the state, for each question.
	Choice(
		ctx context.Context, state any, questions []Named[ChoiceQuestion], inputs ...Input,
	) (map[string]ChoiceResult, error)
	// Score asks where the state falls on each scale.
	Score(
		ctx context.Context, state any, questions []Named[ScoreQuestion], inputs ...Input,
	) (map[string]ScoreResult, error)
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

// SupportsImages reports whether questions can come with images: not by
// default.
func (b *Base) SupportsImages() bool { return false }

// SupportsAudio reports whether questions can come with a clip of audio: not by
// default.
func (b *Base) SupportsAudio() bool { return false }

// MinAudio is the shortest clip the classifier accepts: zero by default.
func (b *Base) MinAudio() time.Duration { return 0 }

// MaxAudio is the longest clip the classifier accepts, or zero for no limit,
// the default.
func (b *Base) MaxAudio() time.Duration { return 0 }

// PreferredAudio is how long a clip the classifier answers best: its shortest
// by default.
func (b *Base) PreferredAudio() time.Duration { return b.capabilities().MinAudio() }

// AudioSampleRate is the sample rate clips are resampled to, or zero to keep
// their own, the default.
func (b *Base) AudioSampleRate() int { return 0 }

// FileTypes is the MIME types of the files questions can come with: none by
// default.
func (b *Base) FileTypes() []string { return nil }

// capabilities is what the classifier says about its inputs: its own where it
// overrides the defaults, the base's where it does not.
func (b *Base) capabilities() Capabilities {
	if c, ok := b.asker.(Capabilities); ok {
		return c
	}
	return b
}

// Ask answers several questions about one state, one result per name, each of
// the type its question calls for. inputs are the images, clips of audio and
// files that come with the state, of the kinds the classifier takes.
//
// It fails with an error wrapping ErrClassifier when the answers could not be
// produced, or not in time: every classifier answers or fails within a bound of
// its own, so a caller waiting on it is never left hanging. Also when an input
// is of a kind the classifier does not take, or a clip of a length it does not
// accept.
func (b *Base) Ask(
	ctx context.Context, state any, questions []Named[Question], inputs ...Input,
) (map[string]Result, error) {
	for _, q := range questions {
		if s, ok := q.Question.(ScoreQuestion); ok && len(s.Levels) < 2 {
			return nil, fmt.Errorf("%w: %q has %d levels, want at least two",
				ErrInvalidQuestion, q.Name, len(s.Levels))
		}
	}
	checked := make([]Input, 0, len(inputs))
	for _, input := range inputs {
		c, err := b.check(input)
		if err != nil {
			return nil, err
		}
		checked = append(checked, c)
	}
	started := time.Now()
	results, usage, err := b.asker.AskQuestions(ctx, state, questions, checked)
	if err != nil {
		return nil, err
	}
	b.registry.Call(ctx, EventMetrics, b.asker, b.metrics(time.Since(started), usage))
	return results, nil
}

// YesNo asks whether the state meets each condition, and returns how likely each
// answer is yes, by the same names.
func (b *Base) YesNo(
	ctx context.Context, state any, questions []Named[YesNoQuestion], inputs ...Input,
) (map[string]YesNoResult, error) {
	results, err := b.Ask(ctx, state, widen(questions), inputs...)
	if err != nil {
		return nil, err
	}
	return typed[YesNoResult](results)
}

// Choice asks which option fits the state, for each question, and returns the
// option that fits and how likely each one is, by the same names.
func (b *Base) Choice(
	ctx context.Context, state any, questions []Named[ChoiceQuestion], inputs ...Input,
) (map[string]ChoiceResult, error) {
	results, err := b.Ask(ctx, state, widen(questions), inputs...)
	if err != nil {
		return nil, err
	}
	return typed[ChoiceResult](results)
}

// Score asks where the state falls on each scale, and returns the position on
// each scale and how likely each level is, by the same names.
func (b *Base) Score(
	ctx context.Context, state any, questions []Named[ScoreQuestion], inputs ...Input,
) (map[string]ScoreResult, error) {
	results, err := b.Ask(ctx, state, widen(questions), inputs...)
	if err != nil {
		return nil, err
	}
	return typed[ScoreResult](results)
}

// check is the input as AskQuestions gets it, or an error wrapping
// ErrClassifier if the classifier does not take it.
func (b *Base) check(input Input) (Input, error) {
	caps := b.capabilities()
	switch in := input.(type) {
	case Audio:
		return b.clip(caps, in)
	case Image:
		if !caps.SupportsImages() {
			return nil, fmt.Errorf("%w: %s cannot see images", ErrClassifier, b.name)
		}
	case File:
		if !slices.Contains(caps.FileTypes(), in.ContentType) {
			return nil, fmt.Errorf("%w: %s doesn't take %s files", ErrClassifier, b.name, in.ContentType)
		}
	}
	return input, nil
}

// clip is the clip at the classifier's AudioSampleRate, or an error wrapping
// ErrClassifier if the classifier does not accept it.
func (b *Base) clip(caps Capabilities, audio Audio) (Audio, error) {
	if !caps.SupportsAudio() {
		return Audio{}, fmt.Errorf("%w: %s cannot hear audio", ErrClassifier, b.name)
	}
	duration := audio.seconds()
	if least := caps.MinAudio().Seconds(); duration < least {
		return Audio{}, fmt.Errorf("%w: %s needs at least %gs of audio, got %.2fs",
			ErrClassifier, b.name, least, duration)
	}
	if most := caps.MaxAudio().Seconds(); most > 0 && duration > most {
		return Audio{}, fmt.Errorf("%w: %s takes at most %gs of audio, got %.2fs",
			ErrClassifier, b.name, most, duration)
	}
	if rate := caps.AudioSampleRate(); rate > 0 && audio.SampleRate != rate {
		resampled, err := resample.Resample(audio.Audio, audio.SampleRate, rate, 1)
		if err != nil {
			return Audio{}, fmt.Errorf("%w: %s couldn't resample the clip: %w", ErrClassifier, b.name, err)
		}
		audio = Audio{Audio: resampled, SampleRate: rate}
	}
	return audio, nil
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
