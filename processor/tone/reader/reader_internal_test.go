package reader

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/tone"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/perception"
	"github.com/gojargo/jargo/processor/tone/internal/tonetest"
)

var (
	errNoSession = errors.New("no session")
	errNoNetwork = errors.New("no network")
)

type read struct{ text, previous string }

// fakeTone answers the tone classifier's questions from fixed tags, the way
// tone.Classifier does, with the model it "loads" in Setup.
type fakeTone struct {
	*classifier.Base
	cfg     tone.Config
	model   string
	err     error
	loadErr error

	mu     sync.Mutex
	loaded bool
	reads  []read
}

func (c *fakeTone) Model() string { return c.model }

func (c *fakeTone) Setup(ctx context.Context) error {
	if err := c.Base.Setup(ctx); err != nil {
		return err
	}
	if c.loadErr != nil {
		return fmt.Errorf("%w: couldn't load the tone model: %w", classifier.ErrClassifier, c.loadErr)
	}
	c.mu.Lock()
	c.loaded = true
	c.mu.Unlock()
	return nil
}

func (c *fakeTone) Questions() ([]classifier.Named[classifier.Question], error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.loaded {
		return nil, fmt.Errorf("%w: the questions come from the model, loaded in Setup", classifier.ErrClassifier)
	}
	return []classifier.Named[classifier.Question]{
		{Name: tone.Emotion, Question: classifier.ChoiceQuestion{
			Instructions: "Which feeling the turn's words show.",
			Options:      []classifier.Option{{Name: "neutral"}, {Name: "happy"}, {Name: "angry"}, {Name: "sad"}},
		}},
		{Name: tone.Move, Question: classifier.ChoiceQuestion{
			Instructions: "What the turn does in the conversation.",
			Options:      []classifier.Option{{Name: "asks"}, {Name: "answers"}, {Name: "other"}},
		}},
		{Name: tone.Sarcasm, Question: classifier.YesNoQuestion{Instructions: "Is the turn sarcastic?"}},
		{Name: tone.Backchannel, Question: classifier.YesNoQuestion{Instructions: "Is the turn only a listening sound?"}},
	}, nil
}

func (c *fakeTone) AskQuestions(
	_ context.Context, state any, qs []classifier.Named[classifier.Question], _ []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	turn, _ := state.(map[string]any)
	user, _ := turn["user"].(string)
	assistant, _ := turn["assistant"].(string)
	c.mu.Lock()
	c.reads = append(c.reads, read{user, assistant})
	c.mu.Unlock()
	if c.err != nil {
		return nil, nil, fmt.Errorf("%w: couldn't read the turn: %w", classifier.ErrClassifier, c.err)
	}
	all := map[string]classifier.Result{
		tone.Emotion: classifier.ChoiceResult{
			Choice: "angry", Confidence: 0.71,
			Probabilities: map[string]float64{"neutral": 0.2, "happy": 0.04, "angry": 0.71, "sad": 0.05},
		},
		tone.Move:        classifier.ChoiceResult{Choice: "other", Confidence: 0.92},
		tone.Sarcasm:     classifier.YesNoResult{Probability: 0.99},
		tone.Backchannel: classifier.YesNoResult{Probability: 0.01},
	}
	out := map[string]classifier.Result{}
	for _, q := range qs {
		out[q.Name] = all[q.Name]
	}
	return out, nil, nil
}

func (c *fakeTone) read() []read {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.reads)
}

// standIn builds every tone classifier as a fakeTone set up by configure, and
// returns the ones built.
func standIn(t *testing.T, configure func(*fakeTone)) *[]*fakeTone {
	t.Helper()
	built := &[]*fakeTone{}
	old := newTone
	t.Cleanup(func() { newTone = old })
	newTone = func(cfg tone.Config) (toneClassifier, error) {
		// The real classifier names the model the configuration loads.
		actual, err := tone.New(cfg)
		if err != nil {
			return nil, err
		}
		c := &fakeTone{cfg: cfg, model: actual.Model()}
		c.Base = classifier.NewBase("ToneClassifier", "", c)
		if configure != nil {
			configure(c)
		}
		*built = append(*built, c)
		return c, nil
	}
	return built
}

// toneReader is a reader whose tone classifier is a fakeTone set up by
// configure.
func toneReader(t *testing.T, configure func(*fakeTone)) (*Reader, *fakeTone) {
	t.Helper()
	built := standIn(t, configure)
	return newReader(t, Config{}), (*built)[0]
}

func newReader(t *testing.T, cfg Config) *Reader {
	t.Helper()
	r, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func transcription(text string) *frames.TranscriptionFrame {
	return frames.NewTranscriptionFrame(text, "", "now")
}

func names(fs []frames.Frame) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i], _, _ = strings.Cut(f.Name(), "#")
	}
	return out
}

func toneReading() []expressive.PerceptionSignal {
	sarcasm := 0.99
	return []expressive.PerceptionSignal{
		{Name: "emotion", Labels: []expressive.PerceptionLabel{
			{Label: "angry", Confidence: 0.71}, {Label: "neutral", Confidence: 0.2},
		}},
		{Name: "sarcasm", Value: &sarcasm},
	}
}

func TestTheSignalsAreDeclaredAtStart(t *testing.T) {
	r, _ := toneReader(t, nil)

	down, _ := tonetest.Run(t, r, pipeline.Params{})

	metadata := tonetest.Of[*frames.PerceptionMetadataFrame](down)
	if len(metadata) != 1 {
		t.Fatalf("metadata = %v, want one PerceptionMetadataFrame", names(down))
	}
	type term struct {
		name   string
		labels []string
	}
	var got []term
	for _, s := range metadata[0].Perception.Signals {
		got = append(got, term{s.Name, s.Labels})
	}
	want := []term{{"emotion", []string{"neutral", "happy", "angry", "sad"}}, {"sarcasm", nil}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("signals = %+v, want %+v", got, want)
	}
}

func TestTheReadingIsSetOnItsTranscription(t *testing.T) {
	r, _ := toneReader(t, nil)

	down, _ := tonetest.Run(t, r, pipeline.Params{}, transcription("Oh great, a forty minute wait."))

	want := []string{"PerceptionMetadataFrame", "PerceptionFrame", "TranscriptionFrame"}
	if got := names(down); !slices.Equal(got, want) {
		t.Fatalf("down = %v, want %v", got, want)
	}
	if got := tonetest.Of[*frames.TranscriptionFrame](down)[0].Perception; !reflect.DeepEqual(got, toneReading()) {
		t.Errorf("perception = %+v, want %+v", got, toneReading())
	}
}

func TestATurnIsReadAgainstTheBotsWordsBeforeIt(t *testing.T) {
	r, c := toneReader(t, nil)

	tonetest.Run(t, r, pipeline.Params{},
		frames.NewLLMContextAssistantTurnFrame("The next visit is in forty minutes. Sorry.", "now"),
		transcription("Oh great."))

	want := []read{{"Oh great.", "The next visit is in forty minutes. Sorry."}}
	if got := c.read(); !slices.Equal(got, want) {
		t.Errorf("reads = %+v, want %+v", got, want)
	}
}

func TestInterimTranscriptionsAreNotRead(t *testing.T) {
	r, c := toneReader(t, nil)

	tonetest.Run(t, r, pipeline.Params{}, frames.NewInterimTranscriptionFrame("Oh", "", "now"))

	if got := c.read(); len(got) != 0 {
		t.Errorf("reads = %+v, want none", got)
	}
}

func TestAFailedReadIsReportedAndTheTranscriptionStillPasses(t *testing.T) {
	r, _ := toneReader(t, func(c *fakeTone) { c.err = errNoSession })

	down, up := tonetest.Run(t, r, pipeline.Params{}, transcription("Hello."))

	if got := names(down); !slices.Equal(got, []string{"PerceptionMetadataFrame", "TranscriptionFrame"}) {
		t.Fatalf("down = %v", got)
	}
	if len(tonetest.Of[*frames.ErrorFrame](up)) == 0 {
		t.Error("the failed read was not reported")
	}
	if got := tonetest.Of[*frames.TranscriptionFrame](down)[0].Perception; got != nil {
		t.Errorf("perception = %+v, want none", got)
	}
}

func TestAModelThatCantLoadLeavesTheReaderUnusableAndThePipelineRunning(t *testing.T) {
	r, _ := toneReader(t, func(c *fakeTone) { c.loadErr = errNoNetwork })

	down, up := tonetest.Run(t, r, pipeline.Params{}, transcription("Hello there."))

	if r.Usable() {
		t.Error("the reader is still usable")
	}
	if len(tonetest.Of[*frames.ErrorFrame](up)) == 0 {
		t.Error("the failed load was not reported")
	}
	if len(tonetest.Of[*frames.PerceptionMetadataFrame](down)) != 0 {
		t.Error("signals were declared without a model")
	}
	ts := tonetest.Of[*frames.TranscriptionFrame](down)
	if len(ts) != 1 || ts[0].Perception != nil {
		t.Errorf("transcriptions = %+v, want one passed on unread", ts)
	}
}

func TestTheModelDownloadsAtTheReleaseAskedFor(t *testing.T) {
	built := standIn(t, nil)
	for _, cfg := range []Config{{}, {Revision: "reader-rc9"}} {
		tonetest.Run(t, newReader(t, cfg), pipeline.Params{})
	}

	var asked []string
	for _, c := range *built {
		asked = append(asked, c.Model())
	}
	want := []string{"pipecat-ai/tone-models@reader-rc2", "pipecat-ai/tone-models@reader-rc9"}
	if !slices.Equal(asked, want) {
		t.Errorf("downloaded %v, want %v", asked, want)
	}
}

func TestAFolderOfYourOwnLoadsInPlaceOfADownload(t *testing.T) {
	dir := t.TempDir()
	built := standIn(t, nil)

	down, _ := tonetest.Run(t, newReader(t, Config{ModelDir: dir, Precision: tone.FP32}), pipeline.Params{})

	c := (*built)[0]
	if c.Model() != dir || c.cfg.ModelDir != dir || c.cfg.Precision != tone.FP32 {
		t.Errorf("loaded %q at %q, want %q at fp32", c.Model(), c.cfg.Precision, dir)
	}
	if len(tonetest.Of[*frames.PerceptionMetadataFrame](down)) == 0 {
		t.Error("the signals were not declared")
	}
}

// fakeYesNo answers every question with the same chance.
type fakeYesNo struct{ *classifier.Base }

func (*fakeYesNo) Model() string { return "" }

func (*fakeYesNo) AskQuestions(
	_ context.Context, _ any, qs []classifier.Named[classifier.Question], _ []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	out := map[string]classifier.Result{}
	for _, q := range qs {
		out[q.Name] = classifier.YesNoResult{Probability: 0.3}
	}
	return out, nil, nil
}

// besideTone is a reader that asks the tone classifier and another classifier,
// whose signal is name.
func besideTone(t *testing.T, name string) (*Reader, error) {
	t.Helper()
	standIn(t, nil)
	other := &fakeYesNo{}
	other.Base = classifier.NewBase("FakeYesNoClassifier", "", other)
	otherSource, err := perception.NewSource(perception.SourceConfig{
		Classifier: other,
		Questions: []perception.NamedQuestion{{Name: name, Question: perception.Question{
			Question: classifier.YesNoQuestion{Instructions: "Is the user unsure?"},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	toneSource, err := NewSource(SourceConfig{})
	if err != nil {
		t.Fatal(err)
	}
	return perception.New(perception.Config{Sources: []perception.Source{toneSource, otherSource}})
}

func TestToneReadsATurnBesideAnotherClassifier(t *testing.T) {
	r, err := besideTone(t, "hesitation")
	if err != nil {
		t.Fatal(err)
	}

	down, _ := tonetest.Run(t, r, pipeline.Params{}, transcription("Oh great, a forty minute wait."))

	var declared []string
	for _, term := range tonetest.Of[*frames.PerceptionMetadataFrame](down)[0].Perception.Signals {
		declared = append(declared, term.Name)
	}
	if want := []string{"emotion", "sarcasm", "hesitation"}; !slices.Equal(declared, want) {
		t.Errorf("declared %v, want %v", declared, want)
	}
	hesitation := 0.3
	want := append(toneReading(), expressive.PerceptionSignal{Name: "hesitation", Value: &hesitation})
	ts := tonetest.Of[*frames.TranscriptionFrame](down)
	if len(ts) != 1 || !reflect.DeepEqual(ts[0].Perception, want) {
		t.Errorf("transcriptions = %+v, want one read %+v", ts, want)
	}
}

func TestASourceBesideToneCantReportASignalOfTonesName(t *testing.T) {
	_, err := besideTone(t, "sarcasm")

	if !errors.Is(err, perception.ErrInvalidConfig) || !strings.Contains(err.Error(), "sarcasm") {
		t.Errorf("err = %v", err)
	}
}

func TestTonesNamesAreItsQuestionsOnceTheModelLoads(t *testing.T) {
	r, _ := toneReader(t, nil)

	tonetest.Run(t, r, pipeline.Params{})

	source := r.Sources()[0]
	questions, err := source.Questions()
	if err != nil {
		t.Fatal(err)
	}
	var asked []string
	for _, q := range questions {
		asked = append(asked, q.Name)
	}
	if !slices.Equal(source.Names(), asked) {
		t.Errorf("names = %v, questions = %v", source.Names(), asked)
	}
}

func TestTheReadersTimeoutIsItsSources(t *testing.T) {
	standIn(t, nil)
	if got := newReader(t, Config{}).Sources()[0].Timeout(); got != 500*time.Millisecond {
		t.Errorf("default timeout = %v, want 500ms", got)
	}
	timeout := 200 * time.Millisecond
	if got := newReader(t, Config{Timeout: &timeout}).Sources()[0].Timeout(); got != timeout {
		t.Errorf("timeout = %v, want %v", got, timeout)
	}
}
