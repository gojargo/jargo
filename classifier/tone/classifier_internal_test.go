package tone

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/internal/hfhub"
	"github.com/gojargo/jargo/internal/modelcache"
)

//nolint:gochecknoglobals // test fixtures
var testTags = Tags{
	Emotion: EmotionTag{
		Label:      "angry",
		Intensity:  "low",
		Confidence: 0.71,
		Probabilities: []Probability{
			{"neutral", 0.2}, {"happy", 0.04}, {"angry", 0.71}, {"sad", 0.05},
		},
	},
	Move: MoveTag{
		Label: "answers", Confidence: 0.8,
		Probabilities: []Probability{{"asks", 0.1}, {"answers", 0.8}, {"other", 0.1}},
	},
	Sarcasm:     SarcasmTag{Sarcastic: true, Probability: 0.99},
	Backchannel: BackchannelTag{Probability: 0.01},
}

var (
	errNoSession = errors.New("no session")
	errNoNetwork = errors.New("no network")
	errStop      = errors.New("stop after the download")
)

type turnRead struct{ text, previous string }

type fakeModel struct {
	err   error
	mu    sync.Mutex
	reads []turnRead
}

func (m *fakeModel) Emotions() []string { return []string{"neutral", "happy", "angry", "sad"} }
func (m *fakeModel) Moves() []string    { return []string{"asks", "answers", "other"} }

func (m *fakeModel) Tag(text, previous string) (Tags, error) {
	m.mu.Lock()
	m.reads = append(m.reads, turnRead{text, previous})
	m.mu.Unlock()
	if m.err != nil {
		return Tags{}, m.err
	}
	return testTags, nil
}

func (m *fakeModel) read() []turnRead {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.reads)
}

// unshared makes each test load its own model, and opens models with download
// in place of the real one.
func unshared(t *testing.T, download func(revision string, precision Precision) (model, error)) {
	t.Helper()
	oldModels, oldDownload := models, downloadRelease
	t.Cleanup(func() { models, downloadRelease = oldModels, oldDownload })
	models = &modelcache.Cache[modelKey, model]{}
	downloadRelease = func(_ context.Context, revision string, precision Precision, _ int, _ string) (model, error) {
		return download(revision, precision)
	}
}

// setUp is a classifier set up with m as its download.
func setUp(t *testing.T, m model) *Classifier {
	t.Helper()
	unshared(t, func(string, Precision) (model, error) { return m, nil })
	c, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Setup(t.Context()); err != nil {
		t.Fatal(err)
	}
	return c
}

func ownQuestions(t *testing.T, c *Classifier) []classifier.Named[classifier.Question] {
	t.Helper()
	qs, err := c.Questions()
	if err != nil {
		t.Fatal(err)
	}
	return qs
}

func TestTheQuestionsComeFromTheLoadedModel(t *testing.T) {
	qs := ownQuestions(t, setUp(t, &fakeModel{}))

	var names []string
	for _, q := range qs {
		names = append(names, q.Name)
	}
	if want := []string{"emotion", "move", "sarcasm", "backchannel"}; !slices.Equal(names, want) {
		t.Errorf("questions = %v, want %v", names, want)
	}
	emotion, _ := qs[0].Question.(classifier.ChoiceQuestion)
	if got := emotion.OptionNames(); !slices.Equal(got, []string{"neutral", "happy", "angry", "sad"}) {
		t.Errorf("emotion options = %v", got)
	}
	move, _ := qs[1].Question.(classifier.ChoiceQuestion)
	if got := move.OptionNames(); !slices.Equal(got, []string{"asks", "answers", "other"}) {
		t.Errorf("move options = %v", got)
	}
	if _, ok := qs[2].Question.(classifier.YesNoQuestion); !ok {
		t.Errorf("sarcasm is a %T, want a yes/no question", qs[2].Question)
	}
}

func TestNothingIsAnsweredBeforeTheModelLoads(t *testing.T) {
	c, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := c.Questions(); !errors.Is(err, classifier.ErrClassifier) {
		t.Errorf("questions: err = %v", err)
	}
	_, err = c.Ask(t.Context(), "Hello.", []classifier.Named[classifier.Question]{
		{Name: "sarcasm", Question: classifier.YesNoQuestion{Instructions: "Is it sarcastic?"}},
	})
	if !errors.Is(err, classifier.ErrClassifier) {
		t.Errorf("ask: err = %v", err)
	}
}

func TestEachQuestionIsAnsweredFromOneReadOfTheTurn(t *testing.T) {
	m := &fakeModel{}
	c := setUp(t, m)

	results, err := c.Ask(t.Context(), "Oh great.", ownQuestions(t, c))
	if err != nil {
		t.Fatal(err)
	}

	if got, want := m.read(), []turnRead{{"Oh great.", ""}}; !slices.Equal(got, want) {
		t.Errorf("reads = %v, want %v", got, want)
	}
	wantEmotion := classifier.ChoiceResult{
		Choice:        "angry",
		Probabilities: map[string]float64{"neutral": 0.2, "happy": 0.04, "angry": 0.71, "sad": 0.05},
		Confidence:    0.71,
	}
	if !reflect.DeepEqual(results["emotion"], wantEmotion) {
		t.Errorf("emotion = %+v, want %+v", results["emotion"], wantEmotion)
	}
	if got, _ := results["move"].(classifier.ChoiceResult); got.Choice != "answers" {
		t.Errorf("move = %q, want answers", got.Choice)
	}
	if got := results["sarcasm"]; got != (classifier.YesNoResult{Probability: 0.99}) {
		t.Errorf("sarcasm = %+v", got)
	}
	if got := results["backchannel"]; got != (classifier.YesNoResult{Probability: 0.01}) {
		t.Errorf("backchannel = %+v", got)
	}
}

func TestATurnIsReadWithTheBotsWordsBeforeIt(t *testing.T) {
	m := &fakeModel{}
	c := setUp(t, m)

	_, err := c.Ask(t.Context(),
		map[string]any{"assistant": "Your next visit is in forty minutes.", "user": "Oh great."},
		ownQuestions(t, c)[2:3])
	if err != nil {
		t.Fatal(err)
	}

	if got, want := m.read(), []turnRead{{"Oh great.", "Your next visit is in forty minutes."}}; !slices.Equal(got, want) {
		t.Errorf("reads = %v, want %v", got, want)
	}
}

func TestAQuestionOfItsOwnMakingIsRefusedBeforeReading(t *testing.T) {
	m := &fakeModel{}
	c := setUp(t, m)

	_, err := c.Ask(t.Context(), "Oh great.", []classifier.Named[classifier.Question]{{
		Name: "tone",
		Question: classifier.ChoiceQuestion{
			Instructions: "How is the user?", Options: []classifier.Option{{Name: "fine"}},
		},
	}})

	if !errors.Is(err, classifier.ErrClassifier) {
		t.Errorf("err = %v", err)
	}
	if got := m.read(); len(got) != 0 {
		t.Errorf("reads = %v, want none", got)
	}
}

func TestAStateThatIsntATurnIsRefused(t *testing.T) {
	c := setUp(t, &fakeModel{})

	_, err := c.Ask(t.Context(), map[string]any{"transcript": "Oh great."}, ownQuestions(t, c))

	if !errors.Is(err, classifier.ErrClassifier) {
		t.Errorf("err = %v", err)
	}
}

func TestAFailedReadIsAClassifierError(t *testing.T) {
	c := setUp(t, &fakeModel{err: errNoSession})

	if _, err := c.Ask(t.Context(), "Hello.", ownQuestions(t, c)); !errors.Is(err, classifier.ErrClassifier) {
		t.Errorf("err = %v", err)
	}
}

func TestAModelThatCantLoadFailsSetup(t *testing.T) {
	unshared(t, func(string, Precision) (model, error) { return nil, errNoNetwork })
	c, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}

	if err := c.Setup(t.Context()); !errors.Is(err, classifier.ErrClassifier) {
		t.Errorf("err = %v", err)
	}
}

func TestEveryClassifierSharesTheLoadedModel(t *testing.T) {
	loads := 0
	unshared(t, func(string, Precision) (model, error) {
		loads++
		return &fakeModel{}, nil
	})
	for range 2 {
		c, err := New(Config{})
		if err != nil {
			t.Fatal(err)
		}
		if err := c.Setup(t.Context()); err != nil {
			t.Fatal(err)
		}
	}

	if loads != 1 {
		t.Errorf("loads = %d, want 1", loads)
	}
}

func TestTheModelDownloadsItsFilesFromTheReaderFolder(t *testing.T) {
	var seen hfhub.Config
	old := download
	t.Cleanup(func() { download = old })
	download = func(_ context.Context, cfg hfhub.Config) (string, error) {
		seen = cfg
		return "", errStop
	}

	if _, err := DownloadModel(t.Context(), Revision, FP16, 4, ""); !errors.Is(err, errStop) {
		t.Fatalf("err = %v", err)
	}
	if seen.Repo != "pipecat-ai/tone-models" || seen.Revision != "reader-rc2" ||
		!slices.Equal(seen.Files, []string{"reader/config.json", "reader/tokenizer.json", "reader/model_fp16.onnx"}) {
		t.Errorf("downloaded %+v", seen)
	}
}
