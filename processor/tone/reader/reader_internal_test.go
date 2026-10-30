package reader

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/hfhub"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
	"github.com/gojargo/jargo/processor/tone/internal/tonetest"
)

//nolint:gochecknoglobals // test fixtures
var tags = Tags{
	Emotion: EmotionTag{
		Label:      "angry",
		Intensity:  "low",
		Confidence: 0.71,
		Probabilities: []Probability{
			{"neutral", 0.2}, {"happy", 0.04}, {"angry", 0.71}, {"sad", 0.05},
		},
	},
	Move:        MoveTag{Label: "other", Confidence: 0.92},
	Sarcasm:     SarcasmTag{Sarcastic: true, Probability: 0.99},
	Backchannel: BackchannelTag{Probability: 0.01},
}

var (
	errNoSession = errors.New("no session")
	errNoNetwork = errors.New("no network")
)

type read struct{ text, previous string }

type fakeModel struct {
	err   error
	mu    sync.Mutex
	reads []read
}

func (m *fakeModel) Emotions() []string { return []string{"neutral", "happy", "angry", "sad"} }

func (m *fakeModel) Tag(text, previous string) (Tags, error) {
	m.mu.Lock()
	m.reads = append(m.reads, read{text, previous})
	m.mu.Unlock()
	if m.err != nil {
		return Tags{}, m.err
	}
	return tags, nil
}

func (m *fakeModel) read() []read {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.reads)
}

// standIn makes each test load its own model, and opens models with download
// and load in place of the real ones.
func standIn(
	t *testing.T,
	download func(revision string, precision Precision) (model, error),
	load func(dir string, precision Precision) (model, error),
) {
	t.Helper()
	oldModels, oldDownload, oldLoad := models, downloadRelease, loadDir
	t.Cleanup(func() { models, downloadRelease, loadDir = oldModels, oldDownload, oldLoad })
	models = &shared.Cache[modelKey, model]{}
	downloadRelease = func(_ context.Context, revision string, precision Precision, _ int, _ string) (model, error) {
		return download(revision, precision)
	}
	loadDir = func(dir string, precision Precision, _ int) (model, error) {
		if load == nil {
			t.Fatal("loaded a folder")
		}
		return load(dir, precision)
	}
}

// reader is a reader whose model downloads as m.
func reader(t *testing.T, m model) *Reader {
	t.Helper()
	standIn(t, func(string, Precision) (model, error) { return m, nil }, nil)
	return newReader(t, Config{})
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

func TestTheSignalsAreDeclaredAtStart(t *testing.T) {
	down, _ := tonetest.Run(t, reader(t, &fakeModel{}), pipeline.Params{})

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
	down, _ := tonetest.Run(t, reader(t, &fakeModel{}), pipeline.Params{},
		transcription("Oh great, a forty minute wait."))

	if got := names(down); !slices.Equal(got, []string{"PerceptionMetadataFrame", "TranscriptionFrame"}) {
		t.Fatalf("down = %v", got)
	}
	sarcasm := 0.99
	want := []expressive.PerceptionSignal{
		{Name: "emotion", Labels: []expressive.PerceptionLabel{
			{Label: "angry", Confidence: 0.71}, {Label: "neutral", Confidence: 0.2},
		}},
		{Name: "sarcasm", Value: &sarcasm},
	}
	if got := tonetest.Of[*frames.TranscriptionFrame](down)[0].Perception; !reflect.DeepEqual(got, want) {
		t.Errorf("perception = %+v, want %+v", got, want)
	}
}

func TestATurnIsReadAgainstTheBotsWordsBeforeIt(t *testing.T) {
	m := &fakeModel{}
	tonetest.Run(t, reader(t, m), pipeline.Params{},
		frames.NewLLMContextAssistantTurnFrame("The next visit is in forty minutes. Sorry.", "now"),
		transcription("Oh great."))

	want := []read{{"Oh great.", "The next visit is in forty minutes. Sorry."}}
	if got := m.read(); !slices.Equal(got, want) {
		t.Errorf("reads = %+v, want %+v", got, want)
	}
}

func TestInterimTranscriptionsAreNotRead(t *testing.T) {
	m := &fakeModel{}
	tonetest.Run(t, reader(t, m), pipeline.Params{}, frames.NewInterimTranscriptionFrame("Oh", "", "now"))

	if got := m.read(); len(got) != 0 {
		t.Errorf("reads = %+v, want none", got)
	}
}

func TestAFailedReadIsReportedAndTheTranscriptionStillPasses(t *testing.T) {
	down, up := tonetest.Run(t, reader(t, &fakeModel{err: errNoSession}), pipeline.Params{}, transcription("Hello."))

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
	standIn(t, func(string, Precision) (model, error) { return nil, errNoNetwork }, nil)
	r := newReader(t, Config{})

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
	var asked []string
	standIn(t, func(revision string, _ Precision) (model, error) {
		asked = append(asked, revision)
		return &fakeModel{}, nil
	}, nil)
	for _, cfg := range []Config{{}, {Revision: "reader-rc9"}} {
		tonetest.Run(t, newReader(t, cfg), pipeline.Params{})
	}

	if want := []string{"reader-rc2", "reader-rc9"}; !slices.Equal(asked, want) {
		t.Errorf("downloaded %v, want %v", asked, want)
	}
}

var errStop = errors.New("stop after the download")

func TestTheModelDownloadsItsFilesFromTheReaderFolder(t *testing.T) {
	var seen hfhub.Config
	old := download
	t.Cleanup(func() { download = old })
	download = func(_ context.Context, cfg hfhub.Config) (string, error) {
		seen = cfg
		return "", errStop
	}

	if _, err := DownloadModel(context.Background(), Revision, FP16, 4, ""); !errors.Is(err, errStop) {
		t.Fatalf("err = %v", err)
	}
	if seen.Repo != "pipecat-ai/tone-models" || seen.Revision != "reader-rc2" ||
		!slices.Equal(seen.Files, []string{"reader/config.json", "reader/tokenizer.json", "reader/model_fp16.onnx"}) {
		t.Errorf("downloaded %+v", seen)
	}
}

func TestAFolderOfYourOwnLoadsInPlaceOfADownload(t *testing.T) {
	dir := t.TempDir()
	type loaded struct {
		dir       string
		precision Precision
	}
	var got []loaded
	standIn(t,
		func(string, Precision) (model, error) { return nil, errNoNetwork },
		func(d string, p Precision) (model, error) {
			got = append(got, loaded{d, p})
			return &fakeModel{}, nil
		})

	down, _ := tonetest.Run(t, newReader(t, Config{ModelDir: dir, Precision: FP32}), pipeline.Params{})

	if want := []loaded{{dir, FP32}}; !slices.Equal(got, want) {
		t.Errorf("loaded %+v, want %+v", got, want)
	}
	if len(tonetest.Of[*frames.PerceptionMetadataFrame](down)) == 0 {
		t.Error("the signals were not declared")
	}
}
