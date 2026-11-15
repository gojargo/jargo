package moonshine

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// errBoom is a model that fails to load.
//
//nolint:gochecknoglobals // sentinel error
var errBoom = errors.New("boom")

// fakeLanguage is a language and the architectures published for it.
type fakeLanguage struct {
	code   string
	models []string
}

// fakeLanguages are the architectures published per language, best first.
// English has the widest range; the other languages have one to four models
// each.
//
//nolint:gochecknoglobals // test fixture
var fakeLanguages = []fakeLanguage{
	{"ar", []string{"tiny-streaming", "base"}},
	{"de", []string{"small-streaming", "tiny-streaming"}},
	{"es", []string{"small-streaming", "tiny-streaming", "base"}},
	{"en", []string{"medium-streaming", "small-streaming", "base", "tiny-streaming", "tiny"}},
	{"ja", []string{"small-streaming", "tiny-streaming", "base", "tiny"}},
	{"ko", []string{"tiny"}},
	{"tl", []string{"tiny-streaming"}},
	{"vi", []string{"tiny-streaming", "base"}},
	{"uk", []string{"base"}},
	{"zh", []string{"tiny-streaming", "base"}},
}

// fakeCatalog resolves models from fakeLanguages to made-up paths, downloading
// nothing.
type fakeCatalog struct{}

func (fakeCatalog) supportedLanguages() []string {
	codes := make([]string, 0, len(fakeLanguages))
	for _, l := range fakeLanguages {
		codes = append(codes, l.code)
	}
	return codes
}

func (c fakeCatalog) supportedLanguagesFriendly() string {
	return strings.Join(c.supportedLanguages(), ", ")
}

func (fakeCatalog) modelForLanguage(_ context.Context, lang string, arch *modelArch) (string, modelArch, error) {
	i := slices.IndexFunc(fakeLanguages, func(l fakeLanguage) bool { return l.code == lang })
	if i < 0 {
		return "", 0, fmt.Errorf("%w: %s", errLanguageNotFound, lang)
	}
	available := fakeLanguages[i].models
	name := available[0]
	if arch != nil {
		name = arch.String()
	}
	if !slices.Contains(available, name) {
		return "", 0, fmt.Errorf("%w: %s and arch: %s", errModelNotFound, lang, name)
	}
	resolved, err := stringToModelArch(name)
	if err != nil {
		return "", 0, err
	}
	return "/models/" + lang + "-" + name, resolved, nil
}

// fakeTranscriber records what it was loaded with and what it was asked to
// transcribe, and always hears the same words.
type fakeTranscriber struct {
	modelPath string
	modelArch modelArch
	options   map[string]string

	mu         sync.Mutex
	audio      []float32
	sampleRate int
	closed     bool
}

func (f *fakeTranscriber) transcribeWithoutStreaming(audio []float32, sampleRate int) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.audio = audio
	f.sampleRate = sampleRate
	return []string{" hello there "}, nil
}

func (f *fakeTranscriber) close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
}

func (f *fakeTranscriber) heard() ([]float32, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.audio, f.sampleRate
}

// fakeLoader loads fakeTranscribers, keeping each one, or fails with err.
type fakeLoader struct {
	mu        sync.Mutex
	instances []*fakeTranscriber
	err       error
}

func (l *fakeLoader) load(modelPath string, arch modelArch, options map[string]string) (transcriber, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.err != nil {
		return nil, l.err
	}
	t := &fakeTranscriber{modelPath: modelPath, modelArch: arch, options: options}
	l.instances = append(l.instances, t)
	return t, nil
}

func (l *fakeLoader) last(t *testing.T) *fakeTranscriber {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.instances) == 0 {
		t.Fatal("no transcriber was loaded")
	}
	return l.instances[len(l.instances)-1]
}

func (l *fakeLoader) fail(err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.err = err
}

// newFakeSTT builds the service over the fakes.
func newFakeSTT(t *testing.T, cfg STTConfig) (*stt.SegmentService, *sttTranscriber, *fakeLoader) {
	t.Helper()
	loader := &fakeLoader{}
	svc, tr, err := newSTT(cfg, loader.load, fakeCatalog{})
	if err != nil {
		t.Fatalf("newSTT: %v", err)
	}
	return svc, tr, loader
}

func storeValues(tr *sttTranscriber) (model, lang string) {
	return tr.store.Model.Or(""), tr.store.Language.Or("")
}

func TestDefaultsLoadTheEnglishStreamingModel(t *testing.T) {
	_, tr, loader := newFakeSTT(t, STTConfig{})

	model, lang := storeValues(tr)
	if model != "small-streaming" || lang != "en" {
		t.Errorf("settings = %q, %q; want small-streaming, en", model, lang)
	}
	last := loader.last(t)
	if last.modelPath != "/models/en-small-streaming" || last.modelArch != archSmallStreaming {
		t.Errorf("loaded %q as %v, want /models/en-small-streaming as small-streaming", last.modelPath, last.modelArch)
	}
}

func TestRegionalVariantsResolveToTheBaseCode(t *testing.T) {
	_, tr, loader := newFakeSTT(t, STTConfig{Language: language.EnglishGB})

	if _, lang := storeValues(tr); lang != "en" {
		t.Errorf("language = %q, want en", lang)
	}
	if got := loader.last(t).modelPath; got != "/models/en-small-streaming" {
		t.Errorf("loaded %q", got)
	}
}

func TestUnavailableArchitectureFallsBackToTheLanguageDefault(t *testing.T) {
	_, tr, loader := newFakeSTT(t, STTConfig{Language: language.UkrainianUA})

	model, lang := storeValues(tr)
	if lang != "uk" || model != "base" {
		t.Errorf("settings = %q, %q; want base, uk", model, lang)
	}
	if got := loader.last(t).modelPath; got != "/models/uk-base" {
		t.Errorf("loaded %q", got)
	}
}

func TestRequestedArchitectureIsKeptWhenAvailable(t *testing.T) {
	_, tr, loader := newFakeSTT(t, STTConfig{Model: ModelTiny, Language: language.Japanese})

	if model, _ := storeValues(tr); model != "tiny" {
		t.Errorf("model = %q, want tiny", model)
	}
	if got := loader.last(t).modelPath; got != "/models/ja-tiny" {
		t.Errorf("loaded %q", got)
	}
}

func TestNonLatinLanguagesRaiseTheTokenRateCap(t *testing.T) {
	for _, tc := range []struct {
		lang language.Language
		want string
	}{
		{language.English, ""},
		{language.Spanish, ""},
		{language.Japanese, "13.0"},
		{language.Chinese, "13.0"},
		{language.Ukrainian, "13.0"},
	} {
		t.Run(string(tc.lang), func(t *testing.T) {
			_, _, loader := newFakeSTT(t, STTConfig{Language: tc.lang})

			if got := loader.last(t).options["max_tokens_per_second"]; got != tc.want {
				t.Errorf("max_tokens_per_second = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestUnsupportedLanguageRaises(t *testing.T) {
	_, _, err := newSTT(STTConfig{Language: language.French}, (&fakeLoader{}).load, fakeCatalog{})
	if !errors.Is(err, errUnsupportedLanguage) || !strings.Contains(err.Error(), "does not support language 'fr'") {
		t.Fatalf("err = %v", err)
	}
}

func TestUnknownModelRaises(t *testing.T) {
	_, _, err := newSTT(STTConfig{Model: "huge"}, (&fakeLoader{}).load, fakeCatalog{})
	if !errors.Is(err, errInvalidArch) || !strings.Contains(err.Error(), "invalid model architecture string") {
		t.Fatalf("err = %v", err)
	}
}

func TestLanguageToMoonshineLanguage(t *testing.T) {
	for in, want := range map[language.Language]string{
		language.EnglishUS: "en",
		language.ChineseTW: "zh",
		language.Ukrainian: "uk",
		language.German:    "de",
		language.Tagalog:   "tl",
		language.Filipino:  "tl",
	} {
		if got := languageToMoonshineLanguage(in); got != want {
			t.Errorf("languageToMoonshineLanguage(%s) = %q, want %q", in, got, want)
		}
	}
}

func TestMoonshineLanguageToFrameLanguage(t *testing.T) {
	if got := moonshineLanguageToFrameLanguage("es"); got != string(language.Spanish) {
		t.Errorf("es = %q", got)
	}
	if got := moonshineLanguageToFrameLanguage(""); got != "" {
		t.Errorf("none = %q", got)
	}
	if got := moonshineLanguageToFrameLanguage("not-a-language"); got != "" {
		t.Errorf("not-a-language = %q", got)
	}
}

func TestServiceExposesMetricsAndLanguageMapping(t *testing.T) {
	svc, tr, _ := newFakeSTT(t, STTConfig{})

	if !svc.CanGenerateMetrics() {
		t.Error("CanGenerateMetrics = false")
	}
	if got := tr.ServiceLanguage(language.EnglishUS); got != "en" {
		t.Errorf("ServiceLanguage(en-US) = %q", got)
	}
}

// runSegment runs one segment of pcm at rate through svc in a pipeline
// whose input runs at rate, and returns the transcripts it produced.
func runSegment(t *testing.T, svc *stt.SegmentService, pcm []byte, rate int) []*frames.TranscriptionFrame {
	t.Helper()
	var mu sync.Mutex
	var got []*frames.TranscriptionFrame
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		Params:                  pipeline.Params{AudioInSampleRate: rate},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			mu.Lock()
			got = append(got, tf)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))
	task.QueueFrame(frames.NewInputAudioRawFrame(pcm, rate, 1))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}))
	// An EndFrame transcribes what is queued before the pipeline stops.
	task.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not stop")
	}
	mu.Lock()
	defer mu.Unlock()
	return got
}

func TestRunSTTYieldsATranscriptionFrame(t *testing.T) {
	svc, _, loader := newFakeSTT(t, STTConfig{Language: language.Spanish})

	got := runSegment(t, svc, make([]byte, 2*160), SampleRate)

	if len(got) != 1 {
		t.Fatalf("got %d transcripts, want 1", len(got))
	}
	if got[0].Text != "hello there" || got[0].Language != string(language.Spanish) {
		t.Errorf("transcript = %q in %q, want %q in es", got[0].Text, got[0].Language, "hello there")
	}
	if _, rate := loader.last(t).heard(); rate != SampleRate {
		t.Errorf("sample rate = %d, want %d", rate, SampleRate)
	}
}

func TestEightKHzInputReachesTheModelAtSixteenKHz(t *testing.T) {
	svc, _, loader := newFakeSTT(t, STTConfig{})
	svc.SetTrailingSilence(0)
	oneSecond := make([]byte, 2*8000)

	runSegment(t, svc, oneSecond, 8000)

	if audio, _ := loader.last(t).heard(); len(audio) != SampleRate {
		t.Errorf("the model heard %d samples, want %d", len(audio), SampleRate)
	}
}

// update sends svc a settings update carrying field set to value.
func update(t *testing.T, svc *stt.SegmentService, field string, value any) {
	t.Helper()
	delta := &STTSettings{}
	if value == nil {
		delta.Model = settings.Cleared[string]()
	} else if err := settings.SetNamed(delta, field, value); err != nil {
		t.Fatalf("SetNamed(%s): %v", field, err)
	}
	f := frames.NewSTTUpdateSettingsFrame(delta)
	if err := svc.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("update: %v", err)
	}
}

// pushedErrors records the errors svc reports.
func pushedErrors(svc *stt.SegmentService) func() []string {
	var mu sync.Mutex
	var errs []string
	events.On(svc.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		mu.Lock()
		errs = append(errs, ef.Error)
		mu.Unlock()
	})
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(errs)
	}
}

func TestLanguageUpdateReloadsTheModel(t *testing.T) {
	svc, tr, loader := newFakeSTT(t, STTConfig{})
	first := loader.last(t)

	update(t, svc, "language", string(language.Ukrainian))

	model, lang := storeValues(tr)
	if lang != "uk" || model != "base" {
		t.Errorf("settings = %q, %q; want base, uk", model, lang)
	}
	last := loader.last(t)
	if tr.current != last || last.modelPath != "/models/uk-base" {
		t.Errorf("transcribing with %q, want the reloaded /models/uk-base", last.modelPath)
	}
	if !first.closed {
		t.Error("the model swapped out was not freed")
	}
}

func TestFailedReloadKeepsTheLoadedModel(t *testing.T) {
	svc, tr, loader := newFakeSTT(t, STTConfig{})
	loaded := tr.current
	errs := pushedErrors(svc)
	loader.fail(errBoom)

	update(t, svc, "language", string(language.Japanese))

	if tr.current != loaded {
		t.Error("a failed reload replaced the loaded model")
	}
	if got := errs(); !slices.Equal(got, []string{"Moonshine model load error: boom"}) {
		t.Errorf("errors = %q", got)
	}
}

func TestMissingOrBlankModelIsRejected(t *testing.T) {
	for name, value := range map[string]any{"none": nil, "blank": ""} {
		t.Run(name, func(t *testing.T) {
			svc, tr, _ := newFakeSTT(t, STTConfig{})
			loaded := tr.current
			errs := pushedErrors(svc)

			update(t, svc, "model", value)

			got := errs()
			if len(got) != 1 || !strings.Contains(got[0], "Moonshine model must be specified") {
				t.Errorf("errors = %q", got)
			}
			if tr.current != loaded {
				t.Error("the loaded model was replaced")
			}
		})
	}
}
