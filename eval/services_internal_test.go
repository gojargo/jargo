package eval

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/llmclassifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/onnxrt"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/kokoro"
	"github.com/gojargo/jargo/provider/moonshine"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/tts"
	"gopkg.in/yaml.v3"
)

// The fakes the factories under test build.

type fakeTranscriber struct{}

func (fakeTranscriber) Transcribe(context.Context, []byte, int) (string, error) { return "", nil }

// fakeJudgeLLM is an LLM service a factory returns, keeping the config it was
// given.
type fakeJudgeLLM struct{ config map[string]any }

func (*fakeJudgeLLM) RunInference(context.Context, *frames.LLMContext, llm.InferenceOptions) (string, error) {
	return "", nil
}

// fakeServiceClassifier is a classifier a factory returns, keeping the config
// it was given.
type fakeServiceClassifier struct {
	*classifier.Base
	config map[string]any
}

func (*fakeServiceClassifier) Model() string { return "" }

func (*fakeServiceClassifier) AskQuestions(
	context.Context, any, []classifier.Named[classifier.Question], []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	return nil, nil, errors.New("not implemented") //nolint:err113 // a test double
}

// countingTTS is a stand-in synthesizer: RunTTS yields one audio frame.
type countingTTS struct {
	pcm        []byte
	sampleRate int
	mu         sync.Mutex
	calls      int
}

func (c *countingTTS) SampleRate() int { return c.sampleRate }

func (c *countingTTS) RunTTS(_ context.Context, _, _ string, yield func(frames.Frame) error) error {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	return yield(frames.NewTTSAudioRawFrame(bytes.Clone(c.pcm), c.sampleRate, 1))
}

func (c *countingTTS) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// fakeWebsocketTTS is a WebSocket-streaming synthesizer.
type fakeWebsocketTTS struct{ countingTTS }

func (*fakeWebsocketTTS) ConnectWebsocket(context.Context) error    { return nil }
func (*fakeWebsocketTTS) DisconnectWebsocket(context.Context) error { return nil }
func (*fakeWebsocketTTS) ReceiveMessages(context.Context) error     { return nil }
func (*fakeWebsocketTTS) Connected() bool                           { return true }
func (*fakeWebsocketTTS) Connect(context.Context)                   {}
func (*fakeWebsocketTTS) Disconnect(context.Context)                {}

var _ tts.WebsocketService = (*fakeWebsocketTTS)(nil)

func init() {
	RegisterFactory("evaltest.fake_stt", func(map[string]any) (any, error) {
		return stt.NewSegment("FakeSTT", fakeTranscriber{}, 16000), nil
	})
	RegisterFactory("evaltest.not_an_stt", func(config map[string]any) (any, error) { return config, nil })
	RegisterFactory("evaltest.fake_tts", func(map[string]any) (any, error) {
		return &countingTTS{sampleRate: CacheSampleRate}, nil
	})
	RegisterFactory("evaltest.fake_judge_llm", func(config map[string]any) (any, error) {
		return &fakeJudgeLLM{config: config}, nil
	})
	RegisterFactory("evaltest.fake_classifier", func(config map[string]any) (any, error) {
		c := &fakeServiceClassifier{config: config}
		c.Base = classifier.NewBase("FakeServiceClassifier", "", c)
		return c, nil
	})
	RegisterFactory("evaltest.not_a_judge", func(map[string]any) (any, error) { return [2]string{"FAKE", "JUDGE"}, nil })
}

// captureWarnings records what is logged while fn runs.
func captureWarningText(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)
	fn()
	return buf.String()
}

func judgeEvalConfig(t *testing.T, block string) *JudgeEvalConfig {
	t.Helper()
	var cfg JudgeEvalConfig
	if err := yaml.Unmarshal([]byte(block), &cfg); err != nil {
		t.Fatal(err)
	}
	return &cfg
}

// modelOf is the model an LLM service builds with.
func modelOf(t *testing.T, inf llm.Inferencer) string {
	t.Helper()
	s, ok := inf.(*chat.LLMService)
	if !ok {
		t.Fatalf("got a %T, want a chat LLM service", inf)
	}
	return s.Model()
}

// The judge's config

func TestTheDefaultJudgeClassifiesAndExplainsWithOneLLM(t *testing.T) {
	j, err := NewEvalJudgeFromConfig(nil)
	if err != nil {
		t.Fatal(err)
	}
	c, ok := j.Classifier().(*llmclassifier.Classifier)
	if !ok {
		t.Fatalf("classifier = %T, want an LLM classifier", j.Classifier())
	}
	if j.explainer == nil || j.explainer.inf != c.LLM() {
		t.Fatal("the explainer is not the judging LLM")
	}
	if modelOf(t, c.LLM()) != DefaultOllamaJudgeModel {
		t.Fatalf("model = %q", modelOf(t, c.LLM()))
	}
	settings, _ := c.LLM().(*chat.LLMService).Settings().(*chat.LLMSettings)
	if settings == nil || settings.Extra["reasoning_effort"] != "none" {
		t.Fatalf("settings = %+v, want the default extras", settings)
	}
}

func TestAnExplainerBlockNamesTheLLMThatGivesTheReasons(t *testing.T) {
	j, err := NewEvalJudgeFromConfig(judgeEvalConfig(t,
		"service: ollama\nmodel: a\nexplainer: {service: ollama, model: b}\n"))
	if err != nil {
		t.Fatal(err)
	}
	c, _ := j.Classifier().(*llmclassifier.Classifier)
	if modelOf(t, c.LLM()) != "a" || modelOf(t, j.explainer.inf) != "b" {
		t.Fatalf("models = %q, %q; want a, b", modelOf(t, c.LLM()), modelOf(t, j.explainer.inf))
	}
}

func TestExplainerFalseBuildsNoExplainer(t *testing.T) {
	j, err := NewEvalJudgeFromConfig(judgeEvalConfig(t, "service: ollama\nexplainer: false\n"))
	if err != nil {
		t.Fatal(err)
	}
	if j.explainer != nil {
		t.Fatal("an explainer was built")
	}
}

func TestAllowContinueFalseIsReadFromTheConfig(t *testing.T) {
	j, err := NewEvalJudgeFromConfig(judgeEvalConfig(t, "service: ollama\nallow_continue: false\nexplain_below: 0.5\n"))
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains((classifier.ChoiceQuestion{Options: j.outcomes}).OptionNames(), VerdictContinue) {
		t.Fatal("continue is still offered")
	}
	if j.explainBelow != 0.5 {
		t.Fatalf("explain below = %v, want 0.5", j.explainBelow)
	}
}

func TestJudgeFromConfigUnknownServiceRejected(t *testing.T) {
	if _, err := NewEvalJudgeFromConfig(judgeEvalConfig(t, "service: nope\n")); !errors.Is(err, ErrUnknownService) {
		t.Fatalf("err = %v, want an unknown service", err)
	}
}

func TestJudgeFromConfigFactoryEscapeHatch(t *testing.T) {
	j, err := NewEvalJudgeFromConfig(judgeEvalConfig(t, "factory: evaltest.fake_judge_llm\n"))
	if err != nil {
		t.Fatal(err)
	}
	c, _ := j.Classifier().(*llmclassifier.Classifier)
	if c == nil {
		t.Fatalf("classifier = %T", j.Classifier())
	}
	if _, ok := c.LLM().(*fakeJudgeLLM); !ok {
		t.Fatalf("llm = %T, want the factory's", c.LLM())
	}
}

// A factory's classifier is explained by the default LLM, unless the block
// turns the explainer off, and the rest of the block still applies.
func TestAFactorysClassifierIsExplainedByTheDefaultLLM(t *testing.T) {
	j, err := NewEvalJudgeFromConfig(judgeEvalConfig(t, "factory: evaltest.fake_classifier\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := j.Classifier().(*fakeServiceClassifier); !ok || j.explainer == nil ||
		modelOf(t, j.explainer.inf) != DefaultOllamaJudgeModel {
		t.Fatalf("classifier = %T, explainer = %+v", j.Classifier(), j.explainer)
	}
	j, err = NewEvalJudgeFromConfig(judgeEvalConfig(t,
		"factory: evaltest.fake_classifier\nexplainer: false\nallow_continue: false\n"))
	if err != nil {
		t.Fatal(err)
	}
	if j.explainer != nil || j.allowContinue {
		t.Fatalf("explainer = %+v, allow continue = %v", j.explainer, j.allowContinue)
	}
}

// The classifier

func TestAnLLMBlockBuildsAnLLMClassifierOverIt(t *testing.T) {
	c, err := ClassifierFromConfig(&LLMServiceConfig{Service: "ollama", Model: "a"}, "judge.eval")
	if err != nil {
		t.Fatal(err)
	}
	lc, ok := c.(*llmclassifier.Classifier)
	if !ok || modelOf(t, lc.LLM()) != "a" {
		t.Fatalf("classifier = %T", c)
	}
}

func TestAFactoryReturningAClassifierIsUsedAsIs(t *testing.T) {
	cfg := judgeEvalConfig(t, "factory: evaltest.fake_classifier\nmodel: x\n")
	c, err := ClassifierFromConfig(&cfg.LLMServiceConfig, "judge.eval")
	if err != nil {
		t.Fatal(err)
	}
	fc, ok := c.(*fakeServiceClassifier)
	if !ok {
		t.Fatalf("classifier = %T", c)
	}
	if fc.config["model"] != "x" || fc.config["factory"] != "evaltest.fake_classifier" {
		t.Fatalf("the factory was given %v, want the block", fc.config)
	}
}

func TestAFactoryReturningAnLLMIsClassifiedWith(t *testing.T) {
	c, err := ClassifierFromConfig(&LLMServiceConfig{Factory: "evaltest.fake_judge_llm"}, "judge.eval")
	if err != nil {
		t.Fatal(err)
	}
	lc, ok := c.(*llmclassifier.Classifier)
	if !ok {
		t.Fatalf("classifier = %T", c)
	}
	if _, ok := lc.LLM().(*fakeJudgeLLM); !ok {
		t.Fatalf("llm = %T", lc.LLM())
	}
}

func TestAFactoryReturningAnythingElseIsAnError(t *testing.T) {
	_, err := ClassifierFromConfig(&LLMServiceConfig{Factory: "evaltest.not_a_judge"}, "judge.eval")
	if !errors.Is(err, ErrFactory) || !strings.Contains(err.Error(), "[2]string") {
		t.Fatalf("err = %v", err)
	}
}

func TestAFactoryMustBeARegisteredDottedName(t *testing.T) {
	for _, name := range []string{"nodots", "evaltest.unregistered"} {
		if _, err := ClassifierFromConfig(&LLMServiceConfig{Factory: name}, "judge.eval"); !errors.Is(err, ErrFactory) {
			t.Fatalf("%s: err = %v", name, err)
		}
	}
}

// The STT

func TestSTTUnknownServiceRejected(t *testing.T) {
	if _, err := STTServiceFromConfig(&TranscriptionConfig{Service: "nope"}); !errors.Is(err, ErrUnknownService) {
		t.Fatalf("err = %v", err)
	}
}

func TestSTTFactoryEscapeHatch(t *testing.T) {
	s, err := STTServiceFromConfig(&TranscriptionConfig{Factory: "evaltest.fake_stt"})
	if err != nil || s == nil {
		t.Fatalf("stt = %v, %v", s, err)
	}
	if _, err := STTServiceFromConfig(&TranscriptionConfig{Factory: "evaltest.not_an_stt"}); !errors.Is(err, ErrFactory) {
		t.Fatalf("err = %v, want a factory that returned no STT refused", err)
	}
}

func TestABuiltInSTTWithNoBuilderYetIsAClearError(t *testing.T) {
	_, err := STTServiceFromConfig(&TranscriptionConfig{Service: "whisper"})
	if !errors.Is(err, ErrServiceUnavailable) || !strings.Contains(err.Error(), "RegisterSTTService") {
		t.Fatalf("err = %v", err)
	}
}

// Moonshine is the default transcription service, built with libmoonshine from
// JARGO_MOONSHINE_LIB; a path that names no library is reported before anything
// is loaded or downloaded.
func TestSTTMoonshineIsBuiltFromTheLibraryTheEnvironmentNames(t *testing.T) {
	t.Setenv(moonshine.LibPathEnv, filepath.Join(t.TempDir(), "libmoonshine.so"))
	for _, cfg := range []*TranscriptionConfig{nil, {Service: "moonshine", Model: "base", Language: "es"}} {
		if _, err := STTServiceFromConfig(cfg); !errors.Is(err, moonshine.ErrLibraryNotAvailable) {
			t.Fatalf("err = %v, want the missing library reported", err)
		}
	}
	if _, err := STTServiceFromConfig(&TranscriptionConfig{Language: "xx"}); !errors.Is(err, errUnknownLanguage) {
		t.Fatalf("err = %v, want the unknown language reported", err)
	}
}

// The TTS

func TestCacheKeyExcludesSampleRate(t *testing.T) {
	a := TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "v", Model: "m", SampleRate: 16000})
	b := TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "v", Model: "m", SampleRate: 24000})
	if a != b {
		t.Fatal("the sample rate changed the key")
	}
}

func TestCacheKeyDistinguishesVoice(t *testing.T) {
	if a, b := TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "a"}),
		TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "b"}); a == b {
		t.Fatal("two voices share a key")
	}
}

func TestCacheKeyDistinguishesSpeed(t *testing.T) {
	fast, normal := 1.3, 1.0
	// A faster render of the same text is different audio.
	if TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "v"}) ==
		TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "v", Speed: &fast}) {
		t.Fatal("two speeds share a key")
	}
	// An absent speed and the default rate render the same audio.
	if TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "v"}) !=
		TTSCacheKey(&SpeechConfig{Service: "kokoro", Voice: "v", Speed: &normal}) {
		t.Fatal("the default speed changed the key")
	}
}

func TestCacheKeyDistinguishesLanguage(t *testing.T) {
	// Two configs identical except for language must not collide.
	if TTSCacheKey(&SpeechConfig{Service: "cartesia", Voice: "v", Language: "en"}) ==
		TTSCacheKey(&SpeechConfig{Service: "cartesia", Voice: "v", Language: "zh"}) {
		t.Fatal("two languages share a key")
	}
	// An absent language and an explicit empty one key to the same slot.
	if TTSCacheKey(&SpeechConfig{Service: "cartesia", Voice: "v"}) !=
		TTSCacheKey(&SpeechConfig{Service: "cartesia", Voice: "v", Language: ""}) {
		t.Fatal("an empty language changed the key")
	}
}

func TestSampleRateDefault(t *testing.T) {
	if TTSSampleRate(&SpeechConfig{}) != 16000 || TTSSampleRate(&SpeechConfig{SampleRate: 24000}) != 24000 {
		t.Fatal("wrong sample rate")
	}
}

func TestTTSUnknownServiceRejected(t *testing.T) {
	if _, err := TTSServiceFromConfig(&SpeechConfig{Service: "nope", Voice: "v"}, t.TempDir(), true); !errors.Is(
		err, ErrUnknownService) {
		t.Fatalf("err = %v", err)
	}
}

func TestTTSMissingServiceOrVoiceRejected(t *testing.T) {
	if _, err := TTSServiceFromConfig(&SpeechConfig{}, t.TempDir(), true); err == nil {
		t.Fatal("a block with no service or voice was accepted")
	}
}

func TestTTSFactoryEscapeHatch(t *testing.T) {
	s, err := TTSServiceFromConfig(&SpeechConfig{Factory: "evaltest.fake_tts"}, t.TempDir(), true)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Inner().(*countingTTS); !ok {
		t.Fatalf("inner = %T, want the factory's", s.Inner())
	}
}

// A built-in name with no builder registered is a clear error.
func TestABuiltInTTSWithNoBuilderYetIsAClearError(t *testing.T) {
	registry.Lock()
	registry.tts["builtin-test"] = nil
	registry.Unlock()
	t.Cleanup(func() {
		registry.Lock()
		delete(registry.tts, "builtin-test")
		registry.Unlock()
	})
	_, err := TTSServiceFromConfig(&SpeechConfig{Service: "builtin-test", Voice: "af_heart"}, t.TempDir(), true)
	if !errors.Is(err, ErrServiceUnavailable) || !strings.Contains(err.Error(), "RegisterTTSService") {
		t.Fatalf("err = %v", err)
	}
}

// Kokoro is the built-in user TTS. A language that names nothing is reported
// before anything is downloaded or loaded.
func TestTTSKokoroIsBuiltIn(t *testing.T) {
	registry.RLock()
	build := registry.tts[ttsKokoro]
	registry.RUnlock()
	if build == nil {
		t.Fatal("no builder registered for kokoro")
	}
	_, err := TTSServiceFromConfig(&SpeechConfig{Service: "kokoro", Voice: "af_heart", Language: "xx"}, t.TempDir(), true)
	if !errors.Is(err, errUnknownLanguage) {
		t.Fatalf("err = %v, want the unknown language reported", err)
	}
}

// With the model files already in the cache, `service: kokoro` builds a local
// Kokoro TTS synthesizing at the cache's rate, the wrapper streaming at the
// block's rate. It skips when the files, the ONNX Runtime or espeak-ng are
// absent, rather than downloading the model.
func TestTTSKokoroBuildsTheLocalService(t *testing.T) {
	dir, err := kokoro.DefaultCacheDir()
	if err != nil {
		t.Skip(err)
	}
	for _, f := range []string{"kokoro-v1.0.onnx", "voices-v1.0.bin"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Skipf("Kokoro model files not downloaded: %v", err)
		}
	}
	s, err := TTSServiceFromConfig(&SpeechConfig{Service: "kokoro", Voice: "af_heart", SampleRate: 16000},
		t.TempDir(), true)
	if errors.Is(err, kokoro.ErrEspeakNotAvailable) || errors.Is(err, onnxrt.ErrNotConfigured) ||
		(err != nil && strings.Contains(err.Error(), "onnxrt")) {
		t.Skipf("runtime not available: %v", err)
	}
	if err != nil {
		t.Fatalf("TTSServiceFromConfig: %v", err)
	}
	if rate := s.Inner().SampleRate(); rate != CacheSampleRate {
		t.Errorf("inner rate = %d, want %d", rate, CacheSampleRate)
	}
	if rate := s.SampleRate(); rate != 16000 {
		t.Errorf("wrapper rate = %d, want 16000", rate)
	}
}

// cartesiaLanguage is the language a Cartesia TTS's settings hold.
func cartesiaLanguage(t *testing.T, b *tts.Base) string {
	t.Helper()
	s, ok := b.CurrentSettings().(*cartesia.TTSSettings)
	if !ok {
		t.Fatalf("settings = %T", b.CurrentSettings())
	}
	lang, _ := s.Language.Value()
	return lang
}

func TestLanguageReachesCartesiaSettings(t *testing.T) {
	b, err := cartesiaService(&SpeechConfig{Service: "cartesia", Voice: "v", APIKey: "test-key", Language: "zh"})
	if err != nil {
		t.Fatal(err)
	}
	if got := cartesiaLanguage(t, b); got != "zh" {
		t.Fatalf("language = %q, want zh", got)
	}
}

func TestNoLanguageLeavesCartesiaDefault(t *testing.T) {
	b, err := cartesiaService(&SpeechConfig{Service: "cartesia", Voice: "v", APIKey: "test-key"})
	if err != nil {
		t.Fatal(err)
	}
	if got := cartesiaLanguage(t, b); got != "en" {
		t.Fatalf("language = %q, want en", got)
	}
}

func TestWebsocketServiceRejected(t *testing.T) {
	// RunTTS cannot be driven without a pipeline to manage the connection.
	if _, err := NewCachingTTSService(&fakeWebsocketTTS{}, CachingTTSConfig{CacheKey: "k"}); err == nil {
		t.Fatal("a websocket-streaming TTS was accepted")
	}
}

// The language

func TestCfgLanguage(t *testing.T) {
	for _, v := range []any{nil, "", "   "} {
		if got, err := cfgLanguage(v); got != "" || err != nil {
			t.Fatalf("%q: got %q, %v; want unset", v, got, err)
		}
	}
	for v, want := range map[any]language.Language{
		"zh": language.Chinese, " zh-TW ": language.ChineseTW, language.Spanish: language.Spanish,
	} {
		if got, err := cfgLanguage(v); got != want || err != nil {
			t.Fatalf("%v: got %q, %v; want %q", v, got, err, want)
		}
	}
	// YAML reads a bare `language: no` as false rather than Norwegian, so a
	// value that is not a string is refused rather than passed on.
	for _, v := range []any{"notalang", false} {
		if _, err := cfgLanguage(v); err == nil {
			t.Fatalf("%v was accepted", v)
		}
	}
}

// The cache

// tone is a 220 Hz tone of samples at rate.
func tone(rate, samples int) []byte {
	out := make([]byte, 0, samples*2)
	for i := range samples {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/float64(rate)) * 8000)
		out = append(out, byte(v), byte(uint16(v)>>8)) //nolint:gosec // PCM bytes
	}
	return out
}

// runTTSAudio drives RunTTS and returns the audio it yields and the rates it
// is at.
func runTTSAudio(t *testing.T, s *CachingTTSService, text string) ([]byte, []int) {
	t.Helper()
	var (
		pcm   []byte
		rates []int
	)
	err := s.RunTTS(t.Context(), text, "ctx", func(f frames.Frame) error {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			pcm = append(pcm, a.Audio...)
			if !slices.Contains(rates, a.SampleRate) {
				rates = append(rates, a.SampleRate)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return pcm, rates
}

func wantCached(t *testing.T, s *CachingTTSService, text string, pcm []byte) {
	t.Helper()
	path, err := s.cacheFile(text)
	if err != nil {
		t.Fatal(err)
	}
	got, rate, err := readWAV(path)
	if err != nil || rate != CacheSampleRate || !bytes.Equal(got, pcm) {
		t.Fatalf("cached %d bytes at %d Hz (%v), want %d at %d", len(got), rate, err, len(pcm), CacheSampleRate)
	}
}

func cachingTTS(t *testing.T, inner tts.Synthesizer, dir string, rate int) *CachingTTSService {
	t.Helper()
	s, err := NewCachingTTSService(inner, CachingTTSConfig{CacheKey: "k", CacheDir: dir, SampleRate: rate})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestCacheRoundTripAtTheCacheRate(t *testing.T) {
	dir := t.TempDir()
	pcm := tone(CacheSampleRate, 2400)
	inner := &countingTTS{pcm: pcm, sampleRate: CacheSampleRate}
	s := cachingTTS(t, inner, dir, 16000)

	if got, rates := runTTSAudio(t, s, "hello"); !bytes.Equal(got, pcm) || !slices.Equal(rates, []int{CacheSampleRate}) {
		t.Fatalf("first: %d bytes at %v", len(got), rates)
	}
	wantCached(t, s, "hello", pcm)

	// The second call hits the WAV cache; the inner service is not called again.
	if got, rates := runTTSAudio(t, s, "hello"); !bytes.Equal(got, pcm) || !slices.Equal(rates, []int{CacheSampleRate}) {
		t.Fatalf("second: %d bytes at %v", len(got), rates)
	}
	if inner.count() != 1 {
		t.Fatalf("inner called %d times, want once", inner.count())
	}

	// Another pipeline rate reuses the cached utterance; it is resampled as it
	// plays.
	inner8 := &countingTTS{pcm: pcm, sampleRate: CacheSampleRate}
	s8 := cachingTTS(t, inner8, dir, 8000)
	if got, rates := runTTSAudio(t, s8, "hello"); !bytes.Equal(got, pcm) || !slices.Equal(rates, []int{CacheSampleRate}) {
		t.Fatalf("8 kHz: %d bytes at %v", len(got), rates)
	}
	if inner8.count() != 0 {
		t.Fatalf("inner called %d times, want never", inner8.count())
	}

	// A file at another rate is regenerated at the cache's rate.
	path, _ := s.cacheFile("hello")
	if err := writeWAV(path, tone(16000, 1600), 16000); err != nil {
		t.Fatal(err)
	}
	if got, rates := runTTSAudio(t, s, "hello"); !bytes.Equal(got, pcm) || !slices.Equal(rates, []int{CacheSampleRate}) {
		t.Fatalf("regenerated: %d bytes at %v", len(got), rates)
	}
	if inner.count() != 2 {
		t.Fatalf("inner called %d times, want twice", inner.count())
	}
	wantCached(t, s, "hello", pcm)
}

func TestAudioAtAnotherInnerRateIsCachedAtTheCacheRate(t *testing.T) {
	dir := t.TempDir()
	toneIn := tone(22050, 2205)
	cached, err := resample.Resample(toneIn, 22050, CacheSampleRate, 1)
	if err != nil {
		t.Fatal(err)
	}
	inner := &countingTTS{pcm: toneIn, sampleRate: 22050}
	s := cachingTTS(t, inner, dir, 16000)

	want, _ := resample.Resample(toneIn, 22050, 16000, 1)
	got, err := s.Synthesize(t.Context(), "hello")
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("first: %d bytes (%v), want %d", len(got), err, len(want))
	}
	wantCached(t, s, "hello", cached)
	// Synthesize returns the cached utterance at this service's rate too.
	want, _ = resample.Resample(cached, CacheSampleRate, 16000, 1)
	got, err = s.Synthesize(t.Context(), "hello")
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("second: %d bytes (%v), want %d", len(got), err, len(want))
	}
	if inner.count() != 1 {
		t.Fatalf("inner called %d times, want once", inner.count())
	}
}

// Deprecated paid services

func TestCartesiaByNameWarnsAndBuilds(t *testing.T) {
	var (
		s   *CachingTTSService
		err error
	)
	logged := captureWarningText(t, func() {
		s, err = TTSServiceFromConfig(&SpeechConfig{Service: "cartesia", Voice: "v", APIKey: "test-key"}, "", false)
	})
	if err != nil || s == nil {
		t.Fatalf("tts = %v, %v", s, err)
	}
	if !strings.Contains(logged, "`user.speech.service: cartesia`") {
		t.Fatalf("logged %q, want a deprecation warning", logged)
	}
}

func TestOpenAIByNameWarnsAndBuilds(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")
	var err error
	logged := captureWarningText(t, func() {
		_, err = LLMServiceFromConfig(&LLMServiceConfig{Service: "openai"}, "simulator")
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(logged, "`simulator.service: openai`") {
		t.Fatalf("logged %q, want a deprecation warning", logged)
	}
}

func TestUnknownNamesPointAtTheFactory(t *testing.T) {
	_, err := TTSServiceFromConfig(&SpeechConfig{Service: "elevenlabs", Voice: "v"}, "", true)
	if err == nil || !strings.Contains(err.Error(), "Known: kokoro.") || !strings.Contains(err.Error(), "factory") {
		t.Fatalf("tts err = %v", err)
	}
	_, err = LLMServiceFromConfig(&LLMServiceConfig{Service: "anthropic"}, "judge.eval")
	if err == nil || !strings.Contains(err.Error(), "Known: ollama.") {
		t.Fatalf("llm err = %v", err)
	}
	_, err = STTServiceFromConfig(&TranscriptionConfig{Service: "deepgram"})
	if err == nil || !strings.Contains(err.Error(), "Known: moonshine, whisper.") {
		t.Fatalf("stt err = %v", err)
	}
}

// The config blocks

func TestTheConfigBlocksReadFromYAML(t *testing.T) {
	var judge JudgeConfig
	if err := yaml.Unmarshal([]byte(`
modality: audio
transcription: {service: whisper, language: es, device: cuda}
eval:
  factory: evals.judges.typesafe_classifier
  model: jev-9
  explainer: {service: ollama, model: b}
  explain_below: 0.6
  allow_continue: false
`), &judge); err != nil {
		t.Fatal(err)
	}
	e := judge.Eval
	if e.Factory != "evals.judges.typesafe_classifier" || e.Model != "jev-9" || e.Raw["model"] != "jev-9" ||
		e.Explainer.LLM == nil || e.Explainer.LLM.Model != "b" || *e.ExplainBelow != 0.6 || *e.AllowContinue {
		t.Fatalf("eval = %+v", e)
	}
	if judge.Transcription.Service != "whisper" || judge.Transcription.Language != "es" ||
		judge.Transcription.Raw["device"] != "cuda" {
		t.Fatalf("transcription = %+v", judge.Transcription)
	}
	var user UserConfig
	if err := yaml.Unmarshal([]byte("modality: audio\nspeech: {service: kokoro, voice: af_heart, speed: 1.2}\n"),
		&user); err != nil {
		t.Fatal(err)
	}
	if user.Speech.Voice != "af_heart" || *user.Speech.Speed != 1.2 || user.Speech.Raw["service"] != "kokoro" {
		t.Fatalf("speech = %+v", user.Speech)
	}
}
