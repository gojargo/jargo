// Package moonshine transcribes speech with Moonshine, a small, fast ASR model
// family that runs on the CPU: no GPU and no API key.
//
// The models run in libmoonshine, Moonshine's native library, which bundles
// its own ONNX Runtime. It is bound through ebitengine/purego rather than cgo,
// so the package always builds and the library is loaded at run time: from
// STTConfig.LibraryPath, or by its conventional name (libmoonshine.so,
// libmoonshine.dylib, moonshine.dll) on the loader's default search path.
// `make deps-moonshine` installs it under .native/.
//
// A model is downloaded once, on first use, from Moonshine's model CDN into a
// cache under the user's cache directory, and checked against its published
// size and CRC32C. STTConfig.ModelPath loads a model already on disk instead.
package moonshine

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// SampleRate is the rate Moonshine reads audio at: 16 kHz mono.
const SampleRate = 16000

// LibPathEnv is the environment variable applications, the runner and the
// eval harness read STTConfig.LibraryPath from. This package reads no
// environment itself.
const LibPathEnv = "JARGO_MOONSHINE_LIB"

// Moonshine truncates a decode whose token rate exceeds max_tokens_per_second.
// Its default of 6.5 is tuned for Latin scripts; the languages below pack more
// tokens into the same speech and need the higher cap Moonshine documents for
// them, or valid output is cut off.
const (
	maxTokensPerSecondOption   = "max_tokens_per_second" //nolint:gosec // an option name, not a credential
	nonLatinMaxTokensPerSecond = "13.0"
)

// nonLatinLanguages are the languages decoded at the raised token rate.
//
//nolint:gochecknoglobals // a read-only set
var nonLatinLanguages = []string{"ar", "ja", "ko", "uk", "zh"}

// Model is a Moonshine model architecture.
//
// The larger models (ModelSmallStreaming, ModelMediumStreaming) ship only in
// streaming form, but transcribe a whole segment in batch just the same.
//
// Which architectures exist depends on the language: English publishes the
// widest range and is the only language with ModelMediumStreaming, while most
// others ship one or two models. An architecture unavailable for the
// configured language falls back to the best one published for it.
type Model string

// The Moonshine model architectures.
const (
	// ModelTiny is the smallest and fastest, lowest accuracy.
	ModelTiny Model = "tiny"
	// ModelBase is a good size and accuracy balance.
	ModelBase Model = "base"
	// ModelTinyStreaming is the streaming-capable tiny.
	ModelTinyStreaming Model = "tiny-streaming"
	// ModelBaseStreaming is the streaming-capable base.
	ModelBaseStreaming Model = "base-streaming"
	// ModelSmallStreaming is larger and more accurate than base (the default).
	ModelSmallStreaming Model = "small-streaming"
	// ModelMediumStreaming is the largest and most accurate.
	ModelMediumStreaming Model = "medium-streaming"
)

// Errors the service reports when it cannot load a model.
var (
	// errModelRequired reports settings that name no model.
	errModelRequired = errors.New("moonshine: Moonshine model must be specified")
	// errLanguageRequired reports settings that name no language.
	errLanguageRequired = errors.New("moonshine: Moonshine requires a language; its models are language-specific")
	// errUnsupportedLanguage reports a language Moonshine publishes no model for.
	errUnsupportedLanguage = errors.New("moonshine: Moonshine does not support language")
)

// STTSettings is the Moonshine configuration that can change while the
// pipeline runs, through an STTUpdateSettingsFrame: the model and the
// language. Models are per language and per architecture, so a change to
// either reloads the model.
type STTSettings struct {
	settings.STT
}

// STTConfig configures the Moonshine STT service.
type STTConfig struct {
	// Model is the architecture to transcribe with; empty uses
	// ModelSmallStreaming. One the language does not publish falls back to
	// the language's best model, and the settings then name that one.
	Model Model
	// Language of the audio; the zero value uses English. Moonshine publishes
	// models for Arabic, Chinese, English, German, Japanese, Korean, Spanish,
	// Tagalog, Ukrainian and Vietnamese; regional variants resolve to their
	// base code.
	Language language.Language
	// LibraryPath is the libmoonshine shared library to load; empty looks for
	// it by its conventional name on the loader's default search path. The
	// library expects its bundled ONNX Runtime beside it, as the
	// moonshine-voice wheel lays it out.
	LibraryPath string
	// ModelPath is a directory holding the model to load, for Model and
	// Language, used instead of the published catalog: nothing is downloaded.
	// A language or model changed while the pipeline runs is loaded from the
	// same directory.
	ModelPath string
	// CacheDir is where models are downloaded; empty uses DefaultCacheDir.
	CacheDir string
	// HTTPClient downloads the models; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	// Segments are resampled to the 16 kHz the model reads either way.
	SampleRate int

	// TTFSP99 overrides the transcript latency the turn strategies size their
	// wait by; nil uses stt.DefaultTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// languageToMoonshineLanguage is the Moonshine code for a language. Moonshine
// publishes one model per language, keyed by base code, so regional variants
// (SpanishMX) resolve to their base code ("es").
func languageToMoonshineLanguage(l language.Language) string {
	codes := map[language.Language]string{
		language.Arabic:     "ar",
		language.German:     "de",
		language.English:    "en",
		language.Spanish:    "es",
		language.Filipino:   "tl", // Filipino maps to Tagalog.
		language.FilipinoPH: "tl",
		language.Japanese:   "ja",
		language.Korean:     "ko",
		language.Tagalog:    "tl",
		language.Ukrainian:  "uk",
		language.Vietnamese: "vi",
		language.Chinese:    "zh",
	}
	return language.Resolve(l, codes, true)
}

// moonshineLanguageToFrameLanguage is the language a transcript is labeled
// with: the Moonshine code when it names a language jargo knows, and none
// otherwise.
func moonshineLanguageToFrameLanguage(code string) string {
	if code != "" && slices.Contains(language.All(), language.Language(code)) {
		return code
	}
	return ""
}

// transcriber is a model loaded into the library.
type transcriber interface {
	// transcribeWithoutStreaming transcribes float samples in [-1, 1] at
	// sampleRate in one call, returning the text of each transcript line.
	transcribeWithoutStreaming(audio []float32, sampleRate int) ([]string, error)
	// close frees the model.
	close()
}

// loadFunc loads the model at modelPath with the given architecture and
// advanced options.
type loadFunc func(modelPath string, arch modelArch, options map[string]string) (transcriber, error)

// NewSTT builds a Moonshine STT service. It is segmented: a VAD upstream
// delimits each utterance, which is transcribed in one call on the CPU.
//
// The library is loaded and the model downloaded (the first time) and loaded
// here, so a missing library, an unsupported language or an unknown model is
// reported now rather than on the first utterance. A missing library is
// ErrLibraryNotAvailable.
//
// The models are MIT-licensed in every language and size, except the legacy
// non-streaming models for languages other than English (ModelTiny and
// ModelBase), which are under the non-commercial Moonshine Community License
// (https://www.moonshine.ai/license).
func NewSTT(cfg STTConfig) (*stt.SegmentService, error) {
	lib, err := openLibrary(cfg.LibraryPath)
	if err != nil {
		return nil, err
	}
	cat, err := cfg.catalog()
	if err != nil {
		return nil, err
	}
	svc, _, err := newSTT(cfg, lib.loadTranscriber, cat)
	return svc, err
}

// catalog is where the configuration has models come from: the directory it
// names, or the published catalog downloaded into the cache.
func (c STTConfig) catalog() (modelCatalog, error) {
	if c.ModelPath != "" {
		return pathCatalog{path: c.ModelPath}, nil
	}
	dir := c.CacheDir
	if dir == "" {
		var err error
		if dir, err = DefaultCacheDir(); err != nil {
			return nil, err
		}
	}
	return downloadCatalog{dl: &downloader{dir: dir, client: c.HTTPClient}}, nil
}

// newSTT builds the service over load and cat, loading the configured model.
func newSTT(cfg STTConfig, load loadFunc, cat modelCatalog) (*stt.SegmentService, *sttTranscriber, error) {
	t := &sttTranscriber{cfg: cfg, load: load, catalog: cat, store: newSTTSettings(cfg)}
	tr, err := t.loadModel(context.Background())
	if err != nil {
		return nil, nil, err
	}
	t.current = tr
	t.language = t.store.Language.Or("")
	t.svc = stt.NewSegment("MoonshineSTT", t, cfg.SampleRate)
	return t.svc, t, nil
}

// newSTTSettings is the settings store a configuration starts the service
// with: the small streaming English model unless it names others.
func newSTTSettings(cfg STTConfig) *STTSettings {
	s := &STTSettings{}
	s.Model = settings.Set(string(ModelSmallStreaming))
	if cfg.Model != "" {
		s.Model = settings.Set(string(cfg.Model))
	}
	s.Language = settings.Set(languageToMoonshineLanguage(language.English))
	if cfg.Language != "" {
		s.Language = settings.Set(languageToMoonshineLanguage(cfg.Language))
	}
	return s
}

// sttTranscriber transcribes segments with the model the settings name.
type sttTranscriber struct {
	cfg     STTConfig
	load    loadFunc
	catalog modelCatalog
	svc     *stt.SegmentService

	// store is the live settings. The service writes it under its own lock
	// and calls UpdateSettings while holding it, which is where the model is
	// reloaded and the store read.
	store *STTSettings

	// mu guards what a segment is transcribed with, which a settings update
	// replaces while a segment may be in flight.
	mu sync.Mutex
	// current is the model loaded.
	current transcriber
	// language is the Moonshine code transcripts are labeled with.
	language string
}

// Settings is the configuration a caller may change while the pipeline runs.
func (t *sttTranscriber) Settings() any { return t.store }

// Metadata names the model in force, which labels what the service reports.
func (t *sttTranscriber) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: t.cfg.TTFSP99, Model: t.store.Model.Or("")}
}

// SegmentSampleRate asks for segments at 16 kHz, the rate Moonshine reads.
func (*sttTranscriber) SegmentSampleRate(int) int { return SampleRate }

// ServiceLanguage names a language the way Moonshine does.
func (*sttTranscriber) ServiceLanguage(l language.Language) string {
	return languageToMoonshineLanguage(l)
}

// UpdateSettings reloads the model when the language or the model changed:
// Moonshine models are per language and per architecture, so either only takes
// effect once the model is reloaded. A model that fails to load is reported,
// and the one already loaded keeps transcribing.
func (t *sttTranscriber) UpdateSettings(ctx context.Context, changed settings.Changed) (bool, error) {
	if changed.Has("language") || changed.Has("model") {
		tr, err := t.loadModel(ctx)
		if err != nil {
			slog.Error("error loading Moonshine model", "service", t.svc.Name(), "err", err)
			t.svc.PushError(ctx, "Moonshine model load error: "+err.Error(), err, false)
		} else {
			t.mu.Lock()
			old := t.current
			t.current = tr
			t.mu.Unlock()
			old.close()
		}
	}
	t.mu.Lock()
	t.language = t.store.Language.Or("")
	t.mu.Unlock()
	return false, nil
}

// loadModel downloads the model the settings name (the first time) and loads
// it. A model the language does not publish is replaced by the language's
// best one, and the settings are rewritten to name it.
func (t *sttTranscriber) loadModel(ctx context.Context) (transcriber, error) {
	slog.Debug("Loading Moonshine model...")
	model, ok := t.store.Model.Value()
	if !ok || model == "" {
		return nil, errModelRequired
	}
	lang, ok := t.store.Language.Value()
	if !ok || lang == "" {
		return nil, errLanguageRequired
	}
	if !slices.Contains(t.catalog.supportedLanguages(), lang) {
		return nil, fmt.Errorf("%w '%s'. Supported languages: %s",
			errUnsupportedLanguage, lang, t.catalog.supportedLanguagesFriendly())
	}

	wanted, err := stringToModelArch(model)
	if err != nil {
		return nil, err
	}
	path, arch, err := t.catalog.modelForLanguage(ctx, lang, &wanted)
	if errors.Is(err, errModelNotFound) {
		// The architecture is not published for this language; take its best.
		if path, arch, err = t.catalog.modelForLanguage(ctx, lang, nil); err != nil {
			return nil, err
		}
		slog.Warn(fmt.Sprintf("Moonshine model '%s' is unavailable for '%s'; using '%s' instead", model, lang, arch))
		t.store.Model = settings.Set(arch.String())
	} else if err != nil {
		return nil, err
	}

	options := map[string]string{}
	if slices.Contains(nonLatinLanguages, lang) {
		options[maxTokensPerSecondOption] = nonLatinMaxTokensPerSecond
	}
	tr, err := t.load(path, arch, options)
	if err != nil {
		return nil, err
	}
	slog.Debug("Loaded Moonshine model")
	return tr, nil
}

// Transcribe transcribes a segment.
func (t *sttTranscriber) Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error) {
	text, _, err := t.TranscribeLanguage(ctx, audio, sampleRate)
	return text, err
}

// TranscribeLanguage transcribes a segment, 16-bit PCM at 16 kHz (the rate
// SegmentSampleRate asks the service for), joining the transcript's lines, and
// returns the language it is labeled with. An empty transcript is no
// transcript.
func (t *sttTranscriber) TranscribeLanguage(_ context.Context, audio []byte, _ int) (string, string, error) {
	// Signed 16-bit samples, divided by 32768 into the [-1, 1] floats
	// Moonshine reads.
	samples := make([]float32, len(audio)/2)
	for i := range samples {
		samples[i] = float32(int16(uint16(audio[2*i])|uint16(audio[2*i+1])<<8)) / 32768.0
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	lines, err := t.current.transcribeWithoutStreaming(samples, SampleRate)
	if err != nil {
		return "", "", err
	}
	text := strings.TrimSpace(strings.Join(lines, " "))
	if text == "" {
		return "", "", nil
	}
	slog.Debug("Transcription", "text", text)
	return text, moonshineLanguageToFrameLanguage(t.language), nil
}
