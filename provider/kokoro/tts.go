package kokoro

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// Errors the service reports for settings it cannot work without.
var (
	// errVoiceRequired reports settings that name no voice.
	errVoiceRequired = errors.New("Kokoro TTS voice must be specified") //nolint:staticcheck // the service's name leads
	// errLanguageRequired reports settings that name no language.
	errLanguageRequired = errors.New("Kokoro TTS language must be specified") //nolint:staticcheck // as above
	// errSpeedRequired reports settings that hold no speed.
	errSpeedRequired = errors.New("kokoro: expected a speed, got none")
	// errVoiceNotInFile reports a voice the voices file does not hold.
	errVoiceNotInFile = errors.New("is not in the voices file")
)

// The espeak-ng voice names several languages share: espeak-ng has one
// French voice for France, and one Mandarin voice for every script.
const (
	espeakFrench   = "fr-fr"
	espeakMandarin = "cmn"
)

// languageToKokoroLanguage is the language kokoro phonemizes text in for l.
//
// The text is phonemized with espeak-ng, so these are espeak-ng voice names
// rather than ISO codes. They differ for Mandarin ("cmn", not "zh") and
// French, which espeak-ng only offers per region ("fr-fr", no bare "fr"); a
// name espeak-ng does not know fails when speech is synthesized.
func languageToKokoroLanguage(l language.Language) string {
	codes := map[language.Language]string{
		language.English:             "en-us",
		language.EnglishUS:           "en-us",
		language.EnglishGB:           "en-gb",
		language.Spanish:             "es",
		language.French:              espeakFrench,
		language.FrenchBE:            "fr-be",
		language.FrenchCA:            espeakFrench,
		language.FrenchCH:            "fr-ch",
		language.FrenchFR:            espeakFrench,
		language.Hindi:               "hi",
		language.Italian:             "it",
		language.Japanese:            "ja",
		language.Portuguese:          "pt",
		language.PortugueseBR:        "pt-br",
		language.MandarinChinese:     espeakMandarin,
		language.YueChineseCantonese: "yue",
		language.Chinese:             espeakMandarin,
		language.ChineseCN:           espeakMandarin,
		language.ChineseHK:           espeakMandarin,
		language.ChineseTW:           espeakMandarin,
	}
	return language.Resolve(l, codes, true)
}

// TTSSettings is the Kokoro configuration that can change while the pipeline
// runs, through a TTSUpdateSettingsFrame: the voice, the language and the
// speed. Each takes effect with the next sentence synthesized.
type TTSSettings struct {
	settings.TTS

	// Speed is the speech rate multiplier, from 0.5 to 2.0.
	Speed settings.Opt[float64] `settings:"speed"`
}

// newTTSSettings is the settings store a configuration starts the service
// with: no model, and English at the normal rate unless it names others.
func newTTSSettings(cfg Config) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Cleared[string]()
	if cfg.Voice != "" {
		s.Voice = settings.Set(cfg.Voice)
	}
	// The language is stored as the caller gave it; the service names it the
	// way espeak-ng does.
	lang := cfg.Language
	if lang == "" {
		lang = language.English
	}
	s.Language = settings.Set(string(lang))
	s.Speed = settings.Set(1.0)
	if cfg.Speed != nil {
		s.Speed = settings.Set(*cfg.Speed)
	}
	return s
}

// requireGiven is the value a setting the service cannot work without holds,
// or err when it holds none or an empty one.
func requireGiven(o settings.Opt[string], err error) (string, error) {
	v, ok := o.Value()
	if !ok || v == "" {
		return "", err
	}
	return v, nil
}

// loadFunc loads the model at modelPath with the voices at voicesPath.
type loadFunc func(modelPath, voicesPath string, espeak EspeakConfig) (kokoroModel, error)

// ensureFunc downloads the model and voices files that do not exist.
type ensureFunc func(ctx context.Context, cfg Config, modelPath, voicesPath string) error

// NewTTS builds a Kokoro TTS service, which synthesizes each sentence locally
// and streams its audio a batch at a time as it is synthesized.
//
// The model and voices files are downloaded here the first time, and loaded,
// along with espeak-ng, so a missing library, a missing voice or a model that
// fails to load is reported now rather than on the first sentence. A missing
// espeak-ng is ErrEspeakNotAvailable.
func NewTTS(cfg Config) (*tts.Base, error) {
	b, _, err := newTTS(cfg, downloadModelFiles, loadKokoro)
	return b, err
}

// downloadModelFiles is ensureModelFiles with the configured client.
func downloadModelFiles(ctx context.Context, cfg Config, modelPath, voicesPath string) error {
	return ensureModelFiles(ctx, cfg.HTTPClient, modelPath, voicesPath)
}

// loadKokoro is newKokoro as a loadFunc.
func loadKokoro(modelPath, voicesPath string, espeak EspeakConfig) (kokoroModel, error) {
	return newKokoro(modelPath, voicesPath, espeak)
}

// newTTS builds the service over ensure and load, returning its synthesizer
// alongside it.
func newTTS(cfg Config, ensure ensureFunc, load loadFunc) (*tts.Base, *synthesizer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = SampleRate
	}
	s := &synthesizer{cfg: cfg, live: newTTSSettings(cfg)}

	modelPath, voicesPath, err := cfg.filePaths()
	if err != nil {
		return nil, nil, err
	}
	if err = ensure(context.Background(), cfg, modelPath, voicesPath); err != nil {
		return nil, nil, err
	}
	model, err := load(modelPath, voicesPath, cfg.Espeak)
	if err != nil {
		return nil, nil, err
	}
	s.model = model

	voice, err := requireGiven(s.live.Voice, errVoiceRequired)
	if err != nil {
		_ = model.close()
		return nil, nil, err
	}
	if !model.hasVoice(voice) {
		_ = model.close()
		return nil, nil, fmt.Errorf("Kokoro TTS voice '%s' %w", voice, errVoiceNotInFile) //nolint:staticcheck // as above
	}
	return tts.New("KokoroTTS", s), s, nil
}

// filePaths is the model and voices files the configuration names, or their
// place in the cache directory when it names none.
func (c Config) filePaths() (string, string, error) {
	modelPath, voicesPath := c.ModelPath, c.VoicesPath
	if modelPath != "" && voicesPath != "" {
		return modelPath, voicesPath, nil
	}
	dir := c.CacheDir
	if dir == "" {
		var err error
		if dir, err = DefaultCacheDir(); err != nil {
			return "", "", err
		}
	}
	if modelPath == "" {
		modelPath = filepath.Join(dir, modelFileName)
	}
	if voicesPath == "" {
		voicesPath = filepath.Join(dir, voicesFileName)
	}
	return modelPath, voicesPath, nil
}

// synthesizer runs the Kokoro model for the service.
type synthesizer struct {
	cfg   Config
	model kokoroModel
	// live is the settings store, which the base updates on the frame that
	// changes it and which each synthesis reads.
	live *TTSSettings
}

// SampleRate is the rate the service outputs. The model's audio is labeled
// with the rate it is synthesized at, and resampled to this one.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way espeak-ng does, implementing
// tts.LanguageNamer.
func (*synthesizer) ServiceLanguage(l language.Language) string { return languageToKokoroLanguage(l) }

// Close releases the model, implementing tts.Closer.
func (s *synthesizer) Close() error { return s.model.close() }

// RunTTS synthesizes text, yielding its audio a batch at a time as the model
// produces it. A failure is yielded as an error frame.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails.
func (s *synthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	tts.StartUsageMetrics(ctx, text)

	voice, err := requireGiven(s.live.Voice, errVoiceRequired)
	if err != nil {
		return err
	}
	lang, err := requireGiven(s.live.Language, errLanguageRequired)
	if err != nil {
		return err
	}
	speed, ok := s.live.Speed.Value()
	if !ok {
		return errSpeedRequired
	}
	for samples, err := range s.model.createStream(ctx, text, voice, lang, speed) {
		if err != nil {
			return err
		}
		if err := yield(frames.NewTTSAudioRawFrame(floatToPCM(samples), SampleRate, 1)); err != nil {
			return err
		}
	}
	return nil
}

// floatToPCM converts samples in [-1, 1] to 16-bit little-endian PCM, scaled
// by 32767 and truncated toward zero.
func floatToPCM(samples []float32) []byte {
	pcm := make([]byte, 2*len(samples))
	for i, v := range samples {
		// Converted through 32 bits, so a sample past full scale wraps the
		// way a C cast does rather than saturating.
		x := uint16(int16(int32(v * 32767))) //nolint:gosec // the wrap is the conversion's
		pcm[2*i] = byte(x)
		pcm[2*i+1] = byte(x >> 8)
	}
	return pcm
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*synthesizer)(nil)
	_ tts.LanguageNamer  = (*synthesizer)(nil)
	_ tts.Closer         = (*synthesizer)(nil)
)
