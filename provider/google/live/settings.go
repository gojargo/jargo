package live

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/service/settings"
)

// Modality is what the model responds with.
type Modality string

// The response modalities.
const (
	// ModalityAudio is spoken responses, the default.
	ModalityAudio Modality = "AUDIO"
	// ModalityText is text responses, which recent Live models may not support.
	ModalityText Modality = "TEXT"
)

// MediaResolution is how finely the model sees images and video.
type MediaResolution string

// The media resolutions.
const (
	// MediaResolutionUnspecified leaves the resolution to the model, the default.
	MediaResolutionUnspecified MediaResolution = "MEDIA_RESOLUTION_UNSPECIFIED"
	// MediaResolutionLow is 64 tokens.
	MediaResolutionLow MediaResolution = "MEDIA_RESOLUTION_LOW"
	// MediaResolutionMedium is 256 tokens.
	MediaResolutionMedium MediaResolution = "MEDIA_RESOLUTION_MEDIUM"
	// MediaResolutionHigh is zoomed reframing with 256 tokens.
	MediaResolutionHigh MediaResolution = "MEDIA_RESOLUTION_HIGH"
)

// StartSensitivity is how likely speech is to be detected.
type StartSensitivity string

// The start-of-speech sensitivities.
const (
	StartSensitivityHigh StartSensitivity = "START_SENSITIVITY_HIGH"
	StartSensitivityLow  StartSensitivity = "START_SENSITIVITY_LOW"
)

// EndSensitivity is how likely detected speech is to be ended.
type EndSensitivity string

// The end-of-speech sensitivities.
const (
	EndSensitivityHigh EndSensitivity = "END_SENSITIVITY_HIGH"
	EndSensitivityLow  EndSensitivity = "END_SENSITIVITY_LOW"
)

// TurnCoverage is which realtime input a user turn covers.
type TurnCoverage string

// The turn coverages.
const (
	// TurnIncludesOnlyActivity covers only the input during detected activity.
	TurnIncludesOnlyActivity TurnCoverage = "TURN_INCLUDES_ONLY_ACTIVITY"
	// TurnIncludesAllInput covers all input since the last turn.
	TurnIncludesAllInput TurnCoverage = "TURN_INCLUDES_ALL_INPUT"
)

// VADParams tunes the server's voice activity detection. A field left at its
// zero value is not sent, so the server's own default applies.
type VADParams struct {
	// Disabled turns the server's detection off.
	Disabled *bool
	// StartSensitivity is how likely speech is to be detected.
	StartSensitivity StartSensitivity
	// EndSensitivity is how likely detected speech is to be ended.
	EndSensitivity EndSensitivity
	// PrefixPaddingMs is the speech required before its start is committed.
	PrefixPaddingMs *int
	// SilenceDurationMs is the silence required before its end is committed.
	SilenceDurationMs *int
}

// ContextWindowCompression bounds the session's context window by sliding it
// once it grows past a threshold.
type ContextWindowCompression struct {
	// Enabled turns compression on.
	Enabled bool
	// TriggerTokens is the size that triggers compression; nil uses 80% of
	// the context window.
	TriggerTokens *int
}

// Proactivity lets the model decide how to behave on its own.
type Proactivity struct {
	// ProactiveAudio lets the model decline to answer, ignoring speech that is
	// not for it or staying silent until the user makes a request.
	ProactiveAudio *bool
}

// The defaults a setting left unset in the Config takes.
const (
	defaultMaxTokens = 4096
	defaultLanguage  = "en-US"
)

// Settings is what the service holds of its configuration while the pipeline
// runs. An update is stored here but does not reach a session already open: the
// Live API takes all of it in the setup message, so it applies the next time a
// session opens, and the service says so when it is updated.
type Settings struct {
	settings.LLM

	// Voice is the prebuilt voice the model speaks with.
	Voice settings.Opt[string] `settings:"voice"`
	// Modalities is what the model responds with.
	Modalities settings.Opt[Modality] `settings:"modalities"`
	// Language is the language the model speaks, as Gemini's locale code. A
	// language it lists is converted ("es" to "es-ES"); any other is sent as it
	// stands.
	Language settings.Opt[language.Language] `settings:"language"`
	// MediaResolution is how finely the model sees images and video.
	MediaResolution settings.Opt[MediaResolution] `settings:"media_resolution"`
	// VAD tunes the server's voice activity detection; nil leaves its defaults.
	VAD settings.Opt[*VADParams] `settings:"vad"`
	// TurnCoverage is which realtime input a user turn covers; empty uses the
	// model's own default.
	TurnCoverage settings.Opt[TurnCoverage] `settings:"turn_coverage"`
	// ContextWindowCompression bounds the context window; nil leaves it off.
	ContextWindowCompression settings.Opt[*ContextWindowCompression] `settings:"context_window_compression"`
	// Thinking configures the model's reasoning. Live thinking models require a
	// level; when none is set, the lowest the model accepts is applied.
	Thinking settings.Opt[*gemini.ThinkingConfig] `settings:"thinking"`
	// EnableAffectiveDialog lets the model adapt to expression and tone.
	EnableAffectiveDialog settings.Opt[bool] `settings:"enable_affective_dialog"`
	// Proactivity lets the model decide how to behave on its own; nil leaves it
	// off.
	Proactivity settings.Opt[*Proactivity] `settings:"proactivity"`
}

// ApplyUpdate merges an update, converting a language it gives to Gemini's
// locale code first, so the store always holds the code the setup sends.
func (s *Settings) ApplyUpdate(delta any) (settings.Changed, error) {
	if d, ok := delta.(*Settings); ok {
		if l, given := d.Language.Value(); given {
			d.Language = settings.Set(geminiLanguage(l))
		}
	}
	return settings.ApplyFields(s, delta)
}

var _ settings.Updater = (*Settings)(nil)

// newSettings is the starting state, taken from what the service was built
// with, with the defaults for what it left unset.
func newSettings(cfg Config) *Settings {
	s := &Settings{}
	s.Model = settings.Set(cfg.Model)
	s.SystemInstruction = settings.Set(cfg.Instructions)
	s.Voice = settings.Set(cfg.Voice)
	s.MaxTokens = settings.Set(defaultMaxTokens)
	if cfg.MaxTokens != nil {
		s.MaxTokens = settings.Set(*cfg.MaxTokens)
	}
	setOptPtr(&s.Temperature, cfg.Temperature)
	setOptPtr(&s.TopP, cfg.TopP)
	setOptPtr(&s.TopK, cfg.TopK)
	setOptPtr(&s.FrequencyPenalty, cfg.FrequencyPenalty)
	setOptPtr(&s.PresencePenalty, cfg.PresencePenalty)
	setOptPtr(&s.Seed, cfg.Seed)
	s.FilterIncompleteUserTurns = settings.Set(false)

	modalities := cfg.Modalities
	if modalities == "" {
		modalities = ModalityAudio
	}
	s.Modalities = settings.Set(modalities)
	lang := language.Language(defaultLanguage)
	if cfg.Language != "" {
		lang = geminiLanguage(cfg.Language)
	}
	s.Language = settings.Set(lang)
	resolution := cfg.MediaResolution
	if resolution == "" {
		resolution = MediaResolutionUnspecified
	}
	s.MediaResolution = settings.Set(resolution)
	s.VAD = settings.Set(cfg.VAD)
	s.TurnCoverage = settings.Set(cfg.TurnCoverage)
	s.ContextWindowCompression = settings.Set(cfg.ContextWindowCompression)
	s.Thinking = settings.Set(cfg.Thinking)
	s.EnableAffectiveDialog = settings.Set(cfg.EnableAffectiveDialog)
	s.Proactivity = settings.Set(cfg.Proactivity)
	return s
}

// setOptPtr gives o the value p points to, or leaves it unset (sent as
// nothing) when p is nil.
func setOptPtr[T any](o *settings.Opt[T], p *T) {
	if p != nil {
		*o = settings.Set(*p)
	} else {
		*o = settings.Cleared[T]()
	}
}

// warnTextModality reports that the text modality may not work, which recent
// Live models do not all support.
func warnTextModality(m Modality) {
	if m == ModalityText {
		slog.Warn("the TEXT modality may not be supported by recent Gemini Live models",
			"modality", string(m))
	}
}

// Settings returns the store a settings update is merged into.
func (s *Service) Settings() any { return s.live }

// UpdateSettings reports every setting an update changed as not applied: the
// Live API takes them all in the setup message, so they reach the session only
// when it next opens.
func (s *Service) UpdateSettings(ctx context.Context, changed settings.Changed) error {
	s.WarnUnhandledUpdatedSettings(ctx, changed.Names())
	return nil
}

// SetModelModalities sets what the model responds with, for the next session.
func (s *Service) SetModelModalities(m Modality) {
	warnTextModality(m)
	s.live.Modalities = settings.Set(m)
}

// SetLanguage sets the language the model speaks, for the next session.
func (s *Service) SetLanguage(l language.Language) {
	code := geminiLanguage(l)
	s.live.Language = settings.Set(code)
	slog.Info("set the Gemini language", "service", s.Name(), "language", string(code))
}

// model is the model the session runs.
func (s *Service) model() string { return s.live.Model.Or("") }

// geminiLanguage is the locale code Gemini Live takes for l: the one it lists
// for the language, or l as it stands when it lists none.
func geminiLanguage(l language.Language) language.Language {
	return language.Language(language.Resolve(l, geminiLanguages, false))
}

// mandarinCN is the code Gemini Live takes for Chinese, which it speaks as
// Mandarin.
const mandarinCN = "cmn-CN"

// geminiLanguages are the locale codes Gemini Live is verified to take, by
// language. A bare language maps to the region the API defaults it to.
//
//nolint:gochecknoglobals // a lookup table
var geminiLanguages = map[language.Language]string{
	language.Arabic:            "ar-XA",
	language.BengaliIN:         "bn-IN",
	language.MandarinChinese:   mandarinCN,
	language.MandarinChineseCN: mandarinCN,
	language.Chinese:           mandarinCN,
	language.ChineseCN:         mandarinCN,
	language.German:            "de-DE",
	language.GermanDE:          "de-DE",
	language.English:           "en-US",
	language.EnglishUS:         "en-US",
	language.EnglishAU:         "en-AU",
	language.EnglishGB:         "en-GB",
	language.EnglishIN:         "en-IN",
	language.Spanish:           "es-ES",
	language.SpanishES:         "es-ES",
	language.SpanishUS:         "es-US",
	language.French:            "fr-FR",
	language.FrenchFR:          "fr-FR",
	language.FrenchCA:          "fr-CA",
	language.Gujarati:          "gu-IN",
	language.GujaratiIN:        "gu-IN",
	language.Hindi:             "hi-IN",
	language.HindiIN:           "hi-IN",
	language.Indonesian:        "id-ID",
	language.IndonesianID:      "id-ID",
	language.Italian:           "it-IT",
	language.ItalianIT:         "it-IT",
	language.Japanese:          "ja-JP",
	language.JapaneseJP:        "ja-JP",
	language.Kannada:           "kn-IN",
	language.KannadaIN:         "kn-IN",
	language.Korean:            "ko-KR",
	language.KoreanKR:          "ko-KR",
	language.Malayalam:         "ml-IN",
	language.MalayalamIN:       "ml-IN",
	language.Marathi:           "mr-IN",
	language.MarathiIN:         "mr-IN",
	language.Dutch:             "nl-NL",
	language.DutchNL:           "nl-NL",
	language.Polish:            "pl-PL",
	language.PolishPL:          "pl-PL",
	language.PortugueseBR:      "pt-BR",
	language.Russian:           "ru-RU",
	language.RussianRU:         "ru-RU",
	language.Tamil:             "ta-IN",
	language.TamilIN:           "ta-IN",
	language.Telugu:            "te-IN",
	language.TeluguIN:          "te-IN",
	language.Thai:              "th-TH",
	language.ThaiTH:            "th-TH",
	language.Turkish:           "tr-TR",
	language.TurkishTR:         "tr-TR",
	language.Vietnamese:        "vi-VN",
	language.VietnameseVN:      "vi-VN",
}
