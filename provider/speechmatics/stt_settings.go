package speechmatics

import (
	"cmp"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
)

// localFields are the settings applied locally, to the text of a transcript,
// so a change to them needs no new session.
//
//nolint:gochecknoglobals // read-only lookup table
var localFields = map[string]bool{"speaker_format": true}

// Settings is the part of the Speechmatics STT configuration that can change
// while the pipeline runs. See Config for what each field means. A change to
// SpeakerFormat takes effect at once; every other field is sent when a session
// opens, so a change to it reopens the session.
type Settings struct {
	settings.STT

	// Domain requests a language pack optimized for a domain.
	Domain settings.Opt[string] `settings:"domain"`
	// TurnDetectionMode is how turns are closed.
	TurnDetectionMode settings.Opt[TurnDetectionMode] `settings:"turn_detection_mode"`
	// SpeakerFormat labels diarized transcripts with their speaker.
	SpeakerFormat settings.Opt[string] `settings:"speaker_format"`
	// KnownSpeakers are the labels and identifiers of known speakers.
	KnownSpeakers settings.Opt[[]SpeakerIdentifier] `settings:"known_speakers"`
	// AdditionalVocab are words to bias the engine towards.
	AdditionalVocab settings.Opt[[]AdditionalVocabEntry] `settings:"additional_vocab"`
	// EnablePartials includes partial segments in the output.
	EnablePartials settings.Opt[bool] `settings:"enable_partials"`
	// EmitSentences closes a segment at every end of sentence.
	EmitSentences settings.Opt[bool] `settings:"emit_sentences"`
	// PunctuationOverrides overrides the punctuation the engine produces.
	PunctuationOverrides settings.Opt[map[string]any] `settings:"punctuation_overrides"`
	// TranscriptFilteringConfig post-processes the transcript.
	TranscriptFilteringConfig settings.Opt[map[string]any] `settings:"transcript_filtering_config"`
	// EnableDiarization has the engine tell speakers apart.
	EnableDiarization settings.Opt[bool] `settings:"enable_diarization"`
	// SpeakerSensitivity is the diarization sensitivity.
	SpeakerSensitivity settings.Opt[float64] `settings:"speaker_sensitivity"`
	// MaxSpeakers is the most speakers the engine clusters words into.
	MaxSpeakers settings.Opt[int] `settings:"max_speakers"`
	// PreferCurrentSpeaker gives words close together extra weight to be the
	// same speaker.
	PreferCurrentSpeaker settings.Opt[bool] `settings:"prefer_current_speaker"`
}

// ApplyUpdate merges an update, implementing settings.Updater. The session is
// built from the model alone, so a change to the model or to the deprecated
// operating_point alias is resolved into the model. Only the fields the update
// changed are resolved, so a new operating_point wins on its own rather than
// clashing with the model already resolved.
func (s *Settings) ApplyUpdate(delta any) (settings.Changed, error) {
	changed, err := settings.ApplyFields(s, delta)
	if err != nil {
		return changed, err
	}
	// An empty model asks for the default.
	if changed.Has("model") && s.Model.Or("") == "" {
		s.Model = settings.Set(defaultModel)
	}
	return changed, nil
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg Config) *Settings {
	s := &Settings{}
	lang := cfg.Language
	if lang == "" {
		lang = language.English
	}
	s.Language = settings.Set(string(lang))
	s.Model = settings.Set(cmp.Or(cfg.Model, defaultModel))
	mode := cfg.TurnDetectionMode
	if mode == "" {
		mode = defaultTurnDetectionMode
	}
	s.TurnDetectionMode = settings.Set(mode)
	if cfg.Domain != "" {
		s.Domain = settings.Set(cfg.Domain)
	}
	if cfg.SpeakerFormat != "" {
		s.SpeakerFormat = settings.Set(cfg.SpeakerFormat)
	}
	if len(cfg.KnownSpeakers) > 0 {
		s.KnownSpeakers = settings.Set(cfg.KnownSpeakers)
	}
	if len(cfg.AdditionalVocab) > 0 {
		s.AdditionalVocab = settings.Set(cfg.AdditionalVocab)
	}
	setOpt(&s.EnablePartials, cfg.EnablePartials)
	setOpt(&s.EmitSentences, cfg.EmitSentences)
	if cfg.PunctuationOverrides != nil {
		s.PunctuationOverrides = settings.Set(cfg.PunctuationOverrides)
	}
	if cfg.TranscriptFilteringConfig != nil {
		s.TranscriptFilteringConfig = settings.Set(cfg.TranscriptFilteringConfig)
	}
	setOpt(&s.EnableDiarization, cfg.EnableDiarization)
	setOpt(&s.SpeakerSensitivity, cfg.SpeakerSensitivity)
	setOpt(&s.MaxSpeakers, cfg.MaxSpeakers)
	setOpt(&s.PreferCurrentSpeaker, cfg.PreferCurrentSpeaker)
	return s
}

// setOpt gives o a value when the configuration carried one.
func setOpt[T any](o *settings.Opt[T], v *T) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

// serviceClosesTurns reports whether Speechmatics detects turns itself and
// reports where they start and end. Every turn detection mode except
// TurnDetectionExternal, where a VADUserStoppedSpeakingFrame closes each turn,
// has the service close turns.
func (s *Settings) serviceClosesTurns() bool {
	mode, ok := s.TurnDetectionMode.Value()
	return ok && mode != TurnDetectionExternal
}

// transcriptionConfig is the transcription_config a session opens with. Only
// the fields that are set are sent, so Speechmatics applies its own default
// for the rest.
func (s *Settings) transcriptionConfig() map[string]any {
	lang := language.Language(s.Language.Or(string(language.English)))
	code := speechmaticsLanguage(lang)
	tc := map[string]any{
		"language": code,
		"model":    s.Model.Or(defaultModel),
	}
	if locale := speechmaticsLocale(code, lang); locale != "" {
		tc["output_locale"] = locale
	}
	diarization := s.EnableDiarization.Or(false)
	if diarization {
		tc["diarization"] = "speaker"
		if dc := s.diarizationConfig(); dc != nil {
			tc["speaker_diarization_config"] = dc
		}
	}
	if vocab, ok := s.AdditionalVocab.Value(); ok && len(vocab) > 0 {
		tc["additional_vocab"] = vocab
	}
	if domain, ok := s.Domain.Value(); ok && domain != "" {
		tc["domain"] = domain
	}
	putOpt(tc, "enable_partials", s.EnablePartials)
	putOpt(tc, "emit_sentences", s.EmitSentences)
	putOpt(tc, "punctuation_overrides", s.PunctuationOverrides)
	putOpt(tc, "transcript_filtering_config", s.TranscriptFilteringConfig)
	return tc
}

// diarizationConfig is the speaker_diarization_config a session opens with, or
// nil when no diarization knob is set, so an empty one is never sent. Only the
// fields that are set are included.
func (s *Settings) diarizationConfig() map[string]any {
	dc := map[string]any{}
	putOpt(dc, "max_speakers", s.MaxSpeakers)
	putOpt(dc, "speaker_sensitivity", s.SpeakerSensitivity)
	putOpt(dc, "prefer_current_speaker", s.PreferCurrentSpeaker)
	if speakers, ok := s.KnownSpeakers.Value(); ok && len(speakers) > 0 {
		dc["speakers"] = speakers
	}
	if len(dc) == 0 {
		return nil
	}
	return dc
}

// putOpt writes a setting into a config only when it carries a value.
func putOpt[T any](m map[string]any, key string, o settings.Opt[T]) {
	if v, ok := o.Value(); ok {
		m[key] = v
	}
}
