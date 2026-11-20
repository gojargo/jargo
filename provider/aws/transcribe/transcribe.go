// Package transcribe is a streaming speech-to-text service backed by Amazon
// Transcribe's StartStreamTranscription API. Audio streams up as AudioEvents
// over a bidirectional event stream and transcripts come back as
// TranscriptEvents, surfaced as interim and final results. Transcribe finalizes
// per audio segment rather than per turn, so it works best behind a turn
// detector. Credentials and region come from the standard AWS chain
// (environment, shared config, or an IAM role) unless set explicitly.
//
// Transcribe can also identify the spoken language among a set of candidates:
// set LanguageOptions, and each transcript carries the language Transcribe
// identified rather than a configured one.
package transcribe

import (
	"errors"
	"slices"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/settings"
)

const (
	defaultLanguage    = "en-US"
	defaultStability   = "high"
	fallbackSampleRate = 16000
)

// The conflicts in the language identification settings. Transcribe rejects
// each of them only once connected, so it would fail the connection and every
// reconnect after it; they are refused before that instead.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errIdentifyMultipleWithoutOptions = errors.New(
		"transcribe: identify_multiple_languages requires language_options")
	errPreferredWithoutOptions = errors.New("transcribe: preferred_language requires language_options")
	errTooFewLanguageOptions   = errors.New("transcribe: language_options needs at least two languages")
	errPreferredNotAnOption    = errors.New("transcribe: preferred_language must be one of language_options")
	// errSettingsRejected wraps the reason an update was refused.
	errSettingsRejected = errors.New("transcribe: settings update rejected")
	// errUnsupportedLanguage is returned when a session would open with no
	// language to transcribe and no language identification either.
	errUnsupportedLanguage = errors.New("transcribe: unsupported language")
)

// Config configures the Amazon Transcribe streaming STT service.
type Config struct {
	// Region is the AWS region (e.g. us-east-1); empty uses the default chain
	// (AWS_REGION, shared config).
	Region string
	// AccessKeyID and SecretAccessKey set static credentials; leave both empty to
	// use the default AWS credential chain (environment, shared config, IAM role).
	AccessKeyID     string
	SecretAccessKey string
	// SessionToken is the optional session token for temporary credentials.
	SessionToken string
	// Language is the BCP-47 language code of the spoken audio (e.g. en-US,
	// fr-FR); empty uses en-US. It is ignored when LanguageOptions is set.
	Language string
	// LanguageOptions are the languages Transcribe chooses among to identify the
	// spoken language, as Transcribe's language codes (e.g. en-US, es-US): at
	// least two, one locale per language (en-US and en-GB cannot both be
	// listed), and Transcribe recommends no more than five. When set, Language is
	// ignored and each transcript carries whichever language Transcribe
	// identified. Empty uses Language. See
	// https://docs.aws.amazon.com/transcribe/latest/dg/lang-id-stream.html.
	LanguageOptions []string
	// IdentifyMultipleLanguages identifies the language of each segment, for
	// speakers who switch languages mid-conversation, rather than the dominant
	// language of the stream. It requires LanguageOptions.
	IdentifyMultipleLanguages bool
	// PreferredLanguage is the most likely of LanguageOptions, which helps
	// Transcribe identify the language sooner; empty omits it. It requires
	// LanguageOptions.
	PreferredLanguage string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	// Transcribe takes 8000 or 16000 Hz; audio at any other rate is resampled
	// to 16000 Hz before it is sent.
	SampleRate int
	// PartialResultsStabilization enables partial-result stabilization, trading a
	// little accuracy for steadier, lower-latency interim results; nil enables it.
	PartialResultsStabilization *bool
	// PartialResultsStability sets the stabilization level when it is on; empty
	// uses "high".
	PartialResultsStability string `validate:"omitempty,oneof=high medium low"`
	// VocabularyName selects a custom vocabulary; empty omits it.
	VocabularyName string
	// VocabularyFilterName selects a custom vocabulary filter; empty omits it.
	VocabularyFilterName string

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.AWSTranscribeTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
}

// Validate reports whether the configuration is usable. Conflicting language
// identification settings are rejected here, since Transcribe refuses the
// session over them only once it has connected.
func (c Config) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	return validateLanguageIdentification(newSTTSettings(c))
}

// STTSettings is the Transcribe configuration that can change while the pipeline
// runs, through an STTUpdateSettingsFrame. Transcribe binds all of it to a
// connection, so a change to any of it reopens the session. An update whose
// language identification settings conflict is refused whole, with a non-fatal
// error, and the settings in force stay as they were.
type STTSettings struct {
	settings.STT
	// PartialResultsStability is how stable interim results are, trading
	// accuracy against latency: "low" is the most accurate, "high" the fastest.
	PartialResultsStability settings.Opt[string] `settings:"partial_results_stability"`
	// LanguageOptions are the languages Transcribe identifies the spoken one
	// among. See Config.LanguageOptions.
	LanguageOptions settings.Opt[[]string] `settings:"language_options"`
	// IdentifyMultipleLanguages identifies the language of each segment. See
	// Config.IdentifyMultipleLanguages.
	IdentifyMultipleLanguages settings.Opt[bool] `settings:"identify_multiple_languages"`
	// PreferredLanguage is the most likely of LanguageOptions. See
	// Config.PreferredLanguage.
	PreferredLanguage settings.Opt[string] `settings:"preferred_language"`
}

// newSTTSettings is the settings store a configuration starts the service with.
func newSTTSettings(cfg Config) *STTSettings {
	s := &STTSettings{}
	s.Model = settings.Cleared[string]()
	s.Language = settings.Set(defaultLanguage)
	if cfg.Language != "" {
		s.Language = settings.Set(cfg.Language)
	}
	s.PartialResultsStability = settings.Set(defaultStability)
	if cfg.PartialResultsStability != "" {
		s.PartialResultsStability = settings.Set(cfg.PartialResultsStability)
	}
	s.LanguageOptions = settings.Cleared[[]string]()
	if len(cfg.LanguageOptions) > 0 {
		s.LanguageOptions = settings.Set(slices.Clone(cfg.LanguageOptions))
	}
	s.IdentifyMultipleLanguages = settings.Set(cfg.IdentifyMultipleLanguages)
	s.PreferredLanguage = settings.Cleared[string]()
	if cfg.PreferredLanguage != "" {
		s.PreferredLanguage = settings.Set(cfg.PreferredLanguage)
	}
	return s
}

// languageOptionCodes is the language options, each listed once, since
// Transcribe rejects a repeated code.
func languageOptionCodes(options []string) []string {
	codes := make([]string, 0, len(options))
	for _, code := range options {
		if !slices.Contains(codes, code) {
			codes = append(codes, code)
		}
	}
	return codes
}

// validateLanguageIdentification reports whether the language identification
// settings conflict. Transcribe's rule of one locale per language is left to
// Transcribe: some locales sharing a prefix, such as zh-CN and zh-HK, are
// different languages to it.
func validateLanguageIdentification(s *STTSettings) error {
	options, _ := s.LanguageOptions.Value()
	preferred, _ := s.PreferredLanguage.Value()
	if len(options) == 0 {
		if s.IdentifyMultipleLanguages.Or(false) {
			return errIdentifyMultipleWithoutOptions
		}
		if preferred != "" {
			return errPreferredWithoutOptions
		}
		return nil
	}
	codes := languageOptionCodes(options)
	if len(codes) < 2 {
		return errTooFewLanguageOptions
	}
	if preferred != "" && !slices.Contains(codes, preferred) {
		return errPreferredNotAnOption
	}
	return nil
}
