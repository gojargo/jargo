package speechify

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// HTTPTTSConfig configures the Speechify HTTP TTS service.
type HTTPTTSConfig struct {
	// APIKey is the Speechify API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the API base; empty uses "https://api.speechify.ai".
	URL string
	// SampleRate is the PCM rate emitted downstream; 0 uses 24 kHz. A rate
	// Speechify does not render (it renders 8000, 16000, 22050, 24000, 44100
	// and 48000) is synthesized at 24 kHz and resampled to this one.
	SampleRate int
	// Model is the Speechify model; empty uses "simba-3.5", which covers 48
	// languages, English included. The legacy simba-english and
	// simba-multilingual models are retired at the API version the service
	// pins, and are refused.
	Model string
	// Voice is the voice id; empty uses "geffen_32".
	Voice string
	// Language for synthesis; the zero value leaves it to Speechify. Mapped to
	// Speechify's language tag.
	Language language.Language
	// LoudnessNormalization normalizes the audio loudness to a standard level,
	// at the cost of latency. Nil leaves it to Speechify.
	LoudnessNormalization *bool
	// TextNormalization spells out numbers, dates and similar tokens before
	// synthesis, at the cost of latency. Nil leaves it to Speechify.
	TextNormalization *bool
	// SafetyIdentifier is a stable, opaque identifier for the end user a
	// request is synthesized for, so Speechify can rate-limit or block that
	// user instead of the whole workspace. 1 to 64 characters of letters,
	// digits and ". _ - : | + / =": a keyed hash of an internal user id, never
	// personal data such as an email address, which Speechify rejects. Empty
	// sends none.
	SafetyIdentifier string
	// Pronunciations says how specific words in the text should be spoken, up
	// to 200 entries. Each entry gives the "text" to match and exactly one of
	// "ipa" (phones from the model's set, for a single word) or "alias"
	// (replacement text), plus an optional "case_sensitive":
	//
	//	[]map[string]any{
	//		{"text": "jargo", "ipa": "dʒ ˈɑː ɹ ɡ oʊ"},
	//		{"text": "TTS", "alias": "text to speech"},
	//	}
	//
	// Speechify applies these on simba-3.5 in every language it lists under
	// pronunciation_languages in GET /v1/audio/models, which excludes English;
	// any other model or language is refused with pronunciations_unsupported.
	// Text transforms on the service (see tts.Base.SetTextTransformers) are
	// another way to guide pronunciation, applied before the text reaches the
	// provider. Nil sends none.
	Pronunciations []map[string]any
	// TextAggregation is how streamed text is grouped before synthesis:
	// frames.TextTypeSentence (the zero value) sends a sentence at a time,
	// frames.TextTypeToken streams the tokens as they arrive.
	TextAggregation frames.TextType
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPTTSConfig) Validate() error { return validate.Struct(c) }

// TTSSettings is the Speechify configuration that can change while the
// pipeline runs, through a TTSUpdateSettingsFrame. Each request is built from
// it, so a change applies from the next sentence on.
type TTSSettings struct {
	settings.TTS

	// LoudnessNormalization normalizes the audio loudness to a standard level.
	LoudnessNormalization settings.Opt[bool] `settings:"loudness_normalization"`
	// TextNormalization spells out numbers, dates and similar tokens.
	TextNormalization settings.Opt[bool] `settings:"text_normalization"`
	// SafetyIdentifier identifies the end user a request is synthesized for.
	SafetyIdentifier settings.Opt[string] `settings:"safety_identifier"`
	// Pronunciations says how specific words in the text should be spoken.
	Pronunciations settings.Opt[[]map[string]any] `settings:"pronunciations"`
}

// newTTSSettings is the starting state, taken from what the service was built
// with.
func newTTSSettings(cfg HTTPTTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Set(cfg.Model)
	s.Voice = settings.Set(cfg.Voice)
	s.Language = settings.Cleared[string]()
	if cfg.Language != "" {
		s.Language = settings.Set(string(cfg.Language))
	}
	s.LoudnessNormalization = optPtr(cfg.LoudnessNormalization)
	s.TextNormalization = optPtr(cfg.TextNormalization)
	s.SafetyIdentifier = settings.Cleared[string]()
	if cfg.SafetyIdentifier != "" {
		s.SafetyIdentifier = settings.Set(cfg.SafetyIdentifier)
	}
	s.Pronunciations = settings.Cleared[[]map[string]any]()
	if cfg.Pronunciations != nil {
		s.Pronunciations = settings.Set(cfg.Pronunciations)
	}
	return s
}

// optPtr is a pointer's value as a setting, cleared when nil.
func optPtr[T any](v *T) settings.Opt[T] {
	if v == nil {
		return settings.Cleared[T]()
	}
	return settings.Set(*v)
}

// speechifyLanguages are the languages verified against synthesis, each under
// the tag Speechify takes for it.
//
//nolint:gochecknoglobals // read-only lookup table
var speechifyLanguages = map[language.Language]string{
	language.German:     "de-DE",
	language.English:    "en-US",
	language.Spanish:    "es-ES",
	language.French:     "fr-FR",
	language.Italian:    "it-IT",
	language.Portuguese: "pt-BR",
}

// speechifyLanguage maps a language to Speechify's language tag. A language
// outside the officially supported set is sent as its BCP-47 tag, with a
// warning.
func speechifyLanguage(l language.Language) string {
	return language.Resolve(l, speechifyLanguages, false)
}

// outputFormat picks the Speechify PCM output format for a sample rate, with
// the rate Speechify will actually synthesize at. That differs from the
// requested rate when Speechify has no matching PCM format (32000 Hz, say), so
// the audio is labeled with the returned rate and resampled.
func outputFormat(sampleRate int) (string, int) {
	if slices.Contains(pcmSampleRates, sampleRate) {
		return "pcm_" + strconv.Itoa(sampleRate), sampleRate
	}
	slog.Warn("speechify tts: no PCM output format for the sample rate, synthesizing at the default instead",
		"rate", sampleRate, "default", defaultSampleRate)
	return "pcm_" + strconv.Itoa(defaultSampleRate), defaultSampleRate
}

// NewHTTPTTS builds a Speechify TTS service over the HTTP API. Each sentence is
// one request, whose audio and word-level speech marks stream back as
// server-sent events, so the conversation keeps only what was heard before an
// interruption.
//
// Requests pin Speechify's API version rather than letting the workspace
// default decide it.
func NewHTTPTTS(cfg HTTPTTSConfig) *tts.Base {
	b, _ := newHTTPTTS(cfg)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPTTSConfig) (*tts.Base, *synthesizer) {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &synthesizer{
		cfg:  cfg,
		url:  strings.TrimRight(cfg.URL, "/") + streamPath,
		live: newTTSSettings(cfg),
	}
	s.outputFormat, s.audioSampleRate = outputFormat(cfg.SampleRate)
	b := tts.New("SpeechifyHTTPTTS", s)
	if cfg.TextAggregation == frames.TextTypeToken {
		b.SetTextAggregator(ttstext.NewTokenAggregator())
	}
	return b, s
}

// synthesizer speaks each sentence with one request to the streaming API.
type synthesizer struct {
	cfg HTTPTTSConfig
	url string
	// outputFormat is the pcm_<rate> format requested, and audioSampleRate the
	// rate it is synthesized at.
	outputFormat    string
	audioSampleRate int
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *TTSSettings

	// mu guards cumulativeTime. Speech-mark times restart at zero for every
	// request, so successive utterances in a turn are offset by the duration of
	// everything synthesized before them.
	mu             sync.Mutex
	cumulativeTime float64
}

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Speechify model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.Voice}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *synthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Speechify does, implementing
// tts.LanguageNamer.
func (s *synthesizer) ServiceLanguage(l language.Language) string { return speechifyLanguage(l) }

// resetTurn starts the speech-mark timeline of a turn from zero.
func (s *synthesizer) resetTurn() {
	s.mu.Lock()
	s.cumulativeTime = 0
	s.mu.Unlock()
}

// Start starts the timeline from zero when the pipeline starts.
func (s *synthesizer) Start(context.Context) { s.resetTurn() }

// OnAudioContextInterrupted starts the timeline from zero when the user cuts
// in, implementing tts.AudioContextInterrupter.
func (s *synthesizer) OnAudioContextInterrupted(context.Context, string) { s.resetTurn() }

// OnTurnContextCompleted starts the timeline from zero once the turn has been
// spoken, implementing tts.TurnContextCompleter.
func (s *synthesizer) OnTurnContextCompleted(context.Context, string) { s.resetTurn() }

// buildPayload builds the request body for text, carrying the settings in
// effect for this request.
func (s *synthesizer) buildPayload(text string) map[string]any {
	payload := map[string]any{
		"input":         text,
		"voice_id":      s.live.Voice.AsAny(),
		"model":         s.live.Model.AsAny(),
		"output_format": s.outputFormat,
	}
	if lang := s.live.Language.Or(""); lang != "" {
		payload["language"] = lang
	}
	if id := s.live.SafetyIdentifier.Or(""); id != "" {
		payload["safety_identifier"] = id
	}
	if pronunciations := s.live.Pronunciations.Or(nil); len(pronunciations) > 0 {
		payload["pronunciations"] = pronunciations
	}
	options := map[string]any{}
	if v, ok := s.live.LoudnessNormalization.Value(); ok {
		options["loudness_normalization"] = v
	}
	if v, ok := s.live.TextNormalization.Value(); ok {
		options["text_normalization"] = v
	}
	if len(options) > 0 {
		payload["options"] = options
	}
	return payload
}

// RunTTS synthesizes text with nobody listening for the word timing.
func (s *synthesizer) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	return s.RunTTSTimed(ctx, text, contextID, yield, nil)
}

// RunTTSTimed requests speech for text and streams the audio of each event as
// it arrives, reporting the words its speech marks time, implementing
// tts.WordTimestamps. A failure is yielded as an error in the context's
// stream.
func (s *synthesizer) RunTTSTimed(
	ctx context.Context,
	text, _ string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	err := s.runTTS(ctx, text, yield, word)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTSTimed, returning what fails.
func (s *synthesizer) runTTS(
	ctx context.Context,
	text string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	body, err := json.Marshal(s.buildPayload(text))
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Speechify-Version", apiVersion)
	req.Header.Set("Speechify-Caller", callerName)
	if v := callerVersion(); v != "" {
		req.Header.Set("Speechify-Caller-Version", v)
	}

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		errorText, _ := io.ReadAll(resp.Body)
		return yield(frames.NewErrorFrame("Speechify API error: " + string(errorText)))
	}
	tts.StartUsageMetrics(ctx, text)
	return s.processStream(resp.Body, text, yield, word)
}

// processStream reads the server-sent events of one request. An event may carry
// audio, speech marks, or both, and marks lag their audio, so the last event of
// a stream is often marks-only. The turn's timeline advances past this
// utterance once the stream ends.
func (s *synthesizer) processStream(
	r io.Reader,
	text string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	s.mu.Lock()
	offset := s.cumulativeTime
	s.mu.Unlock()
	words := newSpeechMarkAccumulator(text, offset)
	utteranceDuration := 0.0
	report := func(timed []uctx.WordTiming) error {
		if word == nil || len(timed) == 0 {
			return nil
		}
		return word(timed, tts.WordTimingOptions{})
	}

	err := readSSEEvents(r, func(event sseEvent) error {
		switch event.name {
		case "speech.chunk":
			if audio, _ := event.data["audio"].(string); audio != "" {
				pcm, err := base64.StdEncoding.DecodeString(audio)
				if err != nil {
					return err
				}
				if err := yield(frames.NewTTSAudioRawFrame(pcm, s.audioSampleRate, 1)); err != nil {
					return err
				}
			}
			return report(words.add(event.data["speech_marks"]))
		case "speech.done":
			// Preferred over the last mark's end time, since it includes any
			// trailing silence.
			if ms := number(event.data["audio_duration_ms"]); ms != 0 {
				utteranceDuration = max(utteranceDuration, ms/1000)
			}
		case "speech.error":
			// The status code is already committed once the stream has started,
			// so a failure midway arrives as an event rather than an HTTP error.
			detail, _ := event.data["error"].(map[string]any)
			code, ok := detail["code"]
			if !ok {
				code = "unknown"
			}
			message, ok := detail["message"]
			if !ok {
				message = event.data
			}
			return yield(frames.NewErrorFrame(fmt.Sprintf("Speechify API error (%v): %v", code, message)))
		}
		return nil
	})
	if err != nil {
		return err
	}
	// A word split across marks is still pending if the text ran out mid-run.
	if err := report(words.flush()); err != nil {
		return err
	}
	s.mu.Lock()
	s.cumulativeTime += max(utteranceDuration, words.endTime)
	s.mu.Unlock()
	return nil
}

// Compile-time checks that the synthesizer is driven the way the base drives
// it.
var (
	_ tts.WordTimestamps          = (*synthesizer)(nil)
	_ tts.SettingsHolder          = (*synthesizer)(nil)
	_ tts.LanguageNamer           = (*synthesizer)(nil)
	_ tts.Describer               = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
	_ tts.TurnContextCompleter    = (*synthesizer)(nil)
)
