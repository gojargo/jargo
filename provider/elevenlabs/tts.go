package elevenlabs

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttselevenlabs "github.com/gojargo/jargo/adapter/tts/elevenlabs"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	uctx "github.com/gojargo/jargo/utils/context"
	errs "github.com/gojargo/jargo/utils/errors"
)

// errTTSStatus is returned when the synthesis API responds with a non-200
// status.
//
//nolint:gochecknoglobals // sentinel error
var errTTSStatus = errors.New("elevenlabs: unexpected tts status")

// elevenlabsV25Languages are the languages Eleven Flash v2.5 and Turbo v2.5
// accept as a language_code.
//
//nolint:gochecknoglobals // read-only lookup table
var elevenlabsV25Languages = languageSet(
	"ar", "bg", "cs", "da", "de", "el", "en", "es", "fi", "fil", "fr", "hi", "hr",
	"hu", "id", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk",
	"sv", "ta", "tr", "uk", "vi", "zh",
)

// elevenlabsV3Languages are the languages the Eleven v3 models accept as a
// language_code. A superset of elevenlabsV25Languages.
//
//nolint:gochecknoglobals // read-only lookup table
var elevenlabsV3Languages = languageSet(
	"af", "ar", "as", "az", "be", "bg", "bn", "bs", "ca", "ceb", "cs", "cy", "da",
	"de", "el", "en", "es", "et", "fa", "fi", "fil", "fr", "ga", "gl", "gu", "ha",
	"he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "kn",
	"ko", "ky", "lb", "ln", "lt", "lv", "mk", "ml", "mr", "ms", "ne", "nl", "no",
	"ny", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "sk", "sl", "so", "sr", "sv",
	"sw", "ta", "te", "th", "tr", "uk", "ur", "vi", "zh",
)

// elevenlabsV4Languages are the languages the Eleven v4 models accept as a
// language_code. A superset of elevenlabsV3Languages. This is the set the API
// accepts, which differs from the list ElevenLabs publishes for the models.
//
//nolint:gochecknoglobals // read-only lookup table
var elevenlabsV4Languages = withLanguages(elevenlabsV3Languages,
	"ast", "mi", "mn", "mt", "my", "oc", "or", "tg", "uz", "yue",
)

// elevenlabsModelLanguages are the models that accept a language_code, and the
// languages each one takes. Sending a language a model does not cover is
// rejected with a 400, so the language is checked against this mapping rather
// than only the model id.
//
// Models absent from the mapping take no language code at all. For
// eleven_multilingual_v2 that is deliberate: ElevenLabs documents language_code
// as unsupported for it, and omitting the key is how its auto-detection is
// meant to be used.
//
//nolint:gochecknoglobals // read-only lookup table
var elevenlabsModelLanguages = map[string]map[string]struct{}{
	"eleven_flash_v2_5":        elevenlabsV25Languages,
	"eleven_turbo_v2_5":        elevenlabsV25Languages,
	"eleven_v3":                elevenlabsV3Languages,
	"eleven_v3_conversational": elevenlabsV3Languages,
	"eleven_v4":                elevenlabsV4Languages,
	"eleven_v4_turbo":          elevenlabsV4Languages,
}

// languageSet is the set of the language codes given.
func languageSet(codes ...string) map[string]struct{} {
	return withLanguages(nil, codes...)
}

// withLanguages is a new set holding the codes of base and the codes given.
func withLanguages(base map[string]struct{}, codes ...string) map[string]struct{} {
	set := make(map[string]struct{}, len(base)+len(codes))
	for code := range base {
		set[code] = struct{}{}
	}
	for _, code := range codes {
		set[code] = struct{}{}
	}
	return set
}

// elevenlabsLanguageCode resolves the language_code to send for a model, given
// an ElevenLabs language code or "" to send none. It returns "" when the code
// cannot be used: either the model takes no language code, or it does not cover
// this language. Both cases are logged.
func elevenlabsLanguageCode(model, code string) string {
	if code == "" {
		return ""
	}
	supported, ok := elevenlabsModelLanguages[model]
	if !ok {
		models := slices.Sorted(maps.Keys(elevenlabsModelLanguages))
		slog.Warn("language code not applied: language codes can only be used with "+
			strings.Join(models, ", "), "service", "ElevenLabsTTS", "model", model, "language", code)
		return ""
	}
	if _, ok := supported[code]; !ok {
		slog.Warn(fmt.Sprintf("language code not applied: %s supports %d languages, which don't include %s",
			model, len(supported), code), "service", "ElevenLabsTTS", "model", model, "language", code)
		return ""
	}
	slog.Debug("using language code", "service", "ElevenLabsTTS", "language", code)
	return code
}

// contextUnsupportedModels reject the previous_text context parameter.
//
//nolint:gochecknoglobals // read-only lookup table
var contextUnsupportedModels = map[string]struct{}{
	"eleven_v3":                {},
	"eleven_v3_conversational": {},
}

// VoiceSettings overrides a voice's default generation settings. Fields left nil
// are omitted, so ElevenLabs falls back to the voice's configured defaults.
type VoiceSettings struct {
	Stability       *float64 `json:"stability,omitempty"`
	SimilarityBoost *float64 `json:"similarity_boost,omitempty"`
	Style           *float64 `json:"style,omitempty"`
	UseSpeakerBoost *bool    `json:"use_speaker_boost,omitempty"`
	Speed           *float64 `json:"speed,omitempty"`
}

// PronunciationDictionaryLocator references a pronunciation dictionary to apply
// to the request.
type PronunciationDictionaryLocator struct {
	DictionaryID string `json:"pronunciation_dictionary_id"`
	VersionID    string `json:"version_id,omitempty"`
}

// NewTTS builds an ElevenLabs TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	s := &synthesizer{cfg: cfg, http: &http.Client{}}
	b := tts.New("ElevenLabsTTS", s)
	s.expressive = b.ExpressiveConversation
	return b
}

type synthesizer struct {
	cfg  Config
	http *http.Client
	// expressive reports whether the pipeline has expressive conversation on,
	// in which the audio tags sent mark when the sounds are performed.
	expressive func() bool

	// mu guards the state carried across the sentences of one turn. The base
	// synthesizes a turn's sentences one after another, but the lifecycle
	// callbacks that reset this state arrive on the frame goroutine.
	mu sync.Mutex
	// previousText is everything spoken so far this turn, sent as context on the
	// next request so the prosody carries across sentence boundaries.
	previousText string
	// cumulative is where this turn's timeline stands, in seconds. Each request
	// times its characters from zero, so the offsets of one sentence only line up
	// with the turn once this is added to them.
	cumulative float64
	// acc assembles the timed characters into whole words, carrying a word that
	// runs past the end of one chunk into the next.
	acc uctx.CharAccumulator
}

// resetTurn drops everything carried between the sentences of a turn.
func (s *synthesizer) resetTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.previousText = ""
	s.cumulative = 0
	s.acc.Reset()
}

// Start drops any state left over from a previous run when the pipeline starts.
func (s *synthesizer) Start(context.Context) { s.resetTurn() }

// OnAudioContextInterrupted drops the turn's state when the user cuts in. What
// was being said is not going to be finished, so it is not context for whatever
// the bot says next.
func (s *synthesizer) OnAudioContextInterrupted(context.Context, string) { s.resetTurn() }

// OnTurnContextCompleted drops the turn's state once the turn has been spoken.
// The next turn answers a new question and starts its timeline afresh.
func (s *synthesizer) OnTurnContextCompleted(context.Context, string) { s.resetTurn() }

// ExpressionAdapter implements tts.ExpressionAdapterProvider: audio tags on
// Eleven v3 and v4, and none on the other models.
func (s *synthesizer) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	if strings.HasPrefix(s.cfg.Model, "eleven_v3") || strings.HasPrefix(s.cfg.Model, "eleven_v4") {
		return ttselevenlabs.ElevenLabsAudioTagExpressionAdapter{}
	}
	return ttsadapter.BaseExpressionAdapter{}
}

// Metadata reports the ElevenLabs model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// RunTTS requests speech for text and streams the raw PCM downstream. The
// service always asks for timestamps, so this is the same synthesis as
// RunTTSTimed with nobody listening for the word timing.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	return s.synthesize(ctx, text, yield, nil)
}

// RunTTSTimed synthesizes like RunTTS and reports the word timing ElevenLabs
// returns alongside the audio, implementing tts.WordTimestamps.
func (s *synthesizer) RunTTSTimed(
	ctx context.Context,
	text, _ string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.synthesize(ctx, text, yield, word)
}

// requestBody builds the synthesis payload for text, including the turn's
// context when the model accepts it.
func (s *synthesizer) requestBody(text string) ([]byte, error) {
	payload := map[string]any{
		keyText:    text,
		"model_id": s.cfg.Model,
	}
	s.mu.Lock()
	previous := s.previousText
	s.mu.Unlock()
	if _, unsupported := contextUnsupportedModels[s.cfg.Model]; previous != "" && !unsupported {
		payload["previous_text"] = previous
	}
	if s.cfg.VoiceSettings != nil {
		payload["voice_settings"] = s.cfg.VoiceSettings
	}
	if len(s.cfg.PronunciationDictionaryLocators) > 0 {
		payload["pronunciation_dictionary_locators"] = s.cfg.PronunciationDictionaryLocators
	}
	if s.cfg.ApplyTextNormalization != "" {
		payload["apply_text_normalization"] = s.cfg.ApplyTextNormalization
	}
	if code := elevenlabsLanguageCode(s.cfg.Model, elevenlabsLanguage(s.cfg.Language)); code != "" {
		payload["language_code"] = code
	}
	return json.Marshal(payload)
}

// newRequest builds the POST to the with-timestamps streaming endpoint.
func (s *synthesizer) newRequest(ctx context.Context, body []byte) (*http.Request, error) {
	q := url.Values{}
	q.Set("output_format", outputFormat(s.cfg.SampleRate))
	if s.cfg.OptimizeStreamingLatency != nil {
		q.Set("optimize_streaming_latency", strconv.Itoa(*s.cfg.OptimizeStreamingLatency))
	}
	if s.cfg.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.cfg.EnableLogging))
	}
	endpoint := fmt.Sprintf("%s/v1/text-to-speech/%s/stream/with-timestamps?%s",
		s.cfg.BaseURL, s.cfg.VoiceID, q.Encode())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("xi-api-key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

// synthesize runs one request and streams what comes back: audio downstream, and
// the word timing to word when a caller is listening for it.
func (s *synthesizer) synthesize(
	ctx context.Context,
	text string,
	yield func(f frames.Frame) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	body, err := s.requestBody(text)
	if err != nil {
		return err
	}
	req, err := s.newRequest(ctx, body)
	if err != nil {
		return err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errTTSStatus, resp.StatusCode, msg))
	}
	tts.StartUsageMetrics(ctx, text)
	return s.stream(resp.Body, text, tts.PCMYielder(yield, s.SampleRate()), word)
}

// stream reads the response, one JSON object per line, and folds each one in.
// The turn's timeline and its spoken text are advanced once the whole utterance
// has been read, so a request that fails partway leaves neither half-updated.
func (s *synthesizer) stream(
	r io.Reader,
	text string,
	emit func(pcm []byte) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	u := &utterance{syn: s, emit: emit, word: word}
	reader := bufio.NewReader(r)
	for {
		// Audio arrives base64 on a single line, so a line can be far longer than
		// a scanner's default token; read the whole of it however long it is.
		line, err := reader.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			if lerr := u.line(line); lerr != nil {
				return lerr
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
	}
	return u.finish(text)
}

// httpAlignment times each character of a chunk's audio, in seconds from the
// start of this request's audio rather than of the turn. The HTTP schema names
// these arrays differently from the WebSocket one.
type httpAlignment struct {
	Characters      []string  `json:"characters"`
	StartTimesSecs  []float64 `json:"character_start_times_seconds"`
	EndTimesSeconds []float64 `json:"character_end_times_seconds"`
}

// httpMessage is one line of the with-timestamps stream.
type httpMessage struct {
	AudioBase64         string         `json:"audio_base64"`
	Alignment           *httpAlignment `json:"alignment"`
	NormalizedAlignment *httpAlignment `json:"normalized_alignment"`
}

// selectAlignment picks which of the two alignments a chunk carries to time the
// words against.
//
// The normalized one is what was actually spoken after substitutions: a
// pronunciation dictionary, text normalization, or a non-Latin script rendered
// in Latin characters. The plain one is the input as written. Prefer the
// normalized form only when a pronunciation dictionary is configured, where the
// plain alignment restarts overlap and produce duplicated words; otherwise
// prefer the plain one so the conversation records the text as written. Either
// may be absent, so each falls back to the other.
func selectAlignment(m *httpMessage, preferNormalized bool) *httpAlignment {
	if preferNormalized {
		if m.NormalizedAlignment != nil {
			return m.NormalizedAlignment
		}
		return m.Alignment
	}
	if m.Alignment != nil {
		return m.Alignment
	}
	return m.NormalizedAlignment
}

// stripLeadingSpaces drops the spaces an alignment opens with, but only on the
// first chunk of an utterance, so a turn does not begin with whitespace. On a
// later chunk a leading space is a real word separator (the flash models
// commonly split a sentence that way) and has to survive, or the word carried
// over from the chunk before never gets flushed.
func stripLeadingSpaces(a *httpAlignment, shouldStrip bool) *httpAlignment {
	if !shouldStrip || len(a.Characters) == 0 || a.Characters[0] != " " {
		return a
	}
	n := 0
	for n < len(a.Characters) && a.Characters[n] == " " {
		n++
	}
	return &httpAlignment{
		Characters:      a.Characters[n:],
		StartTimesSecs:  cutFrom(a.StartTimesSecs, n),
		EndTimesSeconds: cutFrom(a.EndTimesSeconds, n),
	}
}

// cutFrom drops the first n entries, tolerating an array the server sent shorter
// than the characters it goes with.
func cutFrom(v []float64, n int) []float64 {
	if n >= len(v) {
		return nil
	}
	return v[n:]
}

// utterance folds the chunks of one request together: the audio goes downstream
// as it arrives, and the character timings become whole words.
type utterance struct {
	syn  *synthesizer
	emit func(pcm []byte) error
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error

	// started reports whether an alignment has been seen yet, which is what makes
	// the next one an utterance-leading chunk or not.
	started bool
	// duration is the furthest into this request's audio any chunk reached.
	duration float64
}

// line folds one streamed message in.
func (u *utterance) line(raw []byte) error {
	var m httpMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		// A line that does not parse is skipped rather than failing the turn: the
		// audio around it is still good.
		slog.Warn("skipping an unparseable line of the synthesis stream",
			"service", "ElevenLabsTTS", "error", err)
		return nil
	}
	if m.AudioBase64 != "" {
		pcm, err := base64.StdEncoding.DecodeString(m.AudioBase64)
		if err != nil {
			return err
		}
		if err := u.emit(pcm); err != nil {
			return err
		}
	}
	alignment := selectAlignment(&m, len(u.syn.cfg.PronunciationDictionaryLocators) > 0)
	if alignment == nil {
		return nil
	}
	return u.align(stripLeadingSpaces(alignment, !u.started))
}

// align turns one chunk's character timings into whole words on the turn's
// timeline.
func (u *utterance) align(a *httpAlignment) error {
	u.started = true
	if end := a.EndTimesSeconds; len(end) > 0 {
		u.duration = max(u.duration, end[len(end)-1])
	}
	if len(a.Characters) != len(a.StartTimesSecs) {
		// The two arrays describe the same characters, so a mismatch means the
		// chunk cannot be timed. The audio still plays.
		slog.Warn("skipping a chunk whose character timings do not line up",
			"service", "ElevenLabsTTS", "characters", len(a.Characters), "times", len(a.StartTimesSecs))
		return nil
	}
	u.syn.mu.Lock()
	starts := make([]float64, len(a.StartTimesSecs))
	for i, t := range a.StartTimesSecs {
		starts[i] = u.syn.cumulative + t
	}
	// The audio tags expressive conversation sends mark when its sounds are
	// performed, each reported as one word.
	u.syn.acc.AudioTags = u.syn.expressive != nil && u.syn.expressive()
	words, err := u.syn.acc.Add(a.Characters, starts)
	u.syn.mu.Unlock()
	if err != nil {
		return err
	}
	return u.report(words)
}

// report hands a batch of finished words to the caller listening for timing.
func (u *utterance) report(words []uctx.WordTiming) error {
	if u.word == nil || len(words) == 0 {
		return nil
	}
	return u.word(words, tts.WordTimingOptions{IncludesInterFrameSpaces: u.syn.spacelessLanguage()})
}

// finish closes the utterance out: the word still being assembled ends here
// rather than on a space, the turn's timeline advances past this request's
// audio, and what was spoken becomes context for the next sentence.
func (u *utterance) finish(text string) error {
	u.syn.mu.Lock()
	last, ok := u.syn.acc.Flush()
	u.syn.cumulative += u.duration
	if u.syn.previousText == "" {
		u.syn.previousText = text
	} else {
		u.syn.previousText += " " + text
	}
	u.syn.mu.Unlock()
	if !ok {
		return nil
	}
	return u.report([]uctx.WordTiming{last})
}

// spacelessLanguage reports whether the configured language is written without
// spaces between words, in which case the reported tokens already read as
// continuous text and a consumer joining them must add no spacing of its own.
func (s *synthesizer) spacelessLanguage() bool {
	switch s.cfg.Language.BaseCode() {
	case "zh", "ja":
		return true
	default:
		return false
	}
}

// outputFormat maps a sample rate to ElevenLabs' PCM output_format string.
// Unsupported rates fall back to pcm_24000.
func outputFormat(sampleRate int) string {
	switch sampleRate {
	case 8000, 16000, 22050, 24000, 32000, 44100, 48000:
		return fmt.Sprintf("pcm_%d", sampleRate)
	default:
		slog.Warn("elevenlabs: no PCM output format for sample rate; using 24000", "rate", sampleRate)
		return "pcm_24000"
	}
}

// elevenlabsTTSLanguages are the languages verified against synthesis, each
// under the code ElevenLabs takes for it.
//
//nolint:gochecknoglobals // read-only lookup table
var elevenlabsTTSLanguages = map[language.Language]string{
	language.Afrikaans:           "af",
	language.Arabic:              "ar",
	language.Assamese:            "as",
	language.Asturian:            "ast",
	language.Azerbaijani:         "az",
	language.Belarusian:          "be",
	language.Bulgarian:           "bg",
	language.Bengali:             "bn",
	language.Bosnian:             "bs",
	language.Catalan:             "ca",
	language.Cebuano:             "ceb",
	language.Czech:               "cs",
	language.Welsh:               "cy",
	language.Danish:              "da",
	language.German:              "de",
	language.Greek:               "el",
	language.English:             "en",
	language.Spanish:             "es",
	language.Estonian:            "et",
	language.Persian:             "fa",
	language.Finnish:             "fi",
	language.Filipino:            "fil",
	language.French:              "fr",
	language.Irish:               "ga",
	language.Galician:            "gl",
	language.Gujarati:            "gu",
	language.Hausa:               "ha",
	language.Hebrew:              "he",
	language.Hindi:               "hi",
	language.Croatian:            "hr",
	language.Hungarian:           "hu",
	language.Armenian:            "hy",
	language.Indonesian:          "id",
	language.Icelandic:           "is",
	language.Italian:             "it",
	language.Japanese:            "ja",
	language.Javanese:            "jv",
	language.Georgian:            "ka",
	language.Kazakh:              "kk",
	language.Kannada:             "kn",
	language.Korean:              "ko",
	language.Kyrgyz:              "ky",
	language.Luxembourgish:       "lb",
	language.Lingala:             "ln",
	language.Lithuanian:          "lt",
	language.Latvian:             "lv",
	language.Maori:               "mi",
	language.Macedonian:          "mk",
	language.Malayalam:           "ml",
	language.Mongolian:           "mn",
	language.Marathi:             "mr",
	language.Malay:               "ms",
	language.Maltese:             "mt",
	language.Burmese:             "my",
	language.Nepali:              "ne",
	language.Dutch:               "nl",
	language.Norwegian:           "no",
	language.Chichewa:            "ny",
	language.Occitan:             "oc",
	language.Odia:                "or",
	language.Punjabi:             "pa",
	language.Polish:              "pl",
	language.Pashto:              "ps",
	language.Portuguese:          "pt",
	language.Romanian:            "ro",
	language.Russian:             "ru",
	language.Sindhi:              "sd",
	language.Slovak:              "sk",
	language.Slovenian:           "sl",
	language.Somali:              "so",
	language.Serbian:             "sr",
	language.Swedish:             "sv",
	language.Swahili:             "sw",
	language.Tamil:               "ta",
	language.Telugu:              "te",
	language.Tajik:               "tg",
	language.Thai:                "th",
	language.Turkish:             "tr",
	language.Ukrainian:           "uk",
	language.Urdu:                "ur",
	language.Uzbek:               "uz",
	language.Vietnamese:          "vi",
	language.YueChineseCantonese: "yue",
	language.Chinese:             "zh",
}

// elevenlabsLanguage maps a Language to ElevenLabs' language code: ElevenLabs
// wants the base code, so a language not verified is sent as its base code and
// reported. The zero value maps to "", which leaves the language to the model.
// Whether the model takes the code is decided by elevenlabsLanguageCode.
func elevenlabsLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, elevenlabsTTSLanguages, true)
}

// The HTTP service always asks for timestamps, so it always reports word timing,
// and it carries state across a turn that the base's lifecycle callbacks reset.
var (
	_ tts.Synthesizer             = (*synthesizer)(nil)
	_ tts.WordTimestamps          = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
	_ tts.TurnContextCompleter    = (*synthesizer)(nil)
)
