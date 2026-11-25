package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/gojargo/jargo/errclass"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/provider/google/internal/gcloud"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"golang.org/x/oauth2"
)

const (
	ttsHost          = "texttospeech.googleapis.com"
	ttsEndpoint      = "https://" + ttsHost + "/v1/text:synthesize"
	defaultVoiceName = "en-US-Chirp3-HD-Charon"
	defaultTTSRate   = 24000
	// wavHeaderBytes is the canonical PCM WAV header the synthesizer strips to
	// recover raw samples from a LINEAR16 response.
	wavHeaderBytes = 44
	// minSpeakingRate and maxSpeakingRate bound the speaking rate Google
	// accepts.
	minSpeakingRate = 0.25
	maxSpeakingRate = 2.0
)

// errNoTTSAuth is returned when the configuration gives neither an API key nor
// credentials.
//
//nolint:gochecknoglobals // sentinel error
var errNoTTSAuth = errors.New("google tts: an API key or credentials are required")

// Credentials says how to authorize against Google Cloud: a service-account
// key as JSON, the path of a key file, or a token source. It is the same type
// the speech package takes, so one value serves both.
type Credentials = gcloud.Credentials

// TTSConfig configures the Google Cloud Text-to-Speech service, which
// synthesizes each sentence in one request. It emits 16-bit mono PCM at the
// configured rate.
//
// It authorizes with an API key or with Google Cloud credentials; one of the
// two is required, and the credentials are used when both are given.
type TTSConfig struct {
	// APIKey is the Google API key.
	APIKey string
	// Credentials authorize the requests with an OAuth access token, in place
	// of the API key.
	Credentials Credentials
	// Location is the Google Cloud location of a regional endpoint (e.g.
	// "us-central1"); empty uses the global endpoint.
	Location string
	// VoiceName is the voice id (e.g. "en-US-Chirp3-HD-Charon"); empty uses a
	// default.
	VoiceName string
	// Language selects the spoken language, mapped to a Google language code;
	// the zero value uses US English.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int

	// Pitch adjusts the voice pitch (e.g. "+2st", "-50%"). Chirp and Journey
	// voices take plain text and ignore it, as they do the other SSML controls
	// below.
	Pitch string
	// Rate adjusts the speaking rate through SSML prosody (e.g. "slow",
	// "125%"), for voices other than Chirp and Journey.
	Rate string
	// SpeakingRate is the speaking rate of Chirp and Journey voices, in the
	// range [0.25, 2.0]; nil leaves it to Google.
	SpeakingRate *float64
	// Volume adjusts the volume (e.g. "loud", "+6dB").
	Volume string
	// Emphasis is the emphasis level: "strong", "moderate", "reduced" or
	// "none".
	Emphasis string `validate:"omitempty,oneof=strong moderate reduced none"`
	// Gender is the voice gender preference: "male", "female" or "neutral".
	Gender string `validate:"omitempty,oneof=male female neutral"`
	// GoogleStyle is a Google voice style: "apologetic", "calm", "empathetic",
	// "firm" or "lively".
	GoogleStyle string `validate:"omitempty,oneof=apologetic calm empathetic firm lively"`
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error {
	if err := validate.Struct(c); err != nil {
		return err
	}
	if c.APIKey == "" && !c.Credentials.Given() {
		return errNoTTSAuth
	}
	return nil
}

// TTSSettings is the part of the synthesis configuration that can change while
// the pipeline runs. Each sentence is a request of its own, so a change takes
// effect on the next one.
type TTSSettings struct {
	settings.TTS

	// Pitch adjusts the voice pitch (e.g. "+2st", "-50%").
	Pitch settings.Opt[string] `settings:"pitch"`
	// Rate adjusts the speaking rate through SSML prosody (e.g. "slow",
	// "125%"), for voices other than Chirp and Journey.
	Rate settings.Opt[string] `settings:"rate"`
	// SpeakingRate is the speaking rate of Chirp and Journey voices, in the
	// range [0.25, 2.0].
	SpeakingRate settings.Opt[float64] `settings:"speaking_rate"`
	// Volume adjusts the volume (e.g. "loud", "soft", "+6dB").
	Volume settings.Opt[string] `settings:"volume"`
	// Emphasis is the emphasis level of the text.
	Emphasis settings.Opt[string] `settings:"emphasis"`
	// Gender is the voice gender preference.
	Gender settings.Opt[string] `settings:"gender"`
	// GoogleStyle is a Google voice style.
	GoogleStyle settings.Opt[string] `settings:"google_style"`
}

// ApplyUpdate merges an update, implementing settings.Updater. A speaking
// rate outside the range Google accepts is refused with a warning: it stays as
// it was, while the rest of the update still applies.
func (s *TTSSettings) ApplyUpdate(delta any) (settings.Changed, error) {
	kept := s.SpeakingRate
	changed, err := settings.ApplyFields(s, delta)
	if err != nil || !changed.Has("speaking_rate") {
		return changed, err
	}
	rate, ok := s.SpeakingRate.Value()
	if !ok || (rate >= minSpeakingRate && rate <= maxSpeakingRate) {
		// No value leaves the rate to Google, which needs no check.
		return changed, nil
	}
	slog.Warn("Invalid speaking_rate value, must be between 0.25 and 2.0", "speaking_rate", rate)
	s.SpeakingRate = kept
	delete(changed, "speaking_rate")
	return changed, nil
}

// newTTSSettings is the starting state, taken from what the service was built
// with.
func newTTSSettings(cfg TTSConfig) *TTSSettings {
	s := &TTSSettings{}
	s.Model = settings.Cleared[string]()
	s.Voice = settings.Set(cfg.VoiceName)
	s.Language = settings.Set(languageToGoogleTTS(cfg.Language))
	s.Pitch = stringOpt(cfg.Pitch)
	s.Rate = stringOpt(cfg.Rate)
	s.SpeakingRate = settings.Cleared[float64]()
	if cfg.SpeakingRate != nil {
		s.SpeakingRate = settings.Set(*cfg.SpeakingRate)
	}
	s.Volume = stringOpt(cfg.Volume)
	s.Emphasis = stringOpt(cfg.Emphasis)
	s.Gender = stringOpt(cfg.Gender)
	s.GoogleStyle = stringOpt(cfg.GoogleStyle)
	return s
}

// stringOpt is a setting holding v, or no value when v is empty.
func stringOpt(v string) settings.Opt[string] {
	if v == "" {
		return settings.Cleared[string]()
	}
	return settings.Set(v)
}

// NewTTS builds a Google Cloud Text-to-Speech service.
func NewTTS(cfg TTSConfig) *tts.Base {
	return tts.New("GoogleTTS", newTTSSynthesizer(cfg, &http.Client{}))
}

// newTTSSynthesizer builds the synthesizer behind NewTTS, sending its requests
// through client.
func newTTSSynthesizer(cfg TTSConfig, client *http.Client) *ttsSynthesizer {
	if cfg.VoiceName == "" {
		cfg.VoiceName = defaultVoiceName
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSRate
	}
	return &ttsSynthesizer{cfg: cfg, live: newTTSSettings(cfg), http: client}
}

type ttsSynthesizer struct {
	cfg  TTSConfig
	live *TTSSettings
	http *http.Client

	// mu guards source, which is resolved from the credentials on first use:
	// the service is built without a context to exchange a key in.
	mu     sync.Mutex
	source oauth2.TokenSource
}

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder.
func (s *ttsSynthesizer) Settings() any { return s.live }

// ServiceLanguage names a language the way Cloud Text-to-Speech does,
// implementing tts.LanguageNamer.
func (s *ttsSynthesizer) ServiceLanguage(l language.Language) string {
	return gcloud.TTSLanguage(l)
}

// endpoint is the synthesize URL: the regional endpoint when a location is
// configured, the global one otherwise.
func (s *ttsSynthesizer) endpoint() string {
	if s.cfg.Location == "" {
		return ttsEndpoint
	}
	return "https://" + s.cfg.Location + "-" + ttsHost + "/v1/text:synthesize"
}

// tokenSource resolves the credentials into a token source on first use. The
// token source outlives the request that resolved it, so it is not tied to
// that request's context.
func (s *ttsSynthesizer) tokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.source != nil {
		return s.source, nil
	}
	resolved, err := s.cfg.Credentials.Resolve(context.WithoutCancel(ctx))
	if err != nil {
		return nil, err
	}
	s.source = oauth2.ReuseTokenSource(nil, resolved.TokenSource)
	return s.source, nil
}

// authorize addresses req and authorizes it: with an access token when
// credentials are configured, with the API key otherwise.
func (s *ttsSynthesizer) authorize(ctx context.Context, body []byte) (*http.Request, error) {
	url := s.endpoint()
	if !s.cfg.Credentials.Given() {
		url += "?key=" + s.cfg.APIKey
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if s.cfg.Credentials.Given() {
		source, err := s.tokenSource(ctx)
		if err != nil {
			return nil, err
		}
		tok, err := source.Token()
		if err != nil {
			return nil, fmt.Errorf("google tts: access token: %w", err)
		}
		tok.SetAuthHeader(req)
	}
	return req, nil
}

// plainTextVoice reports whether the voice takes plain text rather than SSML,
// which Chirp (Chirp 3 included) and Journey voices do.
func plainTextVoice(voice string) bool {
	v := strings.ToLower(voice)
	return strings.Contains(v, "chirp") || strings.Contains(v, "journey")
}

// constructSSML wraps text in the SSML that carries the voice controls.
func (s *ttsSynthesizer) constructSSML(text string) string {
	var ssml strings.Builder
	ssml.WriteString("<speak>")

	// Voice tag.
	voice, _ := s.live.Voice.Value()
	languageCode, _ := s.live.Language.Value()
	voiceAttrs := []string{"name='" + voice + "'", "language='" + languageCode + "'"}
	if gender, ok := s.live.Gender.Value(); ok && gender != "" {
		voiceAttrs = append(voiceAttrs, "gender='"+gender+"'")
	}
	ssml.WriteString("<voice " + strings.Join(voiceAttrs, " ") + ">")

	// Prosody tag.
	var prosodyAttrs []string
	if pitch, ok := s.live.Pitch.Value(); ok && pitch != "" {
		prosodyAttrs = append(prosodyAttrs, "pitch='"+pitch+"'")
	}
	if rate, ok := s.live.Rate.Value(); ok && rate != "" {
		prosodyAttrs = append(prosodyAttrs, "rate='"+rate+"'")
	}
	if volume, ok := s.live.Volume.Value(); ok && volume != "" {
		prosodyAttrs = append(prosodyAttrs, "volume='"+volume+"'")
	}
	if len(prosodyAttrs) > 0 {
		ssml.WriteString("<prosody " + strings.Join(prosodyAttrs, " ") + ">")
	}

	// Emphasis tag.
	emphasis, _ := s.live.Emphasis.Value()
	if emphasis != "" {
		ssml.WriteString("<emphasis level='" + emphasis + "'>")
	}

	// Google style tag.
	style, _ := s.live.GoogleStyle.Value()
	if style != "" {
		ssml.WriteString("<google:style name='" + style + "'>")
	}

	ssml.WriteString(escapeSSML(text))

	// Close tags.
	if style != "" {
		ssml.WriteString("</google:style>")
	}
	if emphasis != "" {
		ssml.WriteString("</emphasis>")
	}
	if len(prosodyAttrs) > 0 {
		ssml.WriteString("</prosody>")
	}
	ssml.WriteString("</voice></speak>")
	return ssml.String()
}

// escapeSSML escapes the characters that would otherwise be read as markup:
// ampersands and angle brackets.
func escapeSSML(text string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;").Replace(text)
}

// requestBody builds the synthesize request body for text. Chirp and Journey
// voices are sent plain text, with the speaking rate in the audio config;
// other voices are sent the SSML carrying the voice controls.
func (s *ttsSynthesizer) requestBody(text string) ([]byte, error) {
	voice, _ := s.live.Voice.Value()
	languageCode, _ := s.live.Language.Value()
	plain := plainTextVoice(voice)

	input := map[string]any{"ssml": s.constructSSML(text)}
	if plain {
		input = map[string]any{"text": text}
	}
	audioConfig := map[string]any{
		"audioEncoding":   "LINEAR16",
		"sampleRateHertz": s.cfg.SampleRate,
	}
	if rate, ok := s.live.SpeakingRate.Value(); ok && plain {
		audioConfig["speakingRate"] = rate
	}
	return json.Marshal(map[string]any{
		"input": input,
		"voice": map[string]any{
			"languageCode": languageCode,
			"name":         voice,
		},
		"audioConfig": audioConfig,
	})
}

// RunTTS requests speech for text and passes the decoded raw PCM on in frames
// of tts.ChunkSize.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	body, err := s.requestBody(text)
	if err != nil {
		return err
	}
	req, err := s.authorize(ctx, body)
	if err != nil {
		return err
	}
	pcm, err := s.fetchPCM(req)
	if err != nil {
		return err
	}
	tts.StartUsageMetrics(ctx, text)

	chunkSize := tts.ChunkSize(s.SampleRate())
	for i := 0; i < len(pcm); i += chunkSize {
		if err := emit(pcm[i:min(i+chunkSize, len(pcm))]); err != nil {
			return err
		}
	}
	return nil
}

// fetchPCM issues req and returns the raw PCM decoded from the base64 WAV
// response with its 44-byte header stripped.
func (s *ttsSynthesizer) fetchPCM(req *http.Request) ([]byte, error) {
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, errclass.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	var out struct {
		AudioContent string `json:"audioContent"` //nolint:tagliatelle // Google REST uses camelCase keys
	}
	if err = json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	wav, err := base64.StdEncoding.DecodeString(out.AudioContent)
	if err != nil {
		return nil, err
	}
	if len(wav) <= wavHeaderBytes {
		return nil, nil
	}
	return wav[wavHeaderBytes:], nil
}

// languageToGoogleTTS maps a language to a Google TTS language code, US English
// when none is given.
func languageToGoogleTTS(l language.Language) string {
	if l == "" {
		return defaultLangCode
	}
	return gcloud.TTSLanguage(l)
}
