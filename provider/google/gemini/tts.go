package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

const (
	ttsEndpoint      = "https://texttospeech.googleapis.com/v1/text:synthesize"
	defaultVoiceName = "en-US-Chirp3-HD-Charon"
	defaultTTSRate   = 24000
	// wavHeaderBytes is the canonical PCM WAV header the synthesizer strips to
	// recover raw samples from a LINEAR16 response.
	wavHeaderBytes = 44
)

// TTSConfig configures the Google Cloud Text-to-Speech service. It emits 16-bit
// mono PCM at the configured rate.
type TTSConfig struct {
	// APIKey is the Google API key. Required.
	APIKey string `validate:"required"`
	// VoiceName is the voice id (e.g. "en-US-Chirp3-HD-Charon"); empty uses a
	// default.
	VoiceName string
	// Language selects the spoken language, mapped to a Google language code;
	// the zero value uses US English.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses 24 kHz.
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Google Cloud Text-to-Speech service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.VoiceName == "" {
		cfg.VoiceName = defaultVoiceName
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSRate
	}
	return tts.New("GoogleTTS", &ttsSynthesizer{cfg: cfg, http: &http.Client{}})
}

type ttsSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// requestBody builds the synthesize request body for text.
func (s *ttsSynthesizer) requestBody(text string) ([]byte, error) {
	return json.Marshal(map[string]any{
		"input": map[string]any{"text": text},
		"voice": map[string]any{
			"languageCode": languageToGoogleTTS(s.cfg.Language),
			"name":         s.cfg.VoiceName,
		},
		"audioConfig": map[string]any{
			"audioEncoding":   "LINEAR16",
			"sampleRateHertz": s.cfg.SampleRate,
		},
	})
}

// Synthesize requests speech for text and streams the decoded raw PCM downstream.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	body, err := s.requestBody(text)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ttsEndpoint+"?key="+s.cfg.APIKey,
		bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	pcm, err := s.fetchPCM(req)
	if err != nil {
		return err
	}
	return emit(pcm)
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
		return nil, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
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

// languageToGoogleTTS maps a language to a Google TTS language code, defaulting
// to the full code when the base language is not modeled.
func languageToGoogleTTS(l language.Language) string {
	switch l.BaseCode() {
	case "en":
		return defaultLangCode
	case "fr":
		return "fr-FR"
	case "es":
		return "es-ES"
	case "de":
		return "de-DE"
	case "it":
		return "it-IT"
	case "nl":
		return "nl-NL"
	case "pt":
		return "pt-BR"
	case "pl":
		return "pl-PL"
	case "ru":
		return "ru-RU"
	case "ja":
		return "ja-JP"
	case "ko":
		return "ko-KR"
	case "zh":
		return "cmn-CN"
	default:
		if code := l.Code(); code != "" {
			return code
		}
		return defaultLangCode
	}
}
