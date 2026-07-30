package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

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
	return tts.New("ElevenLabsTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// Metadata reports the ElevenLabs model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	payload := map[string]any{
		keyText:    text,
		"model_id": s.cfg.Model,
	}
	if code := elevenlabsLanguage(s.cfg.Language); code != "" {
		payload["language_code"] = code
	}
	if s.cfg.VoiceSettings != nil {
		payload["voice_settings"] = s.cfg.VoiceSettings
	}
	if s.cfg.ApplyTextNormalization != "" {
		payload["apply_text_normalization"] = s.cfg.ApplyTextNormalization
	}
	if len(s.cfg.PronunciationDictionaryLocators) > 0 {
		payload["pronunciation_dictionary_locators"] = s.cfg.PronunciationDictionaryLocators
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	q := url.Values{}
	q.Set("output_format", outputFormat(s.cfg.SampleRate))
	if s.cfg.OptimizeStreamingLatency != nil {
		q.Set("optimize_streaming_latency", strconv.Itoa(*s.cfg.OptimizeStreamingLatency))
	}
	if s.cfg.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.cfg.EnableLogging))
	}
	endpoint := fmt.Sprintf("%s/v1/text-to-speech/%s/stream?%s", s.cfg.BaseURL, s.cfg.VoiceID, q.Encode())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("xi-api-key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "audio/pcm")
	return tts.StreamResponse(s.http, req, emit)
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

// elevenlabsLanguage maps a Language to ElevenLabs' language_code: ElevenLabs
// wants the base code, so the region is stripped and returned only for languages
// ElevenLabs supports; otherwise "" (the model auto-detects).
func elevenlabsLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "ar", "bg", "cs", "da", "de", "el", "en", "es", "fi", langFil,
		"fr", "hi", "hr", "hu", "id", "it", "ja", "ko", "ms", "nl",
		"no", "pl", "pt", "ro", "ru", "sk", "sv", "ta", "tr", "uk",
		"vi", "zh":
		return base
	default:
		return ""
	}
}
