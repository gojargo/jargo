// Package elevenlabs is a streaming text-to-speech service backed by the
// ElevenLabs HTTP streaming API. The shared TTS base aggregates incoming text
// into sentences; this service synthesizes each and streams raw PCM downstream
// at the configured rate (48 kHz by default, matching the WebRTC Opus rate so no
// resampling is needed).
//
// This wraps the HTTP /stream endpoint. It does not implement ElevenLabs'
// WebSocket multi-stream-input transport or word-level timestamp alignment.
package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

const (
	apiBase = "https://api.elevenlabs.io/v1/text-to-speech"
	// defaultSampleRate matches the WebRTC Opus rate so the output transport
	// does not resample. 48 kHz PCM is not tier-gated (only 44.1 kHz requires a
	// Pro plan).
	defaultSampleRate = 48000
	// defaultVoiceID is a public ElevenLabs voice ("Rachel").
	defaultVoiceID = "21m00Tcm4TlvDq8ikWAM"
	// defaultModel is the lowest-latency multilingual model.
	defaultModel = "eleven_flash_v2_5"
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

// Config configures the TTS service.
type Config struct {
	// APIKey is the ElevenLabs API key; empty uses the ELEVENLABS_API_KEY env var.
	APIKey string
	// VoiceID is the ElevenLabs voice; empty uses a default public voice.
	VoiceID string
	// Model is the ElevenLabs model; empty uses the low-latency flash model.
	Model string
	// SampleRate is the PCM rate requested from ElevenLabs and emitted
	// downstream. Empty uses 48 kHz. Must be a rate ElevenLabs supports
	// (8000, 16000, 22050, 24000, 32000, 44100, 48000).
	SampleRate int
	// Language for multilingual models; the zero value leaves it unset (the
	// model auto-detects). Mapped to ElevenLabs' base code per voice.
	Language language.Language
	// VoiceSettings overrides the voice's default settings when non-nil.
	VoiceSettings *VoiceSettings
}

// NewTTS builds an ElevenLabs TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("ELEVENLABS_API_KEY")
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

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	payload := map[string]any{
		"text":     text,
		"model_id": s.cfg.Model,
	}
	if code := elevenlabsLanguage(s.cfg.Language); code != "" {
		payload["language_code"] = code
	}
	if s.cfg.VoiceSettings != nil {
		payload["voice_settings"] = s.cfg.VoiceSettings
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	endpoint := fmt.Sprintf("%s/%s/stream?output_format=%s",
		apiBase, s.cfg.VoiceID, outputFormat(s.cfg.SampleRate))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("xi-api-key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "audio/pcm")
	return tts.StreamResponse(s.http, req, emit)
}

// outputFormat maps a sample rate to ElevenLabs' PCM output_format string,
// mirroring Pipecat's output_format_from_sample_rate. Unsupported rates fall
// back to pcm_24000.
func outputFormat(sampleRate int) string {
	switch sampleRate {
	case 8000, 16000, 22050, 24000, 32000, 44100, 48000:
		return fmt.Sprintf("pcm_%d", sampleRate)
	default:
		slog.Warn("elevenlabs: no PCM output format for sample rate; using 24000", "rate", sampleRate)
		return "pcm_24000"
	}
}

// elevenlabsLanguage maps a Language to ElevenLabs' language_code, mirroring
// Pipecat's language_to_elevenlabs_language: ElevenLabs wants the base code, so
// the region is stripped and returned only for languages ElevenLabs supports;
// otherwise "" (the model auto-detects).
func elevenlabsLanguage(l language.Language) string {
	switch base := l.BaseCode(); base {
	case "ar", "bg", "cs", "da", "de", "el", "en", "es", "fi", "fil",
		"fr", "hi", "hr", "hu", "id", "it", "ja", "ko", "ms", "nl",
		"no", "pl", "pt", "ro", "ru", "sk", "sv", "ta", "tr", "uk",
		"vi", "zh":
		return base
	default:
		return ""
	}
}
