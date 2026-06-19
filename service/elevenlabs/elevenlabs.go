// Package elevenlabs is a streaming text-to-speech service backed by the
// ElevenLabs HTTP streaming API. The shared TTS base aggregates incoming text
// into sentences; this service synthesizes each and streams the PCM (24 kHz)
// downstream.
package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/gojargo/jargo/service/tts"
)

const (
	apiBase = "https://api.elevenlabs.io/v1/text-to-speech"
	// sampleRate is the PCM rate jargo requests from ElevenLabs. 24 kHz is
	// available on all tiers; the output transport resamples it to 48 kHz.
	sampleRate = 24000
	// defaultVoiceID is a public ElevenLabs voice ("Rachel").
	defaultVoiceID = "21m00Tcm4TlvDq8ikWAM"
	// defaultModel is the lowest-latency multilingual model.
	defaultModel = "eleven_flash_v2_5"
)

// Config configures the TTS service.
type Config struct {
	// APIKey is the ElevenLabs API key; empty uses the ELEVENLABS_API_KEY env var.
	APIKey string
	// VoiceID is the ElevenLabs voice; empty uses a default public voice.
	VoiceID string
	// Model is the ElevenLabs model; empty uses the low-latency flash model.
	Model string
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
	return tts.New("ElevenLabsTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return sampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	body, err := json.Marshal(map[string]any{
		"text":     text,
		"model_id": s.cfg.Model,
	})
	if err != nil {
		return err
	}
	endpoint := fmt.Sprintf("%s/%s/stream?output_format=pcm_%d", apiBase, s.cfg.VoiceID, sampleRate)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("xi-api-key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "audio/pcm")
	return tts.StreamResponse(s.http, req, emit)
}
