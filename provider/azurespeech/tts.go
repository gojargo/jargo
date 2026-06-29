// Package azurespeech provides Azure AI Speech services. NewTTS is a streaming
// text-to-speech service over the REST endpoint: it sends SSML and requests raw
// 16-bit mono PCM, which streams straight downstream.
package azurespeech

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

const (
	defaultTTSVoice = "en-US-JennyNeural"
	defaultTTSRate  = 24000
	// ttsUserAgent is sent on every request: Azure requires a non-empty
	// User-Agent and rejects the request otherwise.
	ttsUserAgent = "jargo"
)

// TTSConfig configures the Azure AI Speech TTS service.
type TTSConfig struct {
	// APIKey is the Speech resource key, sent as Ocp-Apim-Subscription-Key.
	// Required.
	APIKey string `validate:"required"`
	// Region is the Speech resource region, e.g. "eastus" or "francecentral".
	// Required unless Host is set.
	Region string
	// Host overrides the full TTS host (for sovereign clouds or custom domains),
	// e.g. https://my-resource.tts.speech.azure.us. Empty derives it from Region.
	Host string
	// Voice is the SSML voice name (e.g. "en-US-JennyNeural"); empty uses a
	// default.
	Voice string
	// Language sets the SSML xml:lang; the zero value derives it from the voice's
	// locale.
	Language language.Language
	// SampleRate is the PCM output rate; empty uses 24 kHz. Must be one Azure
	// offers as raw PCM (8000, 16000, 22050, 24000, 44100, 48000).
	SampleRate int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds an Azure AI Speech TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.Voice == "" {
		cfg.Voice = defaultTTSVoice
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSRate
	}
	return tts.New("AzureTTS", &ttsSynthesizer{cfg: cfg, http: &http.Client{}})
}

type ttsSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports the PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *ttsSynthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	ssml, err := s.ssml(text)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint(), bytes.NewReader(ssml))
	if err != nil {
		return err
	}
	req.Header.Set("Ocp-Apim-Subscription-Key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/ssml+xml")
	req.Header.Set("X-Microsoft-OutputFormat", outputFormat(s.cfg.SampleRate))
	req.Header.Set("User-Agent", ttsUserAgent)
	return tts.StreamResponse(s.http, req, emit)
}

func (s *ttsSynthesizer) endpoint() string {
	host := s.cfg.Host
	if host == "" {
		host = fmt.Sprintf("https://%s.tts.speech.microsoft.com", s.cfg.Region)
	}
	return strings.TrimSuffix(host, "/") + "/cognitiveservices/v1"
}

// ssml wraps text in a minimal SSML document with the configured voice and
// language, escaping the text so it cannot break the markup.
func (s *ttsSynthesizer) ssml(text string) ([]byte, error) {
	lang := s.cfg.Language.Code()
	if lang == "" {
		lang = localeFromVoice(s.cfg.Voice)
	}
	var esc bytes.Buffer
	if err := xml.EscapeText(&esc, []byte(text)); err != nil {
		return nil, err
	}
	doc := fmt.Sprintf("<speak version='1.0' xml:lang='%s'><voice name='%s'>%s</voice></speak>",
		lang, s.cfg.Voice, esc.String())
	return []byte(doc), nil
}

// localeFromVoice extracts the BCP-47 locale prefix from an Azure voice name
// ("fr-FR-DeniseNeural" -> "fr-FR"), defaulting to en-US.
func localeFromVoice(voice string) string {
	parts := strings.SplitN(voice, "-", 3)
	if len(parts) >= 2 {
		return parts[0] + "-" + parts[1]
	}
	return "en-US"
}

// outputFormat maps a sample rate to Azure's raw-PCM X-Microsoft-OutputFormat.
// Unsupported rates fall back to 24 kHz.
func outputFormat(rate int) string {
	switch rate {
	case 8000:
		return "raw-8khz-16bit-mono-pcm"
	case 16000:
		return "raw-16khz-16bit-mono-pcm"
	case 22050:
		return "raw-22050hz-16bit-mono-pcm"
	case 24000:
		return "raw-24khz-16bit-mono-pcm"
	case 44100:
		return "raw-44100hz-16bit-mono-pcm"
	case 48000:
		return "raw-48khz-16bit-mono-pcm"
	default:
		slog.Warn("azurespeech: unsupported TTS sample rate; using 24000", "rate", rate)
		return "raw-24khz-16bit-mono-pcm"
	}
}
