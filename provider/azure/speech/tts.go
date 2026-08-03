package speech

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
)

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
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
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
	body := esc.String()
	if s.cfg.ForceLocale {
		// A multilingual voice reads the language out of the text unless it is
		// told; the <lang> element is what tells it.
		body = fmt.Sprintf("<lang xml:lang='%s'>%s</lang>", lang, body)
	}
	doc := fmt.Sprintf("<speak version='1.0' xml:lang='%s'><voice name='%s'>%s</voice></speak>",
		lang, s.cfg.Voice, body)
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
