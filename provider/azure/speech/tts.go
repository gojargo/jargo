package speech

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/expressive"
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

// ExpressionAdapter implements tts.ExpressionAdapterProvider: style markers on
// DragonHD and DragonHD Omni voices, and none on the others. DragonHD Flash
// voices each have styles of their own, so they perform none.
func (s *ttsSynthesizer) ExpressionAdapter() expressive.ExpressionAdapter {
	return voiceExpressionAdapter(s.cfg.Voice)
}

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	ssml := s.ssml(text)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint(), bytes.NewReader(ssml))
	if err != nil {
		return err
	}
	req.Header.Set("Ocp-Apim-Subscription-Key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/ssml+xml")
	req.Header.Set("X-Microsoft-OutputFormat", outputFormat(s.cfg.SampleRate))
	req.Header.Set("User-Agent", ttsUserAgent)
	return tts.StreamSpeech(s.http, req, text, emit)
}

func (s *ttsSynthesizer) endpoint() string {
	host := s.cfg.Host
	if host == "" {
		host = fmt.Sprintf("https://%s.tts.speech.microsoft.com", s.cfg.Region)
	}
	return strings.TrimSuffix(host, "/") + "/cognitiveservices/v1"
}

// ssml wraps text in the SSML document it is synthesized from, escaping the
// text so it cannot break the markup.
func (s *ttsSynthesizer) ssml(text string) []byte {
	lang := defaultTTSLanguage
	if s.cfg.Language != "" {
		lang = azureLanguage(s.cfg.Language)
	}
	return []byte(constructSSML(text, ssmlParams{
		Voice:           s.cfg.Voice,
		Language:        lang,
		Effect:          s.cfg.Effect,
		VoiceParameters: s.cfg.VoiceParameters,
		ForceLocale:     s.cfg.ForceLocale,
		Style:           s.cfg.Style,
		StyleDegree:     s.cfg.StyleDegree,
		Role:            s.cfg.Role,
		Rate:            s.cfg.Rate,
		Pitch:           s.cfg.Pitch,
		Volume:          s.cfg.Volume,
		Emphasis:        s.cfg.Emphasis,
	}))
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
