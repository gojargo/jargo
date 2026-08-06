package inworld

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/google/uuid"
)

// NewTTS builds an Inworld AI TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultURL
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
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	return tts.New("InworldTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// requestBody builds the Inworld request body for text.
func (s *synthesizer) requestBody(text string) ([]byte, error) {
	audioConfig := map[string]any{
		"audioEncoding":   s.cfg.Encoding,
		"sampleRateHertz": s.cfg.SampleRate,
	}
	if s.cfg.SpeakingRate != nil {
		audioConfig["speakingRate"] = *s.cfg.SpeakingRate
	}
	m := map[string]any{
		"text":        text,
		"voiceId":     s.cfg.VoiceID,
		"modelId":     s.cfg.Model,
		"audioConfig": audioConfig,
	}
	if s.cfg.Temperature != nil {
		m["temperature"] = *s.cfg.Temperature
	}
	if s.cfg.DeliveryMode != "" {
		m["deliveryMode"] = s.cfg.DeliveryMode
	}
	if lang := inworldLanguage(s.cfg.Language); lang != "" {
		m["language"] = lang
	}
	return json.Marshal(m)
}

// Synthesize posts text and streams the decoded PCM downstream.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	body, err := s.requestBody(text)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Basic "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", uuid.NewString())
	resp, err := s.http.Do(req) //nolint:gosec // request target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	return stream(resp.Body, emit)
}

// streamMessage is the subset of an Inworld streaming line we read.
type streamMessage struct {
	Result struct {
		AudioContent string `json:"audioContent"` //nolint:tagliatelle // vendor API field
	} `json:"result"`
}

// stream reads the newline-delimited JSON body and emits each chunk's PCM.
func stream(body io.Reader, emit func(pcm []byte) error) error {
	reader := bufio.NewReader(body)
	for {
		line, err := reader.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			if perr := emitLine(line, emit); perr != nil {
				return perr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// emitLine decodes one JSON line's audio and streams it, stripping any WAV header.
func emitLine(line []byte, emit func(pcm []byte) error) error {
	var m streamMessage
	if json.Unmarshal(bytes.TrimSpace(line), &m) != nil {
		return nil //nolint:nilerr // skip blank or non-JSON lines
	}
	if m.Result.AudioContent == "" {
		return nil
	}
	pcm, err := base64.StdEncoding.DecodeString(m.Result.AudioContent)
	if err != nil {
		return err
	}
	if len(pcm) > wavHeaderSize && bytes.HasPrefix(pcm, []byte("RIFF")) {
		pcm = pcm[wavHeaderSize:]
	}
	if len(pcm) == 0 {
		return nil
	}
	return emit(pcm)
}

// inworldTags gives each language Inworld verifies the tag it is named by, which
// carries a region the language itself does not.
//
//nolint:gochecknoglobals // lookup table
var inworldTags = map[string]string{
	"ar": "ar-SA", "de": "de-DE", "en": "en-US", "es": "es-ES", "fr": "fr-FR",
	"he": "he-IL", "hi": "hi-IN", "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR",
	"nl": "nl-NL", "pl": "pl-PL", "pt": "pt-BR", "ru": "ru-RU", "zh": "zh-CN",
}

// inworldLanguage names a language the way Inworld does. The zero value leaves
// it unset, a verified language takes its tag, and anything else is passed
// through as it was given.
//
// The lookup is on the whole code rather than the base, so a caller asking for a
// region gets that region: British English stays en-GB rather than being read as
// verified English and answered in en-US.
func inworldLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	if c, ok := inworldTags[l.Code()]; ok {
		return c
	}
	return l.Code()
}
