package groq

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	defaultTTSModel = "canopylabs/orpheus-v1-english"
	defaultTTSVoice = "autumn"
	// ttsSampleRate is Groq TTS's fixed output rate; the API streams 48 kHz audio.
	ttsSampleRate = 48000
	// ttsReadChunk is the size of each PCM read from the streamed response.
	ttsReadChunk = 4096
)

// errFormat is returned when the TTS response is not the expected WAV stream.
//
//nolint:gochecknoglobals // sentinel error
var errFormat = errors.New("groq: unexpected audio format")

// TTSConfig configures the Groq TTS service. It targets Groq's OpenAI-compatible
// /audio/speech endpoint, which returns a WAV stream at 48 kHz.
type TTSConfig struct {
	// APIKey is the Groq API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the API base.
	BaseURL string
	// Model is the TTS model; empty uses the default.
	Model string
	// Voice is the voice name; empty uses a default voice.
	Voice string
	// Speed multiplies the speaking rate; nil omits it (server default 1.0).
	Speed *float64
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Groq TTS service. Groq streams 48 kHz WAV; the service strips
// the container and emits raw 16-bit mono PCM downstream.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = baseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultTTSVoice
	}
	return tts.New("GroqTTS", &ttsSynthesizer{cfg: cfg, http: &http.Client{}})
}

type ttsSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports Groq's fixed PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return ttsSampleRate }

// Synthesize requests speech for text and streams the decoded PCM downstream.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	payload := map[string]any{
		"model":           s.cfg.Model,
		"voice":           s.cfg.Voice,
		"input":           text,
		"response_format": "wav",
	}
	if s.cfg.Speed != nil {
		payload["speed"] = *s.cfg.Speed
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	return streamWAV(resp.Body, emit)
}

// streamWAV skips the RIFF/WAVE header, then streams the PCM data payload to emit
// in chunks. Groq wraps 16-bit mono PCM in a WAV container; the base pipeline
// consumes raw PCM.
func streamWAV(body io.Reader, emit func(pcm []byte) error) error {
	var riff [12]byte
	if _, err := io.ReadFull(body, riff[:]); err != nil {
		return err
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return errFormat
	}
	// Walk the chunk list until the data chunk; its payload is the PCM.
	var head [8]byte
	for {
		if _, err := io.ReadFull(body, head[:]); err != nil {
			return err
		}
		if string(head[0:4]) == "data" {
			break
		}
		size := int64(binary.LittleEndian.Uint32(head[4:8]))
		if size%2 == 1 {
			size++ // chunks are word-aligned
		}
		if _, err := io.CopyN(io.Discard, body, size); err != nil {
			return err
		}
	}
	buf := make([]byte, ttsReadChunk)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if perr := emit(chunk); perr != nil {
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
