// Package elevenlabs is a streaming text-to-speech service backed by the
// ElevenLabs HTTP streaming API. It aggregates incoming text into sentences,
// synthesizes each, and pushes the audio downstream as TTSAudioRawFrames
// (24 kHz) bracketed by TTSStarted/TTSStopped frames.
package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errUnexpectedStatus is returned when ElevenLabs responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errUnexpectedStatus = errors.New("elevenlabs: unexpected status")

const (
	apiBase = "https://api.elevenlabs.io/v1/text-to-speech"
	// sampleRate is the PCM rate jargo requests from ElevenLabs. 24 kHz is
	// available on all tiers; the output transport resamples it to 48 kHz.
	sampleRate = 24000
	// defaultVoiceID is a public ElevenLabs voice ("Rachel").
	defaultVoiceID = "21m00Tcm4TlvDq8ikWAM"
	// defaultModel is the lowest-latency multilingual model.
	defaultModel = "eleven_flash_v2_5"
	// readChunk is the size of each audio read from the HTTP stream.
	readChunk = 4096
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

// Service is an ElevenLabs streaming TTS processor.
type Service struct {
	*processor.Base
	cfg         Config
	http        *http.Client
	aggregation string
}

// New builds an ElevenLabs TTS service.
func New(cfg Config) *Service {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("ELEVENLABS_API_KEY")
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoiceID
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	s := &Service{cfg: cfg, http: &http.Client{}}
	s.Base = processor.New("ElevenLabsTTS", s)
	return s
}

// ProcessFrame aggregates text into sentences and synthesizes them.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return s.aggregate(ctx, fr.Text)
	case *frames.TextFrame:
		if err := s.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return s.aggregate(ctx, fr.Text)
	case *frames.LLMFullResponseEndFrame:
		if err := s.flush(ctx); err != nil {
			return err
		}
		return s.PushFrame(ctx, f, dir)
	default:
		return s.PushFrame(ctx, f, dir)
	}
}

// aggregate buffers text and synthesizes once a sentence is complete.
func (s *Service) aggregate(ctx context.Context, text string) error {
	s.aggregation += text
	if endOfSentence(s.aggregation) {
		sentence := s.aggregation
		s.aggregation = ""
		return s.synthesize(ctx, sentence)
	}
	return nil
}

// flush synthesizes any buffered text that didn't end on a sentence boundary.
func (s *Service) flush(ctx context.Context) error {
	if strings.TrimSpace(s.aggregation) == "" {
		s.aggregation = ""
		return nil
	}
	sentence := s.aggregation
	s.aggregation = ""
	return s.synthesize(ctx, sentence)
}

// synthesize requests speech for text and streams it downstream as audio.
func (s *Service) synthesize(ctx context.Context, text string) error {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if err := s.PushFrame(ctx, frames.NewTTSStartedFrame(), processor.Downstream); err != nil {
		return err
	}
	if err := s.stream(ctx, text); err != nil && ctx.Err() == nil {
		s.PushError(ctx, "elevenlabs synthesis failed", err, false)
	}
	return s.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
}

func (s *Service) stream(ctx context.Context, text string) error {
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

	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errUnexpectedStatus, resp.StatusCode, msg)
	}

	buf := make([]byte, readChunk)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if perr := s.PushFrame(ctx, frames.NewTTSAudioRawFrame(chunk, sampleRate, 1), processor.Downstream); perr != nil {
				return perr
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// endOfSentence reports whether text ends on a sentence-terminating mark.
func endOfSentence(text string) bool {
	trimmed := strings.TrimRight(text, " \t\n\"')]}")
	if trimmed == "" {
		return false
	}
	switch trimmed[len(trimmed)-1] {
	case '.', '!', '?', ':', ';':
		return true
	}
	// Catch full-width CJK terminators.
	for _, suffix := range []string{"。", "！", "？", "…"} {
		if strings.HasSuffix(trimmed, suffix) {
			return true
		}
	}
	return false
}
