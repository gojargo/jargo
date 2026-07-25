package xtts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gojargo/jargo/service/tts"
)

// NewTTS builds a Coqui XTTS TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.Language == "" {
		cfg.Language = defaultLanguage
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("XTTSTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client

	mu       sync.Mutex
	speakers map[string]speaker
}

// speaker holds one studio speaker's conditioning tensors, passed back verbatim
// in each synthesis request.
type speaker struct {
	SpeakerEmbedding json.RawMessage `json:"speaker_embedding"`
	GPTCondLatent    json.RawMessage `json:"gpt_cond_latent"`
}

// SampleRate reports XTTS's PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams raw PCM downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	spk, err := s.lookupSpeaker(ctx)
	if err != nil {
		return err
	}
	payload := map[string]any{
		"text":              cleanText(text),
		"language":          s.cfg.Language,
		"speaker_embedding": spk.SpeakerEmbedding,
		"gpt_cond_latent":   spk.GPTCondLatent,
		"add_wav_header":    false,
		"stream_chunk_size": 20,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/tts_stream", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}

// lookupSpeaker returns the configured studio speaker's embeddings, fetching and
// caching the server's speaker list on first use.
func (s *synthesizer) lookupSpeaker(ctx context.Context) (speaker, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.speakers == nil {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.BaseURL+"/studio_speakers", nil)
		if err != nil {
			return speaker{}, err
		}
		resp, err := s.http.Do(req)
		if err != nil {
			return speaker{}, err
		}
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			return speaker{}, fmt.Errorf("%w %d", errStatus, resp.StatusCode)
		}
		if err := json.NewDecoder(resp.Body).Decode(&s.speakers); err != nil {
			return speaker{}, err
		}
	}
	spk, ok := s.speakers[s.cfg.Voice]
	if !ok {
		return speaker{}, fmt.Errorf("%w %q", errUnknownVoice, s.cfg.Voice)
	}
	return spk, nil
}

// cleanText drops characters the XTTS server mishandles.
func cleanText(text string) string {
	return strings.NewReplacer(".", "", "*", "").Replace(text)
}
