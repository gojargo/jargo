package minimax

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/gojargo/jargo/service/tts"
)

// NewTTS builds a MiniMax TTS service.
func NewTTS(cfg Config) *tts.Base {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = defaultVoice
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	return tts.New("MiniMaxTTS", &synthesizer{cfg: cfg, http: &http.Client{}})
}

type synthesizer struct {
	cfg  Config
	http *http.Client
}

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize requests speech for text and streams the hex-decoded PCM
// downstream.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	body, err := json.Marshal(s.request(text))
	if err != nil {
		return err
	}
	endpoint := s.cfg.BaseURL + "/v1/t2a_v2?GroupId=" + url.QueryEscape(s.cfg.GroupID)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
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
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	return scanSSE(resp.Body, func(data []byte) error {
		var chunk t2aChunk
		_ = json.Unmarshal(data, &chunk) // a non-audio event leaves Audio empty
		if chunk.Data.Audio == "" || chunk.Data.Status == statusFinal {
			return nil
		}
		pcm, err := hex.DecodeString(chunk.Data.Audio)
		if err != nil {
			return err
		}
		return emit(pcm)
	})
}

func (s *synthesizer) request(text string) map[string]any {
	voice := map[string]any{"voice_id": s.cfg.VoiceID}
	if s.cfg.Speed != nil {
		voice["speed"] = *s.cfg.Speed
	}
	if s.cfg.Volume != nil {
		voice["vol"] = *s.cfg.Volume
	}
	if s.cfg.Pitch != nil {
		voice["pitch"] = *s.cfg.Pitch
	}
	return map[string]any{
		"model":         s.cfg.Model,
		"text":          text,
		"stream":        true,
		"voice_setting": voice,
		"audio_setting": map[string]any{
			"sample_rate": s.cfg.SampleRate,
			"format":      "pcm",
			"channel":     1,
		},
	}
}

// t2aChunk is the subset of a MiniMax SSE event we use.
type t2aChunk struct {
	Data struct {
		Audio  string `json:"audio"`
		Status int    `json:"status"`
	} `json:"data"`
}

// scanSSE reads a Server-Sent Events stream, invoking fn for each non-empty
// "data:" payload.
func scanSSE(r io.Reader, fn func(data []byte) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 4096), sseMaxLine)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "" || data == "[DONE]" {
			continue
		}
		if err := fn([]byte(data)); err != nil {
			return err
		}
	}
	return sc.Err()
}
