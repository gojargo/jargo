// Package google is a streaming LLM service backed by Google's Gemini API
// (generateContent with SSE). It consumes an LLMContextFrame and emits the
// response as LLM response frames, like every other jargo LLM service.
package google

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("google: unexpected status")

const (
	apiBase          = "https://generativelanguage.googleapis.com/v1beta/models"
	defaultModel     = "gemini-2.5-flash"
	defaultMaxTokens = 1024
)

// Config configures the Gemini LLM service.
type Config struct {
	// APIKey is the Gemini API key; empty uses the GEMINI_API_KEY env var.
	APIKey string
	// Model is the model id; empty uses a low-latency flash default.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
}

// Service is a streaming Gemini LLM processor.
type Service struct {
	*llm.Base
	cfg  Config
	http *http.Client
}

// NewLLM builds a Gemini LLM service.
func NewLLM(cfg Config) *Service {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("GEMINI_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	s := &Service{cfg: cfg, http: &http.Client{}}
	s.Base = llm.New("GoogleLLM", s)
	return s
}

// genChunk is the subset of a streamGenerateContent SSE chunk we read.
type genChunk struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// Generate streams a Gemini completion, emitting each text delta.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	reqBody := map[string]any{
		"contents":         toContents(convo),
		"generationConfig": map[string]any{"maxOutputTokens": s.cfg.MaxTokens},
	}
	if sys := convo.System(); sys != "" {
		reqBody["systemInstruction"] = map[string]any{"parts": []map[string]any{{"text": sys}}}
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}
	url := fmt.Sprintf("%s/%s:streamGenerateContent?alt=sse", apiBase, s.cfg.Model)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", s.cfg.APIKey)
	return s.stream(req, emit)
}

func (s *Service) stream(req *http.Request, emit llm.Emit) error {
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	return llm.ScanSSE(resp.Body, func(data string) error {
		var chunk genChunk
		if json.Unmarshal([]byte(data), &chunk) == nil {
			for _, c := range chunk.Candidates {
				for _, p := range c.Content.Parts {
					if err := emit(p.Text); err != nil {
						return err
					}
				}
			}
		}
		return nil // Skip malformed chunks.
	})
}

// toContents converts the conversation into Gemini contents. The system prompt
// is sent separately as systemInstruction, and the assistant role maps to
// "model".
func toContents(convo *frames.LLMContext) []map[string]any {
	msgs := convo.Messages()
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		role := "user"
		if m.Role == frames.RoleAssistant {
			role = "model"
		}
		out = append(out, map[string]any{
			"role":  role,
			"parts": []map[string]any{{"text": m.Text}},
		})
	}
	return out
}
