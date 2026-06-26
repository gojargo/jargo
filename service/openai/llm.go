// Package openai provides OpenAI's LLM, STT and TTS services, plus the
// OpenAI-compatible LLM base that other providers (Groq, Together, Fireworks and
// the rest) wrap with their own base URL, key and default model.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("openai: unexpected status")

const (
	defaultLLMBaseURL   = "https://api.openai.com/v1"
	defaultLLMModel     = "gpt-4o-mini"
	defaultLLMMaxTokens = 1024
)

// LLMConfig configures an OpenAI (or OpenAI-compatible) LLM service. The
// sampling controls are pointers so a deliberate zero is distinguishable from
// "unset"; a nil value is omitted from the request, leaving the API default.
type LLMConfig struct {
	// APIKey is the API key; empty uses the provider's env var.
	APIKey string
	// BaseURL overrides the API base (e.g. an OpenAI-compatible endpoint).
	BaseURL string
	// Model is the model id; empty uses the provider default.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// MaxCompletionTokens caps the completion length on models that require it in
	// place of MaxTokens; nil omits it.
	MaxCompletionTokens *int
	// Temperature is the sampling temperature (0.0 to 2.0); nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter (0.0 to 1.0); nil omits it.
	TopP *float64
	// FrequencyPenalty penalizes frequent tokens (-2.0 to 2.0); nil omits it.
	FrequencyPenalty *float64
	// PresencePenalty penalizes already-present tokens (-2.0 to 2.0); nil omits it.
	PresencePenalty *float64
	// Seed requests deterministic sampling for a fixed seed; nil omits it.
	Seed *int
	// Extra sets arbitrary additional request-body fields not modeled above
	// (e.g. provider-specific parameters), applied to every request.
	Extra map[string]any
}

// LLMService is a streaming OpenAI-compatible chat-completions LLM processor.
type LLMService struct {
	*llm.Base
	cfg  LLMConfig
	http *http.Client
}

// NewLLM builds an OpenAI LLM service.
func NewLLM(cfg LLMConfig) *LLMService {
	return NewCompatLLM("OpenAILLM", defaultLLMBaseURL, "OPENAI_API_KEY", defaultLLMModel, cfg)
}

// NewCompatLLM builds an LLM service for any OpenAI-compatible endpoint. name is
// the processor label, baseURL the API base, envVar the key's environment
// variable, and defaultModel the model used when cfg.Model is empty.
func NewCompatLLM(name, baseURL, envVar, defaultModel string, cfg LLMConfig) *LLMService {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv(envVar)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = baseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaultLLMMaxTokens
	}
	s := &LLMService{cfg: cfg, http: &http.Client{}}
	s.Base = llm.New(name, s)
	return s
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model               string        `json:"model"`
	Messages            []chatMessage `json:"messages"`
	Stream              bool          `json:"stream"`
	MaxTokens           int           `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int          `json:"max_completion_tokens,omitempty"`
	Temperature         *float64      `json:"temperature,omitempty"`
	TopP                *float64      `json:"top_p,omitempty"`
	FrequencyPenalty    *float64      `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64      `json:"presence_penalty,omitempty"`
	Seed                *int          `json:"seed,omitempty"`
}

// encodeBody marshals the request, merging any extra fields over the modeled
// ones. The merge cost is paid only when extra is non-empty.
func encodeBody(req chatRequest, extra map[string]any) ([]byte, error) {
	if len(extra) == 0 {
		return json.Marshal(req)
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	maps.Copy(m, extra)
	return json.Marshal(m)
}

type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// Generate streams a chat completion, emitting each content delta.
func (s *LLMService) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	reqBody := chatRequest{
		Model:               s.cfg.Model,
		Messages:            toMessages(convo),
		Stream:              true,
		MaxTokens:           s.cfg.MaxTokens,
		MaxCompletionTokens: s.cfg.MaxCompletionTokens,
		Temperature:         s.cfg.Temperature,
		TopP:                s.cfg.TopP,
		FrequencyPenalty:    s.cfg.FrequencyPenalty,
		PresencePenalty:     s.cfg.PresencePenalty,
		Seed:                s.cfg.Seed,
	}
	body, err := encodeBody(reqBody, s.cfg.Extra)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return s.stream(req, emit)
}

func (s *LLMService) stream(req *http.Request, emit llm.Emit) error {
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
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
			return emit(chunk.Choices[0].Delta.Content)
		}
		return nil // Skip empty or malformed chunks.
	})
}

// toMessages converts the conversation into OpenAI chat messages, with the
// system prompt as the leading system message.
func toMessages(convo *frames.LLMContext) []chatMessage {
	var out []chatMessage
	if sys := convo.System(); sys != "" {
		out = append(out, chatMessage{Role: "system", Content: sys})
	}
	for _, m := range convo.Messages() {
		out = append(out, chatMessage{Role: string(m.Role), Content: m.Text})
	}
	return out
}
