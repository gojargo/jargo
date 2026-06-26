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
	"strings"

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
	// toolTypeFunction is the only tool type OpenAI's chat API defines.
	toolTypeFunction = "function"
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
	s.Base.SetModel(cfg.Model)
	return s
}

type chatMessage struct {
	Role       string        `json:"role"`
	Content    string        `json:"content"`
	ToolCalls  []toolCallMsg `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// toolCallMsg is an assistant tool-call entry in a request message.
type toolCallMsg struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function toolCallFunc `json:"function"`
}

type toolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// openaiTool is a function tool advertised on the request.
type openaiTool struct {
	Type     string     `json:"type"`
	Function openaiFunc `json:"function"`
}

type openaiFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
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
	Tools               []openaiTool  `json:"tools,omitempty"`
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

type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatDelta struct {
	Content   string          `json:"content"`
	ToolCalls []toolCallDelta `json:"tool_calls"`
}

type chatChunk struct {
	Choices []struct {
		Delta chatDelta `json:"delta"`
	} `json:"choices"`
}

// Generate streams a chat completion, emitting each content delta.
func (s *LLMService) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	body, err := encodeBody(s.baseRequest(convo), s.cfg.Extra)
	if err != nil {
		return err
	}
	req, err := s.newHTTPRequest(ctx, body)
	if err != nil {
		return err
	}
	return s.stream(req, emit)
}

// baseRequest builds the streaming request shared by both generation paths.
func (s *LLMService) baseRequest(convo *frames.LLMContext) chatRequest {
	return chatRequest{
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
}

// newHTTPRequest builds the POST request to the chat-completions endpoint.
func (s *LLMService) newHTTPRequest(ctx context.Context, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return req, nil
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

// GenerateWithTools streams a tool-capable completion. It emits text deltas to
// the sink as they arrive and, once the stream completes, reports each tool call
// the model produced. The conversation's tools are sent on the request, and any
// tool turns already in the context are replayed as the matching messages.
func (s *LLMService) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	reqBody := s.baseRequest(convo)
	if tools := convo.Tools(); len(tools) > 0 {
		reqBody.Tools = toTools(tools)
	}
	body, err := encodeBody(reqBody, s.cfg.Extra)
	if err != nil {
		return err
	}
	req, err := s.newHTTPRequest(ctx, body)
	if err != nil {
		return err
	}
	return s.streamTools(req, sink)
}

// toolAccumulator coalesces the streamed fragments of one tool call.
type toolAccumulator struct {
	id, name string
	args     strings.Builder
}

// toolCoalescer assembles tool calls from streamed deltas, keyed by index and
// remembering arrival order.
type toolCoalescer struct {
	calls map[int]*toolAccumulator
	order []int
}

// add folds one delta in: text deltas go straight to the sink, tool-call
// fragments accumulate by index.
func (c *toolCoalescer) add(delta chatDelta, sink llm.Sink) error {
	if len(delta.ToolCalls) == 0 {
		return sink.Text(delta.Content)
	}
	for _, d := range delta.ToolCalls {
		a := c.calls[d.Index]
		if a == nil {
			a = &toolAccumulator{}
			c.calls[d.Index] = a
			c.order = append(c.order, d.Index)
		}
		if d.ID != "" {
			a.id = d.ID
		}
		a.name += d.Function.Name
		a.args.WriteString(d.Function.Arguments)
	}
	return nil
}

// emit reports the assembled calls to the sink in arrival order, defaulting
// empty arguments to an empty JSON object.
func (c *toolCoalescer) emit(sink llm.Sink) error {
	for _, idx := range c.order {
		a := c.calls[idx]
		if a.name == "" {
			continue
		}
		args := a.args.String()
		if args == "" {
			args = "{}"
		}
		if err := sink.Tool(frames.ToolCall{ID: a.id, Name: a.name, Args: json.RawMessage(args)}); err != nil {
			return err
		}
	}
	return nil
}

// streamTools streams a tool-capable completion, emitting text deltas live and
// coalescing the streamed tool_call fragments into whole calls reported once the
// stream completes.
func (s *LLMService) streamTools(req *http.Request, sink llm.Sink) error {
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}

	c := &toolCoalescer{calls: map[int]*toolAccumulator{}}
	scanErr := llm.ScanSSE(resp.Body, func(data string) error {
		var chunk chatChunk
		if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
			return c.add(chunk.Choices[0].Delta, sink)
		}
		return nil // Skip empty or malformed chunks.
	})
	if scanErr != nil {
		return scanErr
	}
	return c.emit(sink)
}

// toMessages converts the conversation into OpenAI chat messages, with the
// system prompt as the leading system message. Tool turns become an assistant
// message carrying tool_calls and one "tool" message per result.
func toMessages(convo *frames.LLMContext) []chatMessage {
	var out []chatMessage
	if sys := convo.System(); sys != "" {
		out = append(out, chatMessage{Role: "system", Content: sys})
	}
	for _, m := range convo.Messages() {
		switch {
		case len(m.ToolResults) > 0:
			for _, r := range m.ToolResults {
				out = append(out, chatMessage{Role: "tool", ToolCallID: r.ID, Content: r.Content})
			}
		case len(m.ToolCalls) > 0:
			msg := chatMessage{Role: string(frames.RoleAssistant), Content: m.Text}
			for _, c := range m.ToolCalls {
				args := string(c.Args)
				if args == "" {
					args = "{}"
				}
				msg.ToolCalls = append(msg.ToolCalls, toolCallMsg{
					ID:       c.ID,
					Type:     toolTypeFunction,
					Function: toolCallFunc{Name: c.Name, Arguments: args},
				})
			}
			out = append(out, msg)
		default:
			out = append(out, chatMessage{Role: string(m.Role), Content: m.Text})
		}
	}
	return out
}

// toTools converts the context's tools into OpenAI function tools.
func toTools(tools []frames.Tool) []openaiTool {
	out := make([]openaiTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, openaiTool{
			Type: toolTypeFunction,
			Function: openaiFunc{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	return out
}
