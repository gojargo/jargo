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
	"maps"
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
	// Gemini content/part map keys, hoisted to avoid repeated string literals.
	keyRole  = "role"
	keyParts = "parts"
	keyName  = "name"
	keyText  = "text"
)

// Config configures the Gemini LLM service. The sampling controls are pointers
// so a deliberate zero is distinguishable from "unset"; a nil value is omitted
// from the request, leaving the API default.
type Config struct {
	// APIKey is the Gemini API key; empty uses the GEMINI_API_KEY env var.
	APIKey string
	// Model is the model id; empty uses a low-latency flash default.
	Model string
	// MaxTokens caps the response length; 0 uses a small default suited to voice.
	MaxTokens int
	// Temperature is the sampling temperature (0.0 to 2.0); nil omits it.
	Temperature *float64
	// TopP is the nucleus-sampling parameter (0.0 to 1.0); nil omits it.
	TopP *float64
	// TopK is the top-k sampling parameter; nil omits it.
	TopK *int
	// Extra sets arbitrary additional generationConfig fields not modeled above,
	// applied to every request.
	Extra map[string]any
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
	s.Base.SetModel(cfg.Model)
	return s
}

// genPart is one part of a candidate's content: text or a function call.
type genPart struct {
	Text         string `json:"text"`
	FunctionCall *struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	} `json:"functionCall"` //nolint:tagliatelle // Gemini REST uses camelCase keys
}

// genChunk is the subset of a streamGenerateContent SSE chunk we read.
type genChunk struct {
	Candidates []struct {
		Content struct {
			Parts []genPart `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// Generate streams a Gemini completion, emitting each text delta.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	req, err := s.newRequest(ctx, s.requestBody(convo, false))
	if err != nil {
		return err
	}
	return s.stream(req, emit)
}

// genConfig builds the generationConfig block from the configured controls.
func (s *Service) genConfig() map[string]any {
	g := map[string]any{"maxOutputTokens": s.cfg.MaxTokens}
	if s.cfg.Temperature != nil {
		g["temperature"] = *s.cfg.Temperature
	}
	if s.cfg.TopP != nil {
		g["topP"] = *s.cfg.TopP
	}
	if s.cfg.TopK != nil {
		g["topK"] = *s.cfg.TopK
	}
	maps.Copy(g, s.cfg.Extra)
	return g
}

// requestBody builds the generateContent body, optionally advertising tools.
func (s *Service) requestBody(convo *frames.LLMContext, withTools bool) map[string]any {
	body := map[string]any{
		"contents":         toContents(convo),
		"generationConfig": s.genConfig(),
	}
	if sys := convo.System(); sys != "" {
		body["systemInstruction"] = map[string]any{keyParts: []map[string]any{{keyText: sys}}}
	}
	if withTools {
		if tools := convo.Tools(); len(tools) > 0 {
			body["tools"] = toTools(tools)
		}
	}
	return body
}

// newRequest marshals reqBody and builds the streamGenerateContent request.
func (s *Service) newRequest(ctx context.Context, reqBody map[string]any) (*http.Request, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/%s:streamGenerateContent?alt=sse", apiBase, s.cfg.Model)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", s.cfg.APIKey)
	return req, nil
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

// GenerateWithTools streams a tool-capable completion. It emits text deltas to
// the sink as they arrive and reports each functionCall the model produces. The
// conversation's tools are sent on the request, and any tool turns already in
// the context are replayed as functionCall / functionResponse parts.
func (s *Service) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	req, err := s.newRequest(ctx, s.requestBody(convo, true))
	if err != nil {
		return err
	}
	return s.streamTools(req, sink)
}

// geminiToolStream consumes streamed parts, forwarding text and assigning each
// functionCall a synthetic id (Gemini has none; results are paired by name).
type geminiToolStream struct {
	sink llm.Sink
	idx  int
}

// part forwards one streamed part to the sink.
func (t *geminiToolStream) part(p genPart) error {
	if p.Text != "" {
		if err := t.sink.Text(p.Text); err != nil {
			return err
		}
	}
	if p.FunctionCall == nil {
		return nil
	}
	args := p.FunctionCall.Args
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	id := fmt.Sprintf("call_%d", t.idx)
	t.idx++
	return t.sink.Tool(frames.ToolCall{ID: id, Name: p.FunctionCall.Name, Args: args})
}

// consume forwards every part of a chunk to the sink.
func (t *geminiToolStream) consume(chunk genChunk) error {
	for _, c := range chunk.Candidates {
		for _, p := range c.Content.Parts {
			if err := t.part(p); err != nil {
				return err
			}
		}
	}
	return nil
}

// streamTools streams a tool-capable completion, forwarding text and tool calls.
func (s *Service) streamTools(req *http.Request, sink llm.Sink) error {
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	ts := &geminiToolStream{sink: sink}
	return llm.ScanSSE(resp.Body, func(data string) error {
		var chunk genChunk
		if json.Unmarshal([]byte(data), &chunk) == nil {
			return ts.consume(chunk)
		}
		return nil // Skip malformed chunks.
	})
}

// toContents converts the conversation into Gemini contents. The system prompt
// is sent separately as systemInstruction, and the assistant role maps to
// "model". Tool turns become functionCall parts (model) and functionResponse
// parts (user); Gemini pairs results to calls by name.
func toContents(convo *frames.LLMContext) []map[string]any {
	msgs := convo.Messages()
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case len(m.ToolResults) > 0:
			out = append(out, map[string]any{keyRole: "user", keyParts: toolResultParts(m.ToolResults)})
		case len(m.ToolCalls) > 0:
			out = append(out, map[string]any{keyRole: "model", keyParts: toolCallParts(m.Text, m.ToolCalls)})
		default:
			role := "user"
			if m.Role == frames.RoleAssistant {
				role = "model"
			}
			out = append(out, map[string]any{
				keyRole:  role,
				keyParts: []map[string]any{{keyText: m.Text}},
			})
		}
	}
	return out
}

// toolCallParts renders an assistant turn's optional preamble and tool calls as
// Gemini parts.
func toolCallParts(text string, calls []frames.ToolCall) []map[string]any {
	parts := make([]map[string]any, 0, len(calls)+1)
	if text != "" {
		parts = append(parts, map[string]any{keyText: text})
	}
	for _, c := range calls {
		args := c.Args
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		parts = append(parts, map[string]any{
			"functionCall": map[string]any{keyName: c.Name, "args": args},
		})
	}
	return parts
}

// toolResultParts renders tool outputs as Gemini functionResponse parts.
func toolResultParts(results []frames.ToolResult) []map[string]any {
	parts := make([]map[string]any, 0, len(results))
	for _, r := range results {
		parts = append(parts, map[string]any{
			"functionResponse": map[string]any{
				keyName:    r.Name,
				"response": functionResponseDict(r.Content),
			},
		})
	}
	return parts
}

// functionResponseDict shapes a tool result for Gemini's functionResponse,
// which requires an object: a JSON object passes through, anything else is
// wrapped as {"value": content}.
func functionResponseDict(content string) any {
	var obj map[string]any
	if json.Unmarshal([]byte(content), &obj) == nil {
		return json.RawMessage(content)
	}
	return map[string]any{"value": content}
}

// toTools converts the context's tools into Gemini functionDeclarations.
func toTools(tools []frames.Tool) []map[string]any {
	decls := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		d := map[string]any{keyName: t.Name}
		if t.Description != "" {
			d["description"] = t.Description
		}
		if params := geminiParameters(t.Parameters); params != nil {
			d["parameters"] = params
		}
		decls = append(decls, d)
	}
	return []map[string]any{{"functionDeclarations": decls}}
}

// geminiParameters returns the tool's JSON-Schema parameters with
// "additionalProperties" stripped (Gemini rejects it). On a parse error the raw
// schema passes through unchanged.
func geminiParameters(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var schema map[string]any
	if json.Unmarshal(raw, &schema) != nil {
		return raw
	}
	stripAdditionalProperties(schema)
	return schema
}

// stripAdditionalProperties recursively removes "additionalProperties" keys.
func stripAdditionalProperties(v any) {
	switch t := v.(type) {
	case map[string]any:
		delete(t, "additionalProperties")
		for _, val := range t {
			stripAdditionalProperties(val)
		}
	case []any:
		for _, val := range t {
			stripAdditionalProperties(val)
		}
	}
}
