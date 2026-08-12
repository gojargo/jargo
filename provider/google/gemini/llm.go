package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"

	"github.com/gojargo/jargo/adapter"
	geminiadapter "github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// RequestShaper customizes how a generateContent request is addressed and
// authorized, so a deployment with a different URL layout or auth scheme (Vertex
// AI, which addresses models per project and location and authorizes with an
// OAuth token) can reuse this implementation. The default shaper targets the
// Gemini API with an api-key header.
type RequestShaper interface {
	// Endpoint returns the full generateContent URL for model, including any
	// query string. stream asks for the streaming form of it, which is a
	// different method on the same model rather than a flag on the request.
	Endpoint(model string, stream bool) string
	// Authorize sets the authorization headers on req. It takes a context
	// because a scheme may have to mint or refresh a token to do so.
	Authorize(ctx context.Context, req *http.Request) error
}

// apiKeyShaper is the standard Gemini API addressing and api-key authorization.
type apiKeyShaper struct{ apiKey string }

func (apiKeyShaper) Endpoint(model string, stream bool) string {
	if !stream {
		return fmt.Sprintf("%s/%s:generateContent", apiBase, model)
	}
	return fmt.Sprintf("%s/%s:streamGenerateContent?alt=sse", apiBase, model)
}

func (s apiKeyShaper) Authorize(_ context.Context, req *http.Request) error {
	req.Header.Set("x-goog-api-key", s.apiKey)
	return nil
}

// Service is a streaming Gemini LLM processor.
type Service struct {
	*llm.Base
	// adapter converts the conversation into the request Gemini takes.
	adapter geminiadapter.Adapter
	cfg     Config
	http    *http.Client
	shaper  RequestShaper
}

// NewLLM builds a Gemini LLM service.
func NewLLM(cfg Config) *Service {
	return NewShapedLLM("GoogleLLM", apiKeyShaper{apiKey: cfg.APIKey}, cfg)
}

// NewShapedLLM builds a Gemini LLM service whose requests are addressed and
// authorized by shaper. It is the base for deployments that do not use the
// Gemini API's own URL layout or api-key auth; name is the processor label.
func NewShapedLLM(name string, shaper RequestShaper, cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	s := &Service{cfg: cfg, http: &http.Client{}, shaper: shaper}
	s.Base = llm.New(name, s)
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
	body, err := s.requestBody(convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, false)
	if err != nil {
		return err
	}
	req, err := s.newRequest(ctx, body)
	if err != nil {
		return err
	}
	return s.stream(req, emit)
}

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *Service) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	body, err := s.requestBody(
		convo, adapter.Options{SystemInstruction: opts.SystemInstruction}, false,
	)
	if err != nil {
		return "", err
	}
	if opts.MaxTokens > 0 {
		if cfg, ok := body["generationConfig"].(map[string]any); ok {
			cfg["maxOutputTokens"] = opts.MaxTokens
		}
	}
	req, err := s.newRequestTo(ctx, body, false)
	if err != nil {
		return "", err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return "", llm.AsCompletionTimeout(ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	var answer genChunk
	if err := json.NewDecoder(resp.Body).Decode(&answer); err != nil {
		return "", err
	}
	for _, c := range answer.Candidates {
		for _, p := range c.Content.Parts {
			if p.Text != "" {
				return p.Text, nil
			}
		}
	}
	return "", nil
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
func (s *Service) requestBody(
	convo *frames.LLMContext, opts adapter.Options, withTools bool,
) (map[string]any, error) {
	p, err := s.adapter.LLMInvocationParams(convo, opts)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"contents":         p.Contents,
		"generationConfig": s.genConfig(),
	}
	if len(s.cfg.SafetySettings) > 0 {
		body["safetySettings"] = s.cfg.SafetySettings
	}
	if p.SystemInstruction != "" {
		body["systemInstruction"] = map[string]any{
			keyParts: []map[string]any{{keyText: p.SystemInstruction}},
		}
	}
	if withTools && len(p.Tools) > 0 {
		body["tools"] = p.Tools
	}
	return body, nil
}

// newRequest marshals reqBody and builds the streaming generateContent request.
func (s *Service) newRequest(ctx context.Context, reqBody map[string]any) (*http.Request, error) {
	return s.newRequestTo(ctx, reqBody, true)
}

// newRequestTo marshals reqBody and builds the request, streaming or not.
func (s *Service) newRequestTo(
	ctx context.Context, reqBody map[string]any, stream bool,
) (*http.Request, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, s.shaper.Endpoint(s.cfg.Model, stream), bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := s.shaper.Authorize(ctx, req); err != nil {
		return nil, err
	}
	return req, nil
}

func (s *Service) stream(req *http.Request, emit llm.Emit) error {
	s.StartTTFBMetrics()
	resp, err := s.http.Do(req)
	s.StopTTFBMetrics()
	if err != nil {
		return llm.AsCompletionTimeout(req.Context(), err)
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
	body, err := s.requestBody(convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, true)
	if err != nil {
		return err
	}
	req, err := s.newRequest(ctx, body)
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
	s.StartTTFBMetrics()
	resp, err := s.http.Do(req)
	s.StopTTFBMetrics()
	if err != nil {
		return llm.AsCompletionTimeout(req.Context(), err)
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

// LLMAdapter returns the adapter this service converts through, so the base can
// add the tools it implements itself to what every request advertises. It
// implements llm.AdapterHolder.
func (s *Service) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }

// MessagesForLogging renders the conversation as this provider will see it, for
// the generation span. It implements llm.TraceRenderer.
func (s *Service) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	return s.adapter.MessagesForLogging(convo)
}

// ToolsForLogging renders the toolset as this provider will see it, for the
// generation span. It implements llm.TraceRenderer.
func (s *Service) ToolsForLogging(schema frames.ToolsSchema) []any {
	return adapter.ToolsForLogging(&s.adapter, schema)
}
