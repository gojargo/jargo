package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/adapter"
	geminiadapter "github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
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
	s.warnIfThinkingBudgetIgnored()
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
		// FinishReason says why the model stopped. Anything but a normal stop
		// leaves the turn short of what it would otherwise have said.
		FinishReason string `json:"finishReason"` //nolint:tagliatelle // Gemini REST uses camelCase keys
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
		return "", errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
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
	if s.cfg.Seed != nil {
		g["seed"] = *s.cfg.Seed
	}
	if t := thinkingParams(s.cfg.Thinking); t != nil {
		g[keyThinkingConfig] = t
	}
	maps.Copy(g, s.cfg.Extra)
	// Applied last, so a thinking configuration set explicitly, or through
	// Extra, wins over the low-latency default.
	s.applyThinkingDefault(g)
	return g
}

// thinkingParams renders a thinking configuration, or nil when there is none to
// send.
func thinkingParams(t *ThinkingConfig) map[string]any {
	if t == nil {
		return nil
	}
	params := map[string]any{}
	if t.Level != "" {
		params["thinkingLevel"] = t.Level
	}
	if t.Budget != nil {
		params["thinkingBudget"] = *t.Budget
	}
	if t.IncludeThoughts {
		params["includeThoughts"] = true
	}
	if len(params) == 0 {
		return nil
	}
	return params
}

// applyThinkingDefault turns thinking off on the flash models, unless the
// configuration says otherwise.
//
// A flash model is the one a voice pipeline reaches for, and thinking costs it
// the latency it was chosen for: the model emits nothing at all while it
// thinks. The control differs by generation, so the default does too: the 2.5
// series takes a budget of zero, and the 3 series takes the lowest thinking
// level the model accepts. An image model is left alone, having no such control.
func (s *Service) applyThinkingDefault(g map[string]any) {
	model := s.cfg.Model
	if _, set := g[keyThinkingConfig]; set || strings.Contains(model, "image") {
		return
	}
	switch {
	case strings.HasPrefix(model, "gemini-2.5-flash"):
		g[keyThinkingConfig] = map[string]any{"thinkingBudget": 0}
	case strings.HasPrefix(model, "gemini-3") && strings.Contains(model, "flash"):
		g[keyThinkingConfig] = map[string]any{"thinkingLevel": lowestThinkingLevel(model)}
	}
}

// lowestThinkingLevel is the least a model will think. A model not named here
// accepts "minimal", the fastest setting.
func lowestThinkingLevel(model string) string {
	for prefix, lowest := range lowestThinkingLevels {
		if strings.HasPrefix(model, prefix) {
			return lowest
		}
	}
	return "minimal"
}

// warnIfThinkingBudgetIgnored reports a budget set on a model that reads a level
// instead. Whether such a model honors the budget, ignores it, or refuses the
// request varies by model and by backend, and a refusal names no field, so this
// warning is the only signal there is.
func (s *Service) warnIfThinkingBudgetIgnored() {
	if s.cfg.Thinking == nil || s.cfg.Thinking.Budget == nil {
		return
	}
	if !strings.HasPrefix(s.cfg.Model, "gemini-3") {
		return
	}
	slog.Warn("a thinking budget was set on a model that reads a thinking level",
		"service", s.Name(), "model", s.cfg.Model)
}

// handleFinishReason logs why the model stopped, when it stopped for a notable
// reason: the answer was withheld for safety or recitation, a tool call was
// refused, or the output ran into the token limit. Whatever text did arrive is
// still passed on, so without this the turn simply ends short with nothing said
// about why.
func (s *Service) handleFinishReason(reason string) {
	switch reason {
	case "", "STOP", "FINISH_REASON_UNSPECIFIED":
		return
	}
	slog.Warn("the model stopped before it had finished answering",
		"service", s.Name(), "model", s.cfg.Model, "reason", reason)
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
	return s.streamChunks(req, func(chunk genChunk) error {
		for _, c := range chunk.Candidates {
			for _, p := range c.Content.Parts {
				if err := emit(p.Text); err != nil {
					return err
				}
			}
			s.handleFinishReason(c.FinishReason)
		}
		return nil
	})
}

// streamChunks issues req and hands each chunk of the streamed answer to onChunk,
// bounding how long the model may go quiet for.
//
// A request the API accepts and then never answers is the case this exists for:
// nothing closes such a stream, so without a bound the turn stays open and the
// caller waits on a bot that will never speak. With RetryOnTimeout set, a first
// chunk that does not arrive in time re-issues the request once instead, which
// is safe only up to that point: a chunk that has been handed on is already
// downstream, so re-issuing after one would say everything twice.
func (s *Service) streamChunks(req *http.Request, onChunk func(genChunk) error) error {
	if s.cfg.RetryOnTimeout {
		retried, err := s.streamAttempt(req, s.cfg.retryTimeout(), onChunk)
		if err == nil || !retried {
			return err
		}
		slog.DebugContext(req.Context(), "re-issuing a generation that never began",
			"service", s.Name(), "model", s.cfg.Model)
	}
	_, err := s.streamAttempt(req, s.cfg.streamIdleTimeout(), onChunk)
	return err
}

// streamAttempt runs one attempt, reporting whether it timed out before the
// model had said anything, which is the only point a request can be re-issued
// from. Every chunk after the first is bounded by the idle timeout.
func (s *Service) streamAttempt(
	req *http.Request, firstChunk time.Duration, onChunk func(genChunk) error,
) (retriable bool, err error) {
	// The request is rebuilt per attempt: its body has been read by the time an
	// attempt ends, and a re-issue needs one to send.
	attempt, err := cloneRequest(req)
	if err != nil {
		return false, err
	}

	// The watchdog cancels the request when the model goes quiet for too long,
	// which is what unblocks the read. The flag tells that cancellation apart
	// from the pipeline canceling the turn.
	ctx, cancel := context.WithCancel(attempt.Context())
	defer cancel()
	var timedOut atomic.Bool
	idle := s.cfg.streamIdleTimeout()
	var watchdog *time.Timer
	arm := func(d time.Duration) {
		if d <= 0 {
			return
		}
		if watchdog == nil {
			watchdog = time.AfterFunc(d, func() { timedOut.Store(true); cancel() })
			return
		}
		watchdog.Reset(d)
	}
	defer func() {
		if watchdog != nil {
			watchdog.Stop()
		}
	}()

	s.StartTTFBMetrics()
	arm(firstChunk)
	resp, err := s.http.Do(attempt.WithContext(ctx))
	if err != nil {
		return timedOut.Load(), s.streamError(attempt.Context(), &timedOut, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return false, errs.NewHTTPStatusError(resp.StatusCode,
			fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}

	// began records that the model has said something, which is the point past
	// which the request can no longer be re-issued.
	var began bool
	err = llm.ScanSSE(resp.Body, func(data string) error {
		// Any chunk at all is the model still producing, so the bound is on the
		// gap between chunks rather than on the response as a whole.
		arm(idle)
		began = true
		chunk, ok := parseChunk(data)
		if !ok || len(chunk.Candidates) == 0 {
			return nil
		}
		// A leading chunk can carry usage metadata and no candidates, so TTFB
		// ends at the first chunk that holds model output.
		s.StopTTFBMetrics()
		return onChunk(chunk)
	})
	if err != nil {
		return timedOut.Load() && !began, s.streamError(attempt.Context(), &timedOut, err)
	}
	return false, nil
}

// parseChunk reads one streamed chunk, reporting whether it could be read at
// all. One that could not is skipped rather than fatal: the rest of the answer
// is still worth having.
func parseChunk(data string) (genChunk, bool) {
	var chunk genChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return genChunk{}, false
	}
	return chunk, true
}

// streamError names the failure that ended a stream. A read the watchdog cut
// short is reported as a completion timeout, which is what tells the base to
// report it as one and notify anything watching for it.
func (s *Service) streamError(ctx context.Context, timedOut *atomic.Bool, err error) error {
	if timedOut.Load() {
		return fmt.Errorf("%w: the model stopped producing for %s",
			llm.ErrCompletionTimeout, s.cfg.streamIdleTimeout())
	}
	return llm.AsCompletionTimeout(ctx, err)
}

// cloneRequest copies a request along with its body, so an attempt that consumed
// one can be made again.
func cloneRequest(req *http.Request) (*http.Request, error) {
	if req.GetBody == nil {
		return req, nil
	}
	body, err := req.GetBody()
	if err != nil {
		return nil, err
	}
	clone := req.Clone(req.Context())
	clone.Body = body
	return clone, nil
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
	ts := &geminiToolStream{sink: sink}
	return s.streamChunks(req, func(chunk genChunk) error {
		if err := ts.consume(chunk); err != nil {
			return err
		}
		for _, c := range chunk.Candidates {
			s.handleFinishReason(c.FinishReason)
		}
		return nil
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
