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
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/adapter"
	geminiadapter "github.com/gojargo/jargo/adapter/gemini"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/google/uuid"
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
	// live is what an LLMUpdateSettingsFrame may change while the pipeline runs.
	// It is seeded from cfg, and is what every request is built from.
	live   *LLMSettings
	http   *http.Client
	shaper RequestShaper
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
	s := &Service{cfg: cfg, live: newLLMSettings(cfg), http: &http.Client{}, shaper: shaper}
	s.Base = llm.New(name, s)
	s.Base.SetModel(cfg.Model)
	s.warnIfThinkingBudgetIgnored()
	return s
}

// genPart is one part of a candidate's content: text, a thought, a function
// call or inline data, any of which the model may sign.
type genPart struct {
	// Text is nil when the part carries none, which a signed trailing chunk is
	// told apart by: its text is present and empty.
	Text *string `json:"text"`
	// Thought marks text that is a summary of the model's reasoning rather than
	// part of its answer.
	Thought bool `json:"thought"`
	// ThoughtSignature is the opaque signature the model put on this part, which
	// it expects back on the same part on the next turn.
	ThoughtSignature string `json:"thoughtSignature"` //nolint:tagliatelle // Gemini REST uses camelCase keys
	FunctionCall     *struct {
		ID   string          `json:"id"`
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	} `json:"functionCall"` //nolint:tagliatelle // Gemini REST uses camelCase keys
	InlineData *struct {
		MimeType string `json:"mimeType"` //nolint:tagliatelle // Gemini REST uses camelCase keys
		Data     string `json:"data"`
	} `json:"inlineData"` //nolint:tagliatelle // Gemini REST uses camelCase keys
}

// text returns the part's text, empty when it carries none.
func (p genPart) text() string {
	if p.Text == nil {
		return ""
	}
	return *p.Text
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
	cfg := s.requestConfig()
	body, err := s.requestBodyFor(cfg, convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, false)
	if err != nil {
		return err
	}
	req, err := s.newRequestTo(ctx, cfg.Model, body, true)
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
	// The call's own instruction, or else the service's: an inference is still
	// this service answering. The tools go too, as on a streamed turn.
	instruction := opts.SystemInstruction
	if instruction == "" {
		instruction = s.SystemInstruction()
	}
	cfg := s.requestConfig()
	body, err := s.requestBodyFor(cfg, convo, adapter.Options{SystemInstruction: instruction}, true)
	if err != nil {
		return "", err
	}
	if opts.MaxTokens > 0 {
		if g, ok := body["generationConfig"].(map[string]any); ok {
			g["maxOutputTokens"] = opts.MaxTokens
		}
	}
	if schema := s.CheckResponseSchema(ctx, opts.ResponseSchema); schema != nil {
		if g, ok := body["generationConfig"].(map[string]any); ok {
			g["responseMimeType"] = "application/json"
			g["responseJsonSchema"] = schema
		}
	}
	req, err := s.newRequestTo(ctx, cfg.Model, body, false)
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
			// A thought is the model reasoning, not its answer.
			if p.text() != "" && !p.Thought {
				return p.text(), nil
			}
		}
	}
	return "", nil
}

// SupportsResponseSchema implements llm.ResponseSchemaSupporter.
func (s *Service) SupportsResponseSchema() bool { return true }

// modelVersion reads the version out of a Gemini model id: "gemini-2.5-flash"
// is 2.5 and "gemini-2.0-flash" 2.0.
//
//nolint:gochecknoglobals // a compiled pattern
var modelVersion = regexp.MustCompile(`gemini(?:-[a-z]+)*-(\d+)(?:\.(\d+))?`)

// ModelSupportsResponseSchema implements llm.ResponseSchemaSupporter. Gemini
// takes a JSON schema from the 2.5 models on. A model id without a version is
// assumed to support it.
func (s *Service) ModelSupportsResponseSchema(model string) bool {
	m := modelVersion.FindStringSubmatch(model)
	if m == nil {
		return true
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	return major > 2 || (major == 2 && minor >= 5)
}

// genConfig builds the generationConfig block from the settings as they
// stand.
func (s *Service) genConfig() map[string]any { return s.genConfigFor(s.requestConfig()) }

// genConfigFor builds the generationConfig block from cfg's controls.
func (s *Service) genConfigFor(cfg Config) map[string]any {
	g := map[string]any{"maxOutputTokens": cfg.MaxTokens}
	if cfg.Temperature != nil {
		g["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		g["topP"] = *cfg.TopP
	}
	if cfg.TopK != nil {
		g["topK"] = *cfg.TopK
	}
	if cfg.Seed != nil {
		g["seed"] = *cfg.Seed
	}
	if t := thinkingParams(cfg.Thinking); t != nil {
		g[keyThinkingConfig] = t
	}
	maps.Copy(g, cfg.Extra)
	// Applied last, so a thinking configuration set explicitly, or through
	// Extra, wins over the low-latency default.
	s.applyThinkingDefault(g, cfg.Model)
	return g
}

// thinkingParams renders a thinking configuration, or nil when there is none to
// send. It is ThinkingParams under this package's own name.
func thinkingParams(t *ThinkingConfig) map[string]any { return ThinkingParams(t) }

// ThinkingParams renders a thinking configuration as the generationConfig block
// the API takes, or nil when there is nothing to send. It is exported for the
// Live service, which carries the same configuration into its session setup.
func ThinkingParams(t *ThinkingConfig) map[string]any {
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

// applyThinkingDefault sends the model its lowest thinking setting when none
// is configured.
//
// It applies the model's entry in thinkingDefaults, if it has one. A generation
// config that already sets thinkingConfig, from the config or from Extra, is
// left as it is, and so is every one when UseProviderThinkingDefault is set. The
// first request for each model logs the setting sent, or that the model runs at
// Google's default, warning for an unlisted model that can think.
func (s *Service) applyThinkingDefault(g map[string]any, model string) {
	if _, set := g[keyThinkingConfig]; set {
		return
	}
	name := modelName(model)
	var setting string
	if d, ok := thinkingDefaults[name]; ok && !s.UseProviderThinkingDefault() {
		thinking := maps.Clone(d)
		g[keyThinkingConfig] = thinking
		setting = keyThinkingConfig + "=" + thinkingSetting(thinking)
	}
	s.ReportThinkingDefault(model, setting, canThink(name), "Google", "Thinking in the service's Config")
}

// thinkingSetting renders a thinking default as the JSON it is sent as, for the
// log line that reports it.
func thinkingSetting(thinking map[string]any) string {
	b, err := json.Marshal(thinking)
	if err != nil {
		return fmt.Sprint(thinking)
	}
	return string(b)
}

// warnIfThinkingBudgetIgnored reports a budget set on a model that reads a level
// instead. Whether such a model honors the budget, ignores it, or refuses the
// request varies by model and by backend, and a refusal names no field, so this
// warning is the only signal there is.
func (s *Service) warnIfThinkingBudgetIgnored() {
	if s.cfg.Thinking == nil || s.cfg.Thinking.Budget == nil {
		return
	}
	name := modelName(s.cfg.Model)
	if !slices.ContainsFunc(modelsSupportingThinkingLevel, func(prefix string) bool {
		return strings.HasPrefix(name, prefix)
	}) {
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
		"service", s.Name(), "model", s.Model(), "reason", reason)
}

// prefillSupportedPatterns lists the models that accept a request whose
// contents end with a model turn, continuing that turn as the start of the
// response. The newer models reject such a request outright, so this is a
// frozen legacy set: a model not matching it is assumed to reject one and is
// given a trailing user turn instead. Prefixes are matched from the start of
// the id, and gemini-3.5-flash is left out although it accepts a trailing model
// turn, because it shares its prefix with gemini-3.5-flash-lite, which does not.
//
//nolint:gochecknoglobals // fixed lookup table
var prefillSupportedPatterns = []string{
	"gemini-2.",
	"gemini-3-",
	"gemini-3.1-",
	"gemini-pro-latest",
}

// supportsPrefill reports whether model continues a trailing model turn rather
// than refusing the request that carries one.
func supportsPrefill(model string) bool {
	for _, p := range prefillSupportedPatterns {
		if strings.HasPrefix(model, p) {
			return true
		}
	}
	return false
}

// requestBody builds the generateContent body from the settings as they stand,
// optionally advertising tools.
func (s *Service) requestBody(
	convo *frames.LLMContext, opts adapter.Options, withTools bool,
) (map[string]any, error) {
	return s.requestBodyFor(s.requestConfig(), convo, opts, withTools)
}

// requestBodyFor builds the generateContent body from cfg.
func (s *Service) requestBodyFor(
	cfg Config, convo *frames.LLMContext, opts adapter.Options, withTools bool,
) (map[string]any, error) {
	// A model that cannot continue a trailing model turn rejects the request
	// carrying one, which is what an assistant message last in the conversation
	// converts to: filler spoken while a tool ran, say.
	opts.EnsureLastMessageIsUser = !supportsPrefill(cfg.Model)
	p, err := s.adapter.LLMInvocationParams(convo, opts)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"contents":         p.Contents,
		"generationConfig": s.genConfigFor(cfg),
	}
	if len(cfg.SafetySettings) > 0 {
		body["safetySettings"] = cfg.SafetySettings
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

// newRequestTo marshals reqBody and builds the request to model, streaming or
// not.
func (s *Service) newRequestTo(
	ctx context.Context, model string, reqBody map[string]any, stream bool,
) (*http.Request, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, s.shaper.Endpoint(model, stream), bytes.NewReader(body),
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
	rs := &responseStream{s: s, ctx: req.Context(), text: emit}
	return s.streamChunks(req, rs.consume)
}

// responseStream reads the parts of one streamed response: answer text goes to
// text, thoughts and thought signatures go downstream as frames, and each
// function call goes to tool.
type responseStream struct {
	s   *Service
	ctx context.Context //nolint:containedctx // scoped to one response
	// text receives each chunk of the answer.
	text func(string) error
	// tool receives each function call. It is nil when no tools were
	// advertised, and a call is then not reported.
	tool func(frames.ToolCall) error
	// accumulated is the answer text so far, which a signature on a text part
	// is bookmarked by.
	accumulated strings.Builder
}

// consume reads every part of a chunk.
func (r *responseStream) consume(chunk genChunk) error {
	for _, c := range chunk.Candidates {
		for _, p := range c.Content.Parts {
			if err := r.part(p); err != nil {
				return err
			}
		}
		r.s.handleFinishReason(c.FinishReason)
	}
	return nil
}

// part reads one streamed part.
func (r *responseStream) part(p genPart) error {
	var callID string
	switch {
	case p.text() != "":
		if p.Thought {
			// Gemini sends whole thoughts rather than chunks of one, so each is
			// bracketed in a start and an end of its own.
			return r.pushThought(p)
		}
		r.accumulated.WriteString(p.text())
		if err := r.text(p.text()); err != nil {
			return err
		}
	case p.FunctionCall != nil:
		// A turn that only calls tools produces no answer text, so the call
		// itself is what the caller gets and TTFAT ends here rather than going
		// unmeasured.
		r.s.StopTTFATMetrics(r.ctx)
		callID = p.FunctionCall.ID
		if callID == "" {
			callID = uuid.NewString()
		}
		if r.tool != nil {
			args := p.FunctionCall.Args
			if len(args) == 0 {
				args = json.RawMessage("{}")
			}
			if err := r.tool(frames.ToolCall{ID: callID, Name: p.FunctionCall.Name, Args: args}); err != nil {
				return err
			}
		}
	}
	return r.keepThoughtSignature(p, callID)
}

// pushThought pushes a thought part as a whole thought.
func (r *responseStream) pushThought(p genPart) error {
	if err := r.s.PushFrame(r.ctx, frames.NewLLMThoughtStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if err := r.s.PushFrame(r.ctx, frames.NewLLMThoughtTextFrame(p.text()), processor.Downstream); err != nil {
		return err
	}
	if err := r.s.PushFrame(r.ctx, frames.NewLLMThoughtEndFrame(), processor.Downstream); err != nil {
		return err
	}
	return r.keepThoughtSignature(p, "")
}

// keepThoughtSignature writes the signature a part carries into the
// conversation, bookmarked so the adapter can put it back on the right part
// when the conversation is sent again.
//
// Gemini 2.5 signs function calls and then the last part of every response
// after the first call; Gemini 3 signs the last part of every response. The
// response is streamed, though, and a text part may arrive in several chunks:
// the signature can come with the first chunk, or in a trailing chunk whose
// text is empty. A text bookmark therefore holds all the answer text seen so
// far in this response.
func (r *responseStream) keepThoughtSignature(p genPart, callID string) error {
	if p.ThoughtSignature == "" {
		return nil
	}
	var bookmark map[string]any
	switch {
	case p.FunctionCall != nil:
		bookmark = map[string]any{geminiadapter.BookmarkFunctionCall: callID}
	case p.InlineData != nil && p.InlineData.Data != "":
		bookmark = map[string]any{geminiadapter.BookmarkInlineData: map[string]any{
			"mimeType": p.InlineData.MimeType, "data": p.InlineData.Data,
		}}
	case p.Text != nil:
		bookmark = map[string]any{geminiadapter.BookmarkText: r.accumulated.String()}
	default:
		slog.WarnContext(r.ctx, "thought signature found on an unhandled part type",
			"service", r.s.Name())
		return nil
	}
	msg := geminiadapter.NewThoughtSignature(p.ThoughtSignature, bookmark)
	return r.s.PushFrame(r.ctx,
		frames.NewLLMMessagesAppendFrame([]frames.Message{msg}), processor.Downstream)
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
			"service", s.Name(), "model", s.Model())
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
		s.StopTTFBMetrics(ctx)
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
	cfg := s.requestConfig()
	body, err := s.requestBodyFor(cfg, convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, true)
	if err != nil {
		return err
	}
	req, err := s.newRequestTo(ctx, cfg.Model, body, true)
	if err != nil {
		return err
	}
	return s.streamTools(req, sink)
}

// streamTools streams a tool-capable completion, forwarding text and tool calls.
// A call is reported under the id Gemini gave it, or a fresh one when it gave
// none, so its result can be paired back with it.
func (s *Service) streamTools(req *http.Request, sink llm.Sink) error {
	rs := &responseStream{s: s, ctx: req.Context(), text: sink.Text, tool: sink.Tool}
	return s.streamChunks(req, rs.consume)
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
