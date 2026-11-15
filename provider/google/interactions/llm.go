package interactions

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"

	"github.com/gojargo/jargo/adapter"
	geminiadapter "github.com/gojargo/jargo/adapter/gemini"
	interactionsadapter "github.com/gojargo/jargo/adapter/geminiinteractions"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/google/internal/stream"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
)

// errStatus is returned when the API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("google interactions: unexpected status")

// The keys of an Interactions request.
const (
	keyGenerationConfig = "generation_config"
	keyThinkingLevel    = "thinking_level"
)

// The statuses an interaction that produced a full response ends with.
const (
	statusCompleted      = "completed"
	statusRequiresAction = "requires_action"
)

// RequestShaper says where an Interactions request is sent and how it is
// authorized, so a deployment with a different URL layout or auth scheme (Vertex
// AI, which addresses the API per project and location and authorizes with an
// OAuth token) can reuse this implementation. The default shaper targets the
// Gemini API with an api-key header.
type RequestShaper interface {
	// Endpoint returns the URL interactions are created at.
	Endpoint() string
	// Authorize sets the authorization headers on req. It takes a context
	// because a scheme may have to mint or refresh a token to do so.
	Authorize(ctx context.Context, req *http.Request) error
}

// apiKeyShaper is the Gemini API's addressing and api-key authorization.
type apiKeyShaper struct{ apiKey string }

func (apiKeyShaper) Endpoint() string { return apiURL }

func (s apiKeyShaper) Authorize(_ context.Context, req *http.Request) error {
	req.Header.Set("x-goog-api-key", s.apiKey)
	return nil
}

// Service is a streaming Gemini LLM processor using the Interactions API.
type Service struct {
	*llm.Base
	// adapter converts the conversation into the steps the API takes.
	adapter interactionsadapter.Adapter
	cfg     Config
	// live is what an LLMUpdateSettingsFrame may change while the pipeline runs.
	live   *LLMSettings
	http   *http.Client
	shaper RequestShaper

	// warnedMu guards warnedUnsupported, the settings already reported as
	// ignored, so each is reported once.
	warnedMu          sync.Mutex
	warnedUnsupported map[string]bool
}

// NewLLM builds an Interactions LLM service on the Gemini API.
func NewLLM(cfg Config) *Service {
	return NewShapedLLM("GoogleInteractionsLLM", apiKeyShaper{apiKey: cfg.APIKey}, cfg)
}

// NewShapedLLM builds an Interactions LLM service whose requests are addressed
// and authorized by shaper. It is the base for deployments that do not use the
// Gemini API's own URL or api-key auth; name is the processor label.
func NewShapedLLM(name string, shaper RequestShaper, cfg Config) *Service {
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = defaultMaxTokens
	}
	s := &Service{
		cfg:               cfg,
		live:              newLLMSettings(cfg),
		http:              &http.Client{},
		shaper:            shaper,
		warnedUnsupported: map[string]bool{},
	}
	s.Base = llm.New(name, s)
	s.Base.SetModel(cfg.Model)
	return s
}

// SupportsResponseSchema implements llm.ResponseSchemaSupporter.
func (s *Service) SupportsResponseSchema() bool { return true }

// ModelSupportsResponseSchema implements llm.ResponseSchemaSupporter: every
// model the Interactions API serves can enforce a response schema.
func (s *Service) ModelSupportsResponseSchema(string) bool { return true }

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

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *Service) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	instruction := opts.SystemInstruction
	if instruction == "" {
		instruction = s.SystemInstruction()
	}
	params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{SystemInstruction: instruction})
	if err != nil {
		return "", err
	}
	request := s.buildRequest(s.requestConfig(), params)
	if opts.MaxTokens > 0 {
		generationConfig(request)["max_output_tokens"] = opts.MaxTokens
	}
	if schema := s.CheckResponseSchema(ctx, opts.ResponseSchema); schema != nil {
		request["response_format"] = map[string]any{
			"type":      "text",
			"mime_type": "application/json",
			"schema":    schema,
		}
	}

	req, err := s.newRequest(ctx, request)
	if err != nil {
		return "", err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return "", llm.AsCompletionTimeout(ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return "", statusError(resp)
	}
	var result interaction
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if result.Status != statusCompleted && result.Status != statusRequiresAction {
		slog.WarnContext(ctx, "response incomplete, the interaction ended "+result.Status, "service", s.Name())
	}
	var text strings.Builder
	for _, step := range result.Steps {
		if step.Type != interactionsadapter.StepModelOutput {
			continue
		}
		for _, item := range step.Content {
			text.WriteString(item.Text)
		}
	}
	return text.String(), nil
}

// Generate streams a response to the conversation, emitting each text delta.
// The conversation advertises no tools, but the service's own may still be
// offered, and the calls the model makes to them are run.
func (s *Service) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	var calls []frames.ToolCall
	err := s.processContext(ctx, convo, emit, func(call frames.ToolCall) error {
		calls = append(calls, call)
		return nil
	})
	if err == nil && len(calls) > 0 {
		return s.RunFunctionCalls(ctx, convo, calls)
	}
	return err
}

// GenerateWithTools streams a tool-capable response, emitting text deltas to the
// sink as they arrive and reporting each function call the model made once the
// response is complete.
func (s *Service) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	return s.processContext(ctx, convo, sink.Text, sink.Tool)
}

// processContext runs one streamed response, reporting its token usage once it
// is over, whether it completed or not.
func (s *Service) processContext(
	ctx context.Context, convo *frames.LLMContext, text func(string) error, tool func(frames.ToolCall) error,
) (err error) {
	response := newResponseState(convo)
	defer func() {
		if !s.UsageMetricsEnabled() {
			return
		}
		if perr := s.PushTokenUsage(ctx, response.usage.tokenUsage()); err == nil {
			err = perr
		}
	}()

	params, err := s.adapter.LLMInvocationParams(convo, adapter.Options{SystemInstruction: s.SystemInstruction()})
	if err != nil {
		return err
	}
	cfg := s.requestConfig()
	request := s.buildRequest(cfg, params)
	request["stream"] = true
	req, err := s.newRequest(ctx, request)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")

	err = stream.WithRetry(req, stream.Config{
		Client:         s.http,
		IdleTimeout:    cfg.streamIdleTimeout(),
		RetryOnTimeout: cfg.RetryOnTimeout,
		RetryTimeout:   cfg.retryTimeout(),
		Begin:          s.StartTTFBMetrics,
		Retrying: func(ctx context.Context) {
			slog.DebugContext(ctx, "retrying content generation due to timeout", "service", s.Name())
		},
		StatusError: statusError,
	}, func(ctx context.Context, data string) error {
		// A malformed event is skipped rather than ending the response.
		ev, ok := decodeEvent(data)
		if !ok {
			return nil
		}
		return s.handleEvent(ctx, ev, response, text)
	})
	if err != nil {
		return err
	}
	for _, call := range response.functionCalls {
		if terr := tool(call); terr != nil {
			return terr
		}
	}
	return nil
}

// decodeEvent parses one stream event, reporting whether it was well-formed.
func decodeEvent(data string) (event, bool) {
	var ev event
	if json.Unmarshal([]byte(data), &ev) != nil {
		return event{}, false
	}
	return ev, true
}

// buildRequest builds an Interactions request from the conversation's
// invocation params and the settings as they stand.
func (s *Service) buildRequest(cfg requestSettings, params interactionsadapter.Params) map[string]any {
	request := map[string]any{"model": cfg.Model, "input": params.Input, "store": false}

	if params.SystemInstruction != "" {
		request["system_instruction"] = params.SystemInstruction
	}

	switch {
	case len(params.Tools) > 0:
		request["tools"] = params.Tools
	case len(cfg.Tools) > 0:
		request["tools"] = interactionsadapter.ToInteractionsTools(cfg.Tools)
	}

	if generation := s.generationConfigFor(cfg); len(generation) > 0 {
		request[keyGenerationConfig] = generation
	}

	if len(cfg.SafetySettings) > 0 {
		safety := make([]map[string]any, 0, len(cfg.SafetySettings))
		for _, setting := range cfg.SafetySettings {
			safety = append(safety, safetySetting(setting))
		}
		request["safety_settings"] = safety
	}

	for key, value := range cfg.Extra {
		if extra, ok := value.(map[string]any); ok && key == keyGenerationConfig {
			maps.Copy(generationConfig(request), extra)
			continue
		}
		request[key] = value
	}

	// Applied last, so an explicit thinking level from the settings or from
	// Extra wins over the low-latency default.
	s.maybeApplyThinkingDefault(request, cfg.Model)

	return request
}

// generationConfigFor builds the generation configuration from the settings,
// warning about each one the Interactions API has no field for.
func (s *Service) generationConfigFor(cfg requestSettings) map[string]any {
	generation := map[string]any{}
	if cfg.Temperature != nil {
		generation["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		generation["top_p"] = *cfg.TopP
	}
	if cfg.MaxTokens != 0 {
		generation["max_output_tokens"] = cfg.MaxTokens
	}
	if cfg.Seed != nil {
		generation["seed"] = *cfg.Seed
	}
	if cfg.TopK != nil {
		s.warnUnsupported("top_k", "the Interactions API has no top_k")
	}
	if cfg.frequencyPenalty != nil {
		s.warnUnsupported("frequency_penalty", "the Interactions API has no frequency_penalty")
	}
	if cfg.presencePenalty != nil {
		s.warnUnsupported("presence_penalty", "the Interactions API has no presence_penalty")
	}

	s.applyThinking(generation, cfg.Thinking)

	if len(cfg.ToolConfig) > 0 {
		generation["tool_choice"] = toolChoice(cfg.ToolConfig)
	}
	return generation
}

// applyThinking puts a thinking configuration into the generation
// configuration. The level and thought summaries apply; a budget is ignored with
// a warning.
func (s *Service) applyThinking(generation map[string]any, thinking *ThinkingConfig) {
	if thinking == nil {
		return
	}
	if thinking.Level != "" {
		generation[keyThinkingLevel] = thinking.Level
	}
	if thinking.IncludeThoughts {
		generation["thinking_summaries"] = "auto"
	}
	if thinking.Budget != nil {
		s.warnUnsupported("thinking_budget", "the Interactions API controls thinking by thinking_level only")
	}
}

// generationConfig is the request's generation configuration, added when it has
// none.
func generationConfig(request map[string]any) map[string]any {
	g, ok := request[keyGenerationConfig].(map[string]any)
	if !ok {
		g = map[string]any{}
		request[keyGenerationConfig] = g
	}
	return g
}

// maybeApplyThinkingDefault sends the model its lowest thinking level when none
// is configured.
//
// It applies the model's thinking default, if it has one. The Gemini 2.5
// models, which take a budget, run at Google's default, since the Interactions
// API takes no budget. With use_provider_thinking_default set, no default is
// sent. The first request for each model logs the setting sent, or that the
// model runs at Google's default.
func (s *Service) maybeApplyThinkingDefault(request map[string]any, model string) {
	if g, ok := request[keyGenerationConfig].(map[string]any); ok {
		if _, set := g[keyThinkingLevel]; set {
			return
		}
	}
	thinking, canThink := gemini.ThinkingDefault(model, true)
	var setting string
	if thinking != nil && thinking.Level != "" && !s.UseProviderThinkingDefault() {
		generationConfig(request)[keyThinkingLevel] = thinking.Level
		setting = fmt.Sprintf(`%s={"%s":%q}`, keyGenerationConfig, keyThinkingLevel, thinking.Level)
	}
	s.ReportThinkingDefault(model, setting, canThink, "Google", "Thinking in the service's Config")
}

// warnUnsupported reports, once per setting, a setting the Interactions API has
// no field for.
func (s *Service) warnUnsupported(name, reason string) {
	s.warnedMu.Lock()
	warned := s.warnedUnsupported[name]
	s.warnedUnsupported[name] = true
	s.warnedMu.Unlock()
	if !warned {
		slog.Warn(fmt.Sprintf("ignoring %s, since %s", name, reason), "service", s.Name())
	}
}

// newRequest marshals the request and builds the POST that sends it.
func (s *Service) newRequest(ctx context.Context, request map[string]any) (*http.Request, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.shaper.Endpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := s.shaper.Authorize(ctx, req); err != nil {
		return nil, err
	}
	return req, nil
}

// statusError is the error a response with a status other than 200 is reported
// with.
func statusError(resp *http.Response) error {
	msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
}

// event is the subset of a streamed Interactions event the service reads.
type event struct {
	EventType string `json:"event_type"` //nolint:tagliatelle // Interactions API field
	Index     int    `json:"index"`
	Step      *struct {
		Type      string          `json:"type"`
		ID        string          `json:"id"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"step"`
	Delta *struct {
		Type      string `json:"type"`
		Text      string `json:"text"`
		Arguments string `json:"arguments"`
		Signature string `json:"signature"`
		Content   *struct {
			Text string `json:"text"`
		} `json:"content"`
	} `json:"delta"`
	Interaction *interaction `json:"interaction"`
	Error       *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// interaction is the subset of an interaction the service reads.
type interaction struct {
	Status string     `json:"status"`
	Model  string     `json:"model"`
	Usage  *usageData `json:"usage"`
	Steps  []struct {
		Type    string `json:"type"`
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	} `json:"steps"`
}

// usageData is an interaction's token accounting.
//
//nolint:tagliatelle // Interactions API fields
type usageData struct {
	TotalInputTokens   int64 `json:"total_input_tokens"`
	TotalOutputTokens  int64 `json:"total_output_tokens"`
	TotalThoughtTokens int64 `json:"total_thought_tokens"`
	TotalToolUseTokens int64 `json:"total_tool_use_tokens"`
	TotalCachedTokens  int64 `json:"total_cached_tokens"`
}

// tokenUsage is the token usage the pipeline carries for an interaction's usage,
// all zero when there is none.
//
// The API counts tool results fed back to the model apart from the input, and
// thinking apart from the output, but they are input and output too.
func (u *usageData) tokenUsage() frames.LLMTokenUsage {
	if u == nil {
		return frames.LLMTokenUsage{}
	}
	reasoning := u.TotalThoughtTokens
	prompt := u.TotalInputTokens + u.TotalToolUseTokens
	completion := u.TotalOutputTokens + reasoning
	return frames.LLMTokenUsage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      prompt + completion,
		CacheReadTokens:  new(u.TotalCachedTokens),
		ReasoningTokens:  new(reasoning),
	}
}

// pendingCall is a function call whose arguments are streaming in.
type pendingCall struct {
	id, name string
	// arguments are the streamed arguments so far, and initial the ones the
	// step started with, used when none stream in.
	arguments string
	initial   json.RawMessage
}

// responseState is what one streamed response has produced so far.
type responseState struct {
	convo *frames.LLMContext
	// Step index to step type, call in progress, and text so far.
	steps         map[int]string
	calls         map[int]*pendingCall
	text          map[int]string
	functionCalls []frames.ToolCall
	// pendingSignature is a thought signature waiting for the step it belongs
	// to, and textBookmarkIndex the text step it was tied to.
	pendingSignature  *string
	textBookmarkIndex *int
	// usage is the token accounting the completed interaction reported.
	usage *usageData
}

func newResponseState(convo *frames.LLMContext) *responseState {
	return &responseState{
		convo: convo,
		steps: map[int]string{},
		calls: map[int]*pendingCall{},
		text:  map[int]string{},
	}
}

// handleEvent acts on one stream event.
func (s *Service) handleEvent(ctx context.Context, ev event, response *responseState, text func(string) error) error {
	switch ev.EventType {
	case "step.start":
		return s.handleStepStart(ctx, ev, response)
	case "step.delta":
		return s.handleDelta(ctx, ev, response, text)
	case "step.stop":
		return s.handleStepStop(ctx, ev, response)
	case "interaction.completed":
		s.handleCompleted(ctx, ev, response)
	case "error":
		message := "Interactions API error"
		if ev.Error != nil && ev.Error.Message != "" {
			message = ev.Error.Message
		}
		s.PushError(ctx, "LLM response error: "+message, nil, false)
	}
	return nil
}

// handleStepStart records a step the model began.
func (s *Service) handleStepStart(ctx context.Context, ev event, response *responseState) error {
	if ev.Step == nil {
		return nil
	}
	// A leading lifecycle event carries no model output, so TTFB ends at the
	// first step.
	s.StopTTFBMetrics(ctx)
	response.steps[ev.Index] = ev.Step.Type
	if ev.Step.Type == interactionsadapter.StepFunctionCall {
		// A turn that only calls tools produces no answer text, so the call
		// itself is what the caller gets and TTFAT ends here rather than going
		// unmeasured.
		s.StopTTFATMetrics(ctx)
		response.calls[ev.Index] = &pendingCall{id: ev.Step.ID, name: ev.Step.Name, initial: ev.Step.Arguments}
	}
	return s.bookmarkPendingSignature(ctx, response, ev.Index)
}

// handleStepStop completes a step: a function call is ready to run, and a text
// step a signature was tied to has the text the signature is stored with.
func (s *Service) handleStepStop(ctx context.Context, ev event, response *responseState) error {
	if call, ok := response.calls[ev.Index]; ok {
		delete(response.calls, ev.Index)
		response.functionCalls = append(response.functionCalls, s.functionCall(ctx, call))
		return nil
	}
	if response.textBookmarkIndex != nil && *response.textBookmarkIndex == ev.Index {
		return s.storeSignature(ctx, response, geminiadapter.BookmarkText, response.text[ev.Index])
	}
	return nil
}

// handleCompleted reads the interaction a response ended with: the model that
// answered, how it ended, and its usage.
func (s *Service) handleCompleted(ctx context.Context, ev event, response *responseState) {
	if ev.Interaction == nil {
		return
	}
	s.SetFullModelName(ev.Interaction.Model)
	status := ev.Interaction.Status
	switch {
	case status != statusCompleted && status != statusRequiresAction:
		slog.WarnContext(ctx, "response incomplete, the interaction ended "+status, "service", s.Name())
	case len(response.functionCalls) == 0 && strings.TrimSpace(joinedText(response)) == "":
		s.reportEmptyResponse(ctx, response)
	}
	response.usage = ev.Interaction.Usage
}

// handleDelta acts on a step's delta: answer text, a function call's arguments,
// a thought summary or a thought signature.
func (s *Service) handleDelta(
	ctx context.Context, ev event, response *responseState, text func(string) error,
) error {
	delta := ev.Delta
	if delta == nil {
		return nil
	}
	switch delta.Type {
	case "text":
		response.text[ev.Index] += delta.Text
		return text(delta.Text)
	case "arguments_delta":
		if call, ok := response.calls[ev.Index]; ok {
			call.arguments += delta.Arguments
		}
	case "thought_summary":
		if delta.Content != nil && delta.Content.Text != "" {
			// The summary arrives whole, so it is bracketed on its own.
			return s.pushThought(ctx, delta.Content.Text)
		}
	case "thought_signature":
		if delta.Signature != "" {
			signature := delta.Signature
			response.pendingSignature = &signature
		}
	}
	return nil
}

// pushThought pushes a thought summary as a whole thought.
func (s *Service) pushThought(ctx context.Context, text string) error {
	for _, f := range []frames.Frame{
		frames.NewLLMThoughtStartFrame(),
		frames.NewLLMThoughtTextFrame(text),
		frames.NewLLMThoughtEndFrame(),
	} {
		if err := s.PushFrame(ctx, f, processor.Downstream); err != nil {
			return err
		}
	}
	return nil
}

// joinedText is all the answer text the response produced.
func joinedText(response *responseState) string {
	var b strings.Builder
	for _, i := range slices.Sorted(maps.Keys(response.text)) {
		b.WriteString(response.text[i])
	}
	return b.String()
}

// reportEmptyResponse reports a completed response that produced no text and no
// function call.
//
// Nothing reaches the TTS for such a response, so the bot stays silent until the
// user speaks again. The warning lists the steps the response produced instead,
// and the non-fatal error lets the application react.
func (s *Service) reportEmptyResponse(ctx context.Context, response *responseState) {
	steps := make([]string, 0, len(response.steps))
	for _, i := range slices.Sorted(maps.Keys(response.steps)) {
		steps = append(steps, response.steps[i])
	}
	slog.WarnContext(ctx, "response completed with no text or function call",
		"service", s.Name(), "steps", steps)
	s.PushError(ctx, "LLM response completed with no text or function call", nil, false)
}

// bookmarkPendingSignature ties a signature received in a thought step to the
// step that follows it.
//
// A function call is named by its id right away. Text is named by what it says,
// so its signature is stored when the text step stops.
func (s *Service) bookmarkPendingSignature(ctx context.Context, response *responseState, index int) error {
	if response.pendingSignature == nil {
		return nil
	}
	switch response.steps[index] {
	case interactionsadapter.StepFunctionCall:
		return s.storeSignature(ctx, response, geminiadapter.BookmarkFunctionCall, response.calls[index].id)
	case interactionsadapter.StepModelOutput:
		response.textBookmarkIndex = &index
	}
	return nil
}

// storeSignature appends the pending thought signature to the conversation, in
// the form the Gemini service stores it, bookmarked by what it was received
// with.
func (s *Service) storeSignature(
	ctx context.Context, response *responseState, bookmarkKey, bookmarkValue string,
) error {
	signature := response.pendingSignature
	response.pendingSignature = nil
	response.textBookmarkIndex = nil
	if signature == nil || bookmarkValue == "" {
		return nil
	}
	bookmark := map[string]any{bookmarkKey: bookmarkValue}
	msg := geminiadapter.NewThoughtSignature(decodeSignature(*signature), bookmark)
	return s.PushFrame(ctx, frames.NewLLMMessagesAppendFrame([]frames.Message{msg}), processor.Downstream)
}

// functionCall is the call a finished function-call step made. Arguments that
// streamed in replace the ones the step started with; arguments that cannot be
// read leave the starting ones in place.
func (s *Service) functionCall(ctx context.Context, call *pendingCall) frames.ToolCall {
	arguments := call.initial
	if len(bytes.TrimSpace(arguments)) == 0 || string(bytes.TrimSpace(arguments)) == "null" {
		arguments = json.RawMessage("{}")
	}
	if call.arguments != "" {
		if json.Valid([]byte(call.arguments)) {
			arguments = json.RawMessage(call.arguments)
		} else {
			slog.WarnContext(ctx, "Failed to parse function call arguments: "+call.arguments, "service", s.Name())
		}
	}
	slog.DebugContext(ctx, "Function call: "+call.name+":"+call.id)
	return frames.ToolCall{ID: call.id, Name: call.name, Args: arguments}
}

// decodeSignature is a thought signature as the Gemini service stores it: raw
// bytes, or the text as it arrived when it does not decode.
func decodeSignature(signature string) any {
	raw, err := base64.StdEncoding.DecodeString(signature)
	if err != nil {
		return signature
	}
	return raw
}

// toolChoice is the Interactions API's tool_choice for a generateContent tool
// configuration.
func toolChoice(toolConfig map[string]any) any {
	config, ok := toolConfig["function_calling_config"].(map[string]any)
	if !ok {
		return toolConfig
	}
	mode := "auto"
	if m, ok := config["mode"].(string); ok && m != "" {
		mode = strings.ToLower(m)
	}
	if names := stringList(config["allowed_function_names"]); len(names) > 0 {
		return map[string]any{"allowed_tools": map[string]any{"mode": mode, "tools": names}}
	}
	return mode
}

// stringList reads a list of strings, whether typed or decoded from JSON.
func stringList(v any) []string {
	switch list := v.(type) {
	case []string:
		return list
	case []any:
		out := make([]string, 0, len(list))
		for _, item := range list {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

// safetySetting is the Interactions API's form of a generateContent safety
// setting: HARM_CATEGORY_HARASSMENT becomes harassment, BLOCK_ONLY_HIGH becomes
// block_only_high, and so on.
func safetySetting(setting SafetySetting) map[string]any {
	result := map[string]any{
		"type":      strings.TrimPrefix(strings.ToLower(setting.Category), "harm_category_"),
		"threshold": strings.ToLower(setting.Threshold),
	}
	if setting.Method != "" {
		result["method"] = strings.ToLower(setting.Method)
	}
	return result
}
