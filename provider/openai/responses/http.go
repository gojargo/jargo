package responses

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/adapter/openai"
	responsesadapter "github.com/gojargo/jargo/adapter/responses"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/internal/apierror"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
)

// HTTPService streams a Responses turn over one request.
type HTTPService struct {
	*llm.Base
	cfg     Config
	http    *http.Client
	variant HTTPVariant
	// adapter converts each conversation into a request. It is held rather than
	// built per request, since it carries the tools the service implements
	// itself.
	adapter responsesadapter.Adapter
}

// HTTPVariant adapts the HTTP service to a server that speaks the Responses
// protocol with departures from OpenAI's. Every hook is optional, and one left
// nil keeps the OpenAI behavior.
type HTTPVariant struct {
	// Name labels the service in logs, metrics and traces.
	Name string
	// ThinkingDefault replaces the lowest-effort default OpenAI models get. It
	// is called on each request body, as JSON fields, after Extra, and decides
	// for itself whether the body already configures reasoning. report is the
	// service, for reporting what it sent.
	ThinkingDefault func(body map[string]any, model string, report ThinkingReporter)
	// Request edits each request body last, as JSON fields.
	Request func(body map[string]any)
	// Events returns a filter for one response's event stream. It is given each
	// event as sent and returns the events to handle in its place, which may be
	// none, or ones it held back earlier.
	Events func() func(data []byte) [][]byte
	// Usage adjusts the token usage of a response, given the usage object as
	// sent, for a server that reports fields OpenAI does not.
	Usage func(raw json.RawMessage, tokens *frames.LLMTokenUsage)
}

// NewHTTPLLM builds a Responses LLM service that streams over a request per
// turn. Prefer NewLLM: the WebSocket service holds one connection open and
// sends only what is new each turn.
func NewHTTPLLM(cfg Config) *HTTPService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return NewHTTPVariantLLM(cfg, HTTPVariant{Name: "OpenAIResponsesHTTPLLM"})
}

// NewHTTPVariantLLM builds a Responses HTTP service for a server that departs
// from OpenAI's in the ways v describes. Unlike NewHTTPLLM it leaves an empty
// Model empty, so a server that can choose the model itself is not sent
// OpenAI's default.
func NewHTTPVariantLLM(cfg Config, v HTTPVariant) *HTTPService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	s := &HTTPService{cfg: cfg, http: &http.Client{}, variant: v}
	s.Base = llm.New(v.Name, s)
	s.Base.SetModel(cfg.Model)
	s.SetSystemInstruction(cfg.Instructions)
	warnIfReasoningUnsupported(v.Name, cfg)
	return s
}

// Generate streams a completion, emitting each text delta.
func (s *HTTPService) Generate(ctx context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	return s.run(ctx, convo, textSink{emit: emit}, false)
}

// GenerateWithTools streams a completion, reporting text and any tool calls the
// model requests. It implements llm.ToolGenerator.
func (s *HTTPService) GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink llm.Sink) error {
	return s.run(ctx, convo, sink, true)
}

// RunInference answers the conversation once, off to the side of the pipeline:
// no streaming, no frames, just the text. It implements llm.Inferencer.
func (s *HTTPService) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, error) {
	opts.ResponseSchema = s.CheckResponseSchema(ctx, opts.ResponseSchema)
	if opts.SystemInstruction == "" {
		opts.SystemInstruction = s.SystemInstruction()
	}
	return runInference(ctx, s.cfg, &s.adapter, s.http, convo, opts, s.encode)
}

// LLMAdapter implements llm.AdapterHolder.
func (s *HTTPService) LLMAdapter() llm.BuiltinToolHolder { return &s.adapter }

// encode renders a request for the wire: the thinking default, Extra over the
// modeled fields, then the variant's edits.
func (s *HTTPService) encode(req request) ([]byte, error) {
	if s.variant.ThinkingDefault == nil && s.variant.Request == nil {
		return s.cfg.encode(req, s)
	}
	if s.variant.ThinkingDefault == nil {
		s.cfg.applyThinkingDefault(&req, s)
	}
	raw, err := encodeBody(req, s.cfg.Extra)
	if err != nil {
		return nil, err
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	if s.variant.ThinkingDefault != nil {
		s.variant.ThinkingDefault(body, s.cfg.Model, s)
	}
	if s.variant.Request != nil {
		s.variant.Request(body)
	}
	return json.Marshal(body)
}

// SupportsResponseSchema implements llm.ResponseSchemaSupporter.
func (s *HTTPService) SupportsResponseSchema() bool { return true }

// ModelSupportsResponseSchema implements llm.ResponseSchemaSupporter. OpenAI
// models before gpt-4o-mini and gpt-4o-2024-08-06 cannot enforce a schema.
func (s *HTTPService) ModelSupportsResponseSchema(model string) bool {
	return openai.ModelSupportsResponseSchema(model)
}

// run issues one turn and feeds its event stream to the state machine.
func (s *HTTPService) run(ctx context.Context, convo *frames.LLMContext, sink llm.Sink, withTools bool) error {
	reqBody, err := s.cfg.newRequest(
		&s.adapter, convo, adapter.Options{SystemInstruction: s.SystemInstruction()}, withTools,
	)
	if err != nil {
		return err
	}
	body, err := s.encode(reqBody)
	if err != nil {
		return err
	}
	s.StartTTFBMetrics()
	resp, err := s.cfg.post(ctx, s.http, body)
	if err != nil {
		return llm.AsCompletionTimeout(ctx, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return apierror.FromResponse(resp, errStatus)
	}

	state := newStreamState(sink, "Responses API error")
	scanErr := s.consume(ctx, resp.Body, state)
	// A response cut short or failed is still billed for its tokens.
	s.reportUsage(ctx, state)
	if scanErr != nil && !errors.Is(scanErr, errStreamDone) {
		return scanErr
	}
	return nil
}

// consume feeds a response's event stream to the state machine, through the
// variant's filter when it has one.
func (s *HTTPService) consume(ctx context.Context, r io.Reader, state *streamState) error {
	handle := func(data []byte) error {
		ev, ok := decodeEvent(data)
		if !ok {
			return nil // skip a malformed event rather than ending the turn
		}
		if carriesModelOutput(ev) {
			s.StopTTFBMetrics(ctx)
		}
		done, err := state.handle(ev)
		if err != nil {
			return err
		}
		if done {
			return errStreamDone
		}
		return nil
	}
	if s.variant.Events == nil {
		return llm.ScanSSE(r, func(data string) error { return handle([]byte(data)) })
	}
	filter := s.variant.Events()
	return llm.ScanSSE(r, func(data string) error {
		for _, ev := range filter([]byte(data)) {
			if err := handle(ev); err != nil {
				return err
			}
		}
		return nil
	})
}

// reportUsage forwards the turn's token accounting, when usage metrics are on.
func (s *HTTPService) reportUsage(ctx context.Context, state *streamState) {
	if state.usage == nil || !s.UsageMetricsEnabled() {
		return
	}
	tokens := state.usage.tokenUsage()
	if s.variant.Usage != nil {
		s.variant.Usage(state.usage.raw, &tokens)
	}
	_ = s.PushTokenUsage(ctx, tokens)
}

// ClassifyError implements processor.ErrorClassifier: a content filter's
// refusal is content the provider refused, not the permanent invalid request
// its status says.
func (s *HTTPService) ClassifyError(err error) errs.Category { return apierror.Classify(err) }
