package responses

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// HTTPService streams a Responses turn over one request.
type HTTPService struct {
	*llm.Base
	cfg  Config
	http *http.Client
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
	s := &HTTPService{cfg: cfg, http: &http.Client{}}
	s.Base = llm.New("OpenAIResponsesHTTPLLM", s)
	s.Base.SetModel(cfg.Model)
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

// run issues one turn and feeds its event stream to the state machine.
func (s *HTTPService) run(ctx context.Context, convo *frames.LLMContext, sink llm.Sink, withTools bool) error {
	body, err := encodeBody(s.cfg.newRequest(convo, withTools), s.cfg.Extra)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	s.StartTTFBMetrics()
	resp, err := s.http.Do(req)
	s.StopTTFBMetrics()
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}

	state := newStreamState(sink)
	scanErr := llm.ScanSSE(resp.Body, func(data string) error {
		ev, ok := decodeEvent([]byte(data))
		if !ok {
			return nil // skip a malformed event rather than ending the turn
		}
		done, err := state.handle(ev)
		if err != nil {
			return err
		}
		if done {
			return errStreamDone
		}
		return nil
	})
	if scanErr != nil && !errors.Is(scanErr, errStreamDone) {
		return scanErr
	}
	s.reportUsage(ctx, state)
	return nil
}

// reportUsage forwards the turn's token accounting, when usage metrics are on.
func (s *HTTPService) reportUsage(ctx context.Context, state *streamState) {
	if state.usage == nil || !s.UsageMetricsEnabled() {
		return
	}
	_ = s.PushTokenUsage(ctx, state.usage.tokenUsage())
}
