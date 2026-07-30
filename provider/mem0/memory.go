package mem0

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Service is the memory frame processor.
type Service struct {
	*processor.Base
	cfg  Config
	http *http.Client
	host string

	// Store de-duplication state, touched only on the process goroutine.
	lastUser      string
	lastAssistant string
	// lastQuery is the query the current recall was retrieved for, so a context
	// replayed for the same user message does not search again. A tool-calling
	// turn replays the context once per round trip.
	lastQuery string
}

// NewMemory builds a memory service from cfg.
func NewMemory(cfg Config) *Service {
	if cfg.SearchLimit <= 0 {
		cfg.SearchLimit = defaultSearchLimit
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultTimeout
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: cfg.Timeout}
	}
	s := &Service{
		cfg:  cfg,
		http: hc,
		host: strings.TrimRight(cfg.Host, "/"),
	}
	s.Base = processor.New("Mem0Memory", s)
	return s
}

// ProcessFrame retrieves and stores memory around the LLM context, forwarding
// every frame untouched.
func (s *Service) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMContextFrame:
		if dir == processor.Downstream {
			s.onContext(ctx, fr.Context)
		}
	case *frames.StartFrame:
		if s.cfg.Prewarm {
			//nolint:gosec // detached like store: the warm-up must outlive a turn that gets interrupted
			go s.prewarm()
		}
	}
	return s.PushFrame(ctx, f, dir)
}

// onContext injects memories relevant to the latest user message and persists
// the new turns. Retrieval is synchronous (the LLM needs the memories now) and
// runs under the turn's context, so an interruption cancels it; storage runs in
// the background under a detached context so memories outlive a barge-in.
func (s *Service) onContext(ctx context.Context, convo *frames.LLMContext) {
	msgs := convo.Messages()
	if query := lastText(msgs, frames.RoleUser); query != "" && query != s.lastQuery {
		s.lastQuery = query
		sctx, cancel := s.searchContext(ctx)
		mems, err := s.search(sctx, query)
		cancel()
		if err != nil && ctx.Err() == nil {
			slog.Warn("mem0 search failed", "processor", s.Name(), "error", err)
		}
		convo.SetRecall(formatMemories(mems))
	}
	s.store(msgs)
}

// searchContext bounds retrieval when SearchTimeout is set, so a slow mem0
// delays a reply by no more than that.
func (s *Service) searchContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if s.cfg.SearchTimeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, s.cfg.SearchTimeout)
}

// prewarm issues one throwaway search at session start so the first real one
// does not pay for whatever the path warms up. Measured over a session on a
// self-hosted server: 1.52s for the first search, 0.58s for the second, 0.19s
// for the third.
func (s *Service) prewarm() {
	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Timeout)
	defer cancel()
	if _, err := s.search(ctx, prewarmQuery); err != nil {
		slog.Debug("mem0 prewarm failed", "processor", s.Name(), "error", err)
	}
}

// store sends the latest user and assistant turns to mem0 once each. The
// de-duplication state is updated here on the process goroutine; only the
// network write is offloaded.
func (s *Service) store(msgs []frames.Message) {
	user := lastText(msgs, frames.RoleUser)
	asst := lastText(msgs, frames.RoleAssistant)

	var batch []apiMessage
	if asst != "" && asst != s.lastAssistant {
		batch = append(batch, apiMessage{Role: "assistant", Content: asst})
		s.lastAssistant = asst
	}
	if user != "" && user != s.lastUser {
		batch = append(batch, apiMessage{Role: "user", Content: user})
		s.lastUser = user
	}
	if len(batch) > 0 {
		go s.add(batch)
	}
}

// searchFilters scopes a search to the configured entities. Search takes them
// nested under "filters" and rejects the top-level fields that add accepts.
func (s *Service) searchFilters() map[string]any {
	filters := make(map[string]any, 3)
	if s.cfg.UserID != "" {
		filters["user_id"] = s.cfg.UserID
	}
	if s.cfg.AgentID != "" {
		filters["agent_id"] = s.cfg.AgentID
	}
	if s.cfg.RunID != "" {
		filters["run_id"] = s.cfg.RunID
	}
	return filters
}

// search returns the memories relevant to query.
func (s *Service) search(ctx context.Context, query string) ([]memory, error) {
	body := searchRequest{
		Query:     query,
		Filters:   s.searchFilters(),
		TopK:      s.cfg.SearchLimit,
		Threshold: s.cfg.SearchThreshold,
	}
	var resp searchResponse
	if err := s.post(ctx, "/search", body, &resp); err != nil {
		return nil, err
	}
	return resp.Results, nil
}

// add persists a batch of turns to mem0 under a detached, time-bounded context.
func (s *Service) add(batch []apiMessage) {
	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Timeout)
	defer cancel()
	body := addRequest{
		Messages: batch,
		UserID:   s.cfg.UserID,
		AgentID:  s.cfg.AgentID,
		RunID:    s.cfg.RunID,
		Metadata: map[string]any{"platform": "jargo"},
	}
	if err := s.post(ctx, "/memories", body, nil); err != nil {
		slog.Warn("mem0 store failed", "processor", s.Name(), "error", err)
	}
}

// post sends in as JSON to path and, when out is non-nil, decodes the response
// into it.
func (s *Service) post(ctx context.Context, path string, in, out any) error {
	buf, err := json.Marshal(in)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.host+path, bytes.NewReader(buf))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if s.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Token "+s.cfg.APIKey)
	}
	res, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = res.Body.Close() }()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("%w: %s %s: %s", errStatus, path, res.Status, strings.TrimSpace(string(b)))
	}
	if out != nil {
		return json.NewDecoder(res.Body).Decode(out)
	}
	return nil
}

// formatMemories renders memories as the recalled-context block folded into the
// system prompt, or "" when there are none.
func formatMemories(mems []memory) string {
	lines := make([]string, 0, len(mems))
	for _, m := range mems {
		if m.Memory != "" {
			lines = append(lines, strconv.Itoa(len(lines)+1)+". "+m.Memory)
		}
	}
	if len(lines) == 0 {
		return ""
	}
	return recallHeader + "\n" + strings.Join(lines, "\n")
}

// lastText returns the text of the most recent plain message with role r,
// skipping tool-call and tool-result turns, or "".
func lastText(msgs []frames.Message, r frames.Role) string {
	for _, m := range slices.Backward(msgs) {
		if m.Role == r && m.Text != "" && len(m.ToolCalls) == 0 && len(m.ToolResults) == 0 {
			return m.Text
		}
	}
	return ""
}

// apiMessage is one message in a mem0 add request.
type apiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type addRequest struct {
	Messages []apiMessage   `json:"messages"`
	UserID   string         `json:"user_id,omitempty"`
	AgentID  string         `json:"agent_id,omitempty"`
	RunID    string         `json:"run_id,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type searchRequest struct {
	Query     string         `json:"query"`
	Filters   map[string]any `json:"filters,omitempty"`
	TopK      int            `json:"top_k,omitempty"`
	Threshold float64        `json:"threshold,omitempty"`
}

// memory is one retrieved memory; the server returns more fields, but the text
// is all the prompt needs.
type memory struct {
	ID     string  `json:"id"`
	Memory string  `json:"memory"`
	Score  float64 `json:"score"`
}

type searchResponse struct {
	Results []memory `json:"results"`
}
