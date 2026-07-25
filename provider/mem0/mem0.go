// Package mem0 adds long-term memory to a jargo voice agent, backed by a mem0
// server (https://github.com/mem0ai/mem0).
//
// It is a frame processor placed between the user aggregator and the LLM. On
// each turn it searches mem0 for memories relevant to the user's latest message
// and folds them into the context's system prompt (see frames.LLMContext.SetRecall),
// then stores the new user and assistant turns back to mem0 in the background so
// retrieval never blocks on writes.
//
//	mem := mem0.NewMemory(mem0.Config{Host: "http://localhost:8000", UserID: caller})
//	pipeline.New(input, stt, agg.User(), mem, llm, tts, output, agg.Assistant())
//
// Point Host at a mem0 REST server — self-hosted, or the managed API with an
// APIKey — and scope memories to a caller with UserID. Memory is best-effort:
// search and store failures are logged and never break the conversation.
package mem0

import (
	"errors"
	"net/http"
	"time"

	"github.com/gojargo/jargo/internal/validate"
)

// errStatus is returned when mem0 answers with a non-2xx status.
var errStatus = errors.New("mem0 returned an error status")

const (
	defaultSearchLimit = 10
	defaultTimeout     = 10 * time.Second
	// recallHeader frames the retrieved memories injected into the system prompt.
	recallHeader = "Based on previous conversations, you recall the following about the user:"
)

// Config configures the memory service.
type Config struct {
	// Host is the base URL of the mem0 REST server, e.g. http://localhost:8000.
	// Required.
	Host string `validate:"required"`
	// APIKey is an optional bearer token sent as "Authorization: Token <key>"
	// (the managed mem0 API, or a secured self-hosted server).
	APIKey string
	// UserID scopes memories to a caller. Recommended; without it memories are
	// not partitioned per user.
	UserID string
	// AgentID and RunID optionally scope memories to an agent or a session.
	AgentID string
	RunID   string
	// SearchLimit caps how many memories are retrieved per turn; 0 uses a default.
	SearchLimit int
	// SearchThreshold is the minimum relevance score for a memory to be used; 0
	// leaves the cutoff to the server.
	SearchThreshold float64
	// Timeout bounds a single mem0 request; 0 uses a default.
	Timeout time.Duration
	// HTTPClient overrides the HTTP client; nil uses one with Timeout.
	HTTPClient *http.Client
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }
