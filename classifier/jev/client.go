// Package jev is a classifier backed by Jev, TypeSafe's hosted classification
// model.
//
// One Client holds one HTTP/2 connection pool, adds the auth header, retries
// when Jev is busy, and counts the tokens every request used. Several
// Classifiers can share one. A Classifier turns each question into a request
// and each reply into a result, through a Client.
package jev

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/classifier/internal/ordered"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/utils/network"
)

const (
	// DefaultBaseURL is where the API is served.
	DefaultBaseURL = "https://api.typesafe.ai"
	// DefaultModel is the Jev model to ask. Pinned so thresholds tuned against
	// it hold.
	DefaultModel = "jev-1.13.0"
	// DefaultTimeout is how long to wait for a reply before giving up.
	DefaultTimeout = 10 * time.Second
	// defaultMaxRetries is how many times a request Jev refused because it was
	// busy is retried.
	defaultMaxRetries = 3

	// tooManyRequests is the status Jev answers with when the caller is rate
	// limited.
	tooManyRequests = 429
	// overloaded is the status Jev answers with when it is temporarily
	// overloaded.
	overloaded = 529
	// keepaliveExpiry is how long an idle connection is kept open. Questions
	// come with gaps between them, and a connection that has closed in the
	// meantime costs a new TLS handshake on the next one. Jev closes idle
	// connections after about five minutes, so this stays under that.
	keepaliveExpiry = 240 * time.Second
)

// ClientConfig configures a Client.
type ClientConfig struct {
	// APIKey is the Jev API key. Required.
	APIKey string `validate:"required"`
	// BaseURL is where the API is served; empty uses DefaultBaseURL.
	BaseURL string
	// Model is the Jev model to ask; empty uses DefaultModel, pinned so
	// thresholds tuned against it keep holding. "jev-latest" follows TypeSafe's
	// newest release.
	Model string
	// Timeout is how long to wait for a reply before giving up; zero uses
	// DefaultTimeout.
	Timeout time.Duration `validate:"min=0"`
	// MaxRetries is how many times to retry a request Jev refused because it
	// was busy; nil uses 3.
	MaxRetries *int `validate:"omitempty,min=0"`
	// HTTPClient sends the requests; nil builds one that keeps HTTP/2
	// connections open between questions.
	HTTPClient *http.Client
}

// Usage is the tokens Jev used: for one request, or over every request a client
// sent.
type Usage struct {
	// InputTokens is the tokens sent.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the tokens received.
	OutputTokens int `json:"output_tokens"`
}

// Question is one question in Jev's own format.
type Question struct {
	// Type is "noul", "choice" or "score".
	Type string
	// Instructions is what is being asked.
	Instructions any
	// Criteria is what counts as each answer, or nil when the question leaves
	// it open.
	Criteria any
}

// MarshalJSON implements json.Marshaler, leaving criteria out when there are
// none.
func (q Question) MarshalJSON() ([]byte, error) {
	obj := ordered.Object{{Key: "type", Value: q.Type}, {Key: "instructions", Value: q.Instructions}}
	if q.Criteria != nil {
		obj.Set("criteria", q.Criteria)
	}
	return obj.MarshalJSON()
}

// NamedQuestion is a question in Jev's format with the name its answer comes
// back under.
type NamedQuestion struct {
	Name     string
	Question Question
}

// Client is an HTTP client for Jev's systemone endpoint.
//
// It holds one HTTP/2 connection pool, so many small requests share a
// connection and can be in flight at the same time. The connection is kept open
// between questions, so a gap between them does not cost a new TLS handshake.
// It retries with backoff when Jev answers 429 or 529, and counts the tokens
// every request used in Usage.
//
// Connect opens the connection ahead of the first question, and Close releases
// it.
type Client struct {
	apiKey     string
	baseURL    string
	model      string
	timeout    time.Duration
	maxRetries int
	http       *http.Client
	// sleep waits between retries; tests replace it.
	sleep func(ctx context.Context, d time.Duration) error

	connectMu sync.Mutex
	connected bool

	mu     sync.Mutex
	usage  Usage
	closed bool
}

// NewClient builds a Jev client.
func NewClient(cfg ClientConfig) (*Client, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, err
	}
	c := &Client{
		apiKey:     cfg.APIKey,
		baseURL:    strings.TrimRight(cfg.BaseURL, "/"),
		model:      cfg.Model,
		timeout:    cfg.Timeout,
		maxRetries: defaultMaxRetries,
		http:       cfg.HTTPClient,
		sleep:      sleep,
	}
	if c.baseURL == "" {
		c.baseURL = DefaultBaseURL
	}
	if c.model == "" {
		c.model = DefaultModel
	}
	if c.timeout == 0 {
		c.timeout = DefaultTimeout
	}
	if cfg.MaxRetries != nil {
		c.maxRetries = *cfg.MaxRetries
	}
	if c.http == nil {
		transport, _ := http.DefaultTransport.(*http.Transport)
		transport = transport.Clone()
		transport.ForceAttemptHTTP2 = true
		transport.IdleConnTimeout = keepaliveExpiry
		c.http = &http.Client{Transport: transport, Timeout: c.timeout}
	}
	return c, nil
}

// Model is the Jev model the questions go to.
func (c *Client) Model() string { return c.model }

// BaseURL is where the questions are sent.
func (c *Client) BaseURL() string { return c.baseURL }

// Timeout is how long the client waits for a reply.
func (c *Client) Timeout() time.Duration { return c.timeout }

// Usage is the tokens used so far, over every request.
func (c *Client) Usage() Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

// Connect opens the connection to Jev.
//
// It lists the models, the cheapest request there is, so the first question
// finds the connection already open. It connects once: a client shared by
// several classifiers is asked by each of them, and only the first call sends
// anything.
func (c *Client) Connect(ctx context.Context) error {
	c.connectMu.Lock()
	defer c.connectMu.Unlock()
	if c.connected {
		return nil
	}
	resp, err := c.do(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return fmt.Errorf("%w: could not connect to Jev: %w", classifier.ErrClassifier, err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%w: Jev refused the connection: HTTP %d", classifier.ErrClassifier, resp.StatusCode)
	}
	c.connected = true
	return nil
}

// Close closes the connection pool.
func (c *Client) Close() {
	c.connectMu.Lock()
	c.connected = false
	c.connectMu.Unlock()
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	c.http.CloseIdleConnections()
}

// Ask sends questions about one state and returns Jev's answers, by the same
// names and in Jev's own format, and the tokens the request used.
//
// It fails with an error wrapping classifier.ErrClassifier when Jev rejected
// the request, kept refusing it because it was busy, could not be reached, or
// left a question unanswered.
func (c *Client) Ask(
	ctx context.Context, state any, questions []NamedQuestion,
) (map[string]map[string]any, Usage, error) {
	if c.isClosed() {
		return nil, Usage{}, fmt.Errorf("%w: Jev request failed: the client is closed", classifier.ErrClassifier)
	}
	qs := ordered.Object{}
	for _, q := range questions {
		qs.Set(q.Name, q.Question)
	}
	body, err := ordered.Marshal(ordered.Object{
		{Key: "model", Value: c.model}, {Key: "state", Value: state}, {Key: "questions", Value: qs},
	})
	if err != nil {
		return nil, Usage{}, fmt.Errorf("%w: cannot write the request: %w", classifier.ErrClassifier, err)
	}

	for attempt := 1; ; attempt++ {
		data, status, err := c.post(ctx, body)
		if err != nil {
			return nil, Usage{}, fmt.Errorf("%w: Jev request failed: %w", classifier.ErrClassifier, err)
		}
		if status == tooManyRequests || status == overloaded {
			if attempt > c.maxRetries {
				return nil, Usage{}, fmt.Errorf("%w: Jev is busy (HTTP %d) after %d attempts",
					classifier.ErrClassifier, status, attempt)
			}
			wait := network.ExponentialBackoffTime(attempt, 250*time.Millisecond, 2*time.Second, 0.25)
			slog.Debug("Jev busy, retrying", "status", status, "wait", wait)
			if err := c.sleep(ctx, wait); err != nil {
				return nil, Usage{}, fmt.Errorf("%w: Jev request failed: %w", classifier.ErrClassifier, err)
			}
			continue
		}
		if status != http.StatusOK {
			return nil, Usage{}, fmt.Errorf("%w: Jev rejected the request: HTTP %d", classifier.ErrClassifier, status)
		}
		return c.read(data, questions)
	}
}

// read turns Jev's reply into answers, counting the tokens it used.
func (c *Client) read(data []byte, questions []NamedQuestion) (map[string]map[string]any, Usage, error) {
	var reply any
	if err := json.Unmarshal(data, &reply); err != nil {
		return nil, Usage{}, fmt.Errorf("%w: Jev reply is not valid JSON: %w", classifier.ErrClassifier, err)
	}
	obj, ok := reply.(map[string]any)
	if !ok {
		return nil, Usage{}, fmt.Errorf("%w: Jev reply is not a JSON object", classifier.ErrClassifier)
	}
	var usage Usage
	if u, isObject := obj["usage"].(map[string]any); isObject {
		usage.InputTokens = intOf(u["input_tokens"])
		usage.OutputTokens = intOf(u["output_tokens"])
	}
	c.mu.Lock()
	c.usage.InputTokens += usage.InputTokens
	c.usage.OutputTokens += usage.OutputTokens
	c.mu.Unlock()

	answers, ok := obj["answers"].(map[string]any)
	if !ok {
		return nil, Usage{}, fmt.Errorf("%w: Jev reply has no answers", classifier.ErrClassifier)
	}
	out := make(map[string]map[string]any, len(questions))
	var missing []string
	for _, q := range questions {
		a, ok := answers[q.Name].(map[string]any)
		if !ok {
			missing = append(missing, q.Name)
			continue
		}
		out[q.Name] = a
	}
	if len(missing) > 0 {
		return nil, Usage{}, fmt.Errorf("%w: Jev reply has no answer for %s",
			classifier.ErrClassifier, strings.Join(missing, ", "))
	}
	return out, usage, nil
}

// post sends a question and returns the reply's body and status.
func (c *Client) post(ctx context.Context, body []byte) ([]byte, int, error) {
	resp, err := c.do(ctx, http.MethodPost, "/v1/systemone", body)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		return nil, 0, err
	}
	return buf.Bytes(), resp.StatusCode, nil
}

// do sends one request with the auth header.
func (c *Client) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return c.http.Do(req)
}

func (c *Client) isClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// intOf reads a token count, 0 for anything that is not a number.
func intOf(v any) int {
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return int(f)
}

// sleep waits for d, or until ctx is done.
func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
