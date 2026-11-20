// Package decisions is a classifier backed by Google's Decisions API.
//
// One Client holds one HTTP/2 connection pool, bounds every request in time,
// and counts the tokens every request used. Several Classifiers can share one.
// A Classifier turns each question into a request and each reply into a
// result, through a Client.
package decisions

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/internal/ordered"
	"github.com/gojargo/jargo/internal/validate"
)

const (
	// DefaultBaseURL is where the API is served.
	DefaultBaseURL = "https://generativelanguage.googleapis.com"
	// DefaultModel is the decision model to ask.
	DefaultModel = "gemini-decision-1"
	// DefaultTimeout is how long to wait for an answer, retries included,
	// before giving up.
	DefaultTimeout = 10 * time.Second
	// defaultMaxRetries is how many times a request that failed for a reason
	// that may pass is retried.
	defaultMaxRetries = 3

	// apiVersion is the API version the Decisions API is served on, the only
	// one.
	apiVersion = "v1beta"
	// keepaliveExpiry is how long an idle connection is kept open. Questions
	// come with gaps between them, and a connection that has closed in the
	// meantime costs a new TLS handshake on the next one.
	keepaliveExpiry = 240 * time.Second
	// initialDelay is the wait before the first retry, doubled for each one
	// after it.
	initialDelay = 250 * time.Millisecond
	// maxDelay is the longest wait between two attempts.
	maxDelay = 2 * time.Second
)

// The keys of a question and of a content block.
const (
	keyType     = "type"
	keyData     = "data"
	keyMimeType = "mime_type"
)

// errTimeout is the cause of a request's context ending at the client's
// timeout.
var errTimeout = errors.New("decisions: timeout")

// ClientConfig configures a Client.
type ClientConfig struct {
	// APIKey is the Gemini API key. Required.
	APIKey string `validate:"required"`
	// BaseURL is where the API is served; empty uses DefaultBaseURL.
	BaseURL string
	// Model is the decision model to ask; empty uses DefaultModel.
	Model string
	// Timeout is how long to wait for an answer, retries included, before
	// giving up; zero uses DefaultTimeout.
	Timeout time.Duration `validate:"min=0"`
	// MaxRetries is how many times to retry a request that failed for a reason
	// that may pass, such as Google being busy; nil uses 3.
	MaxRetries *int `validate:"omitempty,min=0"`
	// HTTPClient sends the requests; nil builds one that keeps HTTP/2
	// connections open between questions.
	HTTPClient *http.Client
}

// Usage is the tokens the Decisions API used: for one request, or over every
// request a client sent.
type Usage struct {
	// InputTokens is the tokens sent.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the tokens received.
	OutputTokens int `json:"output_tokens"`
}

// Question is one question in Google's own format.
type Question struct {
	// Type is "noul", "choice" or "score".
	Type string
	// Instructions is what is being asked, as text.
	Instructions string
	// Criteria is what counts as each answer, or nil when the question leaves
	// it open.
	Criteria any
}

// MarshalJSON implements json.Marshaler, leaving criteria out when there are
// none.
func (q Question) MarshalJSON() ([]byte, error) {
	obj := ordered.Object{{Key: keyType, Value: q.Type}, {Key: "instructions", Value: q.Instructions}}
	if q.Criteria != nil {
		obj.Set("criteria", q.Criteria)
	}
	return obj.MarshalJSON()
}

// NamedQuestion is a question in Google's format with the name its answer
// comes back under.
type NamedQuestion struct {
	Name     string
	Question Question
}

// Client is an HTTP client for Google's Decisions API.
//
// It holds one HTTP/2 connection pool, so many small requests share a
// connection and can be in flight at the same time. The connection is kept open
// between questions, so a gap between them does not cost a new TLS handshake.
// It retries with backoff when Google is busy or can't be reached, within one
// time bound for the whole request, and counts the tokens every request used
// in Usage. Errors carry the reason Google gave.
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

// NewClient builds a Decisions API client.
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

// Model is the decision model the questions go to.
func (c *Client) Model() string { return c.model }

// BaseURL is where the questions are sent.
func (c *Client) BaseURL() string { return c.baseURL }

// Timeout is how long the client waits for an answer, retries included.
func (c *Client) Timeout() time.Duration { return c.timeout }

// Usage is the tokens used so far, over every request.
func (c *Client) Usage() Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

// Connect opens the connection to Google.
//
// It looks up the model, a cheap request that also checks the key and whether
// it has access to the model. It connects once: a client shared by several
// classifiers is asked by each of them, and only the first call sends
// anything.
func (c *Client) Connect(ctx context.Context) error {
	c.connectMu.Lock()
	defer c.connectMu.Unlock()
	if c.connected {
		return nil
	}
	ctx, cancel := context.WithTimeoutCause(ctx, c.timeout, errTimeout)
	defer cancel()
	_, err := c.send(ctx, http.MethodGet, "/"+apiVersion+"/models/"+url.PathEscape(c.model), nil, modelsRetry)
	if err != nil {
		if timedOut(ctx) {
			return c.timeoutError()
		}
		return fmt.Errorf("%w: could not connect to Google Decisions: %w", classifier.ErrClassifier, err)
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

// Ask sends questions about one state and returns Google's answers, by the
// same names and in Google's own format, and the tokens the request used.
//
// input is what the questions are about, as the request's input: text, or a
// list of content blocks.
//
// It fails with an error wrapping classifier.ErrClassifier when Google rejected
// the request, kept refusing it because it was busy, could not be reached, did
// not answer in time, blocked the input, or left a question unanswered.
func (c *Client) Ask(
	ctx context.Context, input any, questions []NamedQuestion,
) (map[string]map[string]any, Usage, error) {
	if c.isClosed() {
		return nil, Usage{}, fmt.Errorf("%w: Google Decisions request failed: the client is closed",
			classifier.ErrClassifier)
	}
	qs := ordered.Object{}
	for _, q := range questions {
		qs.Set(q.Name, q.Question)
	}
	body, err := ordered.Marshal(ordered.Object{
		{Key: "model", Value: c.model}, {Key: "input", Value: input}, {Key: "questions", Value: qs},
	})
	if err != nil {
		return nil, Usage{}, fmt.Errorf("%w: cannot write the request: %w", classifier.ErrClassifier, err)
	}
	ctx, cancel := context.WithTimeoutCause(ctx, c.timeout, errTimeout)
	defer cancel()
	data, err := c.send(ctx, http.MethodPost, "/"+apiVersion+"/decisions", body, decisionsRetry)
	if err != nil {
		if timedOut(ctx) {
			return nil, Usage{}, c.timeoutError()
		}
		return nil, Usage{}, fmt.Errorf("%w: Google Decisions request failed: %w", classifier.ErrClassifier, err)
	}
	return c.read(data, questions)
}

// read turns Google's reply into answers, counting the tokens it used.
func (c *Client) read(data []byte, questions []NamedQuestion) (map[string]map[string]any, Usage, error) {
	var obj map[string]any
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, Usage{}, fmt.Errorf("%w: Google Decisions request failed: the reply is not a JSON object: %w",
			classifier.ErrClassifier, err)
	}
	var usage Usage
	if u, ok := obj["usage"].(map[string]any); ok {
		usage.InputTokens = intOf(u["total_input_tokens"])
		usage.OutputTokens = intOf(u["total_output_tokens"])
	}
	c.mu.Lock()
	c.usage.InputTokens += usage.InputTokens
	c.usage.OutputTokens += usage.OutputTokens
	c.mu.Unlock()

	answers, ok := obj["answers"].(map[string]any)
	if !ok {
		return nil, Usage{}, fmt.Errorf("%w: %s", classifier.ErrClassifier, noAnswers(obj["prompt_feedback"]))
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
		return nil, Usage{}, fmt.Errorf("%w: Google Decisions reply has no answer for %s",
			classifier.ErrClassifier, strings.Join(missing, ", "))
	}
	return out, usage, nil
}

// noAnswers is why a reply has no answers: the reason Google's filters blocked
// the input, if they did.
func noAnswers(feedback any) string {
	f, _ := feedback.(map[string]any)
	reason, _ := f["block_reason"].(string)
	if reason == "" {
		return "Google Decisions reply has no answers"
	}
	message := "Google Decisions blocked the input: " + reason
	if detail, _ := f["block_reason_message"].(string); detail != "" {
		message += ": " + detail
	}
	return message
}

// timeoutError reports a request that was not answered within the timeout.
func (c *Client) timeoutError() error {
	return fmt.Errorf("%w: Google Decisions did not answer within %gs", classifier.ErrClassifier, c.timeout.Seconds())
}

// retryPolicy is when a failed attempt is tried again, and how long to wait
// before it.
type retryPolicy struct {
	// retryable reports whether a reply with this status is tried again.
	retryable func(status int) bool
	// retryAfter is whether the wait a reply asks for, in its Retry-After or
	// retry-after-ms header, replaces the backoff.
	retryAfter bool
	// backoff is the wait before retry n, counted from 0.
	backoff func(n int) time.Duration
}

// decisionsRetry is how a question is retried: on a timeout, a conflict, a
// rate limit or a server error, and when Google can't be reached, waiting as
// long as Google asks or else 250 ms doubled for each retry, less up to a
// quarter, and at most 2 s.
//
//nolint:gochecknoglobals // a fixed policy, shared by every client
var decisionsRetry = retryPolicy{
	retryable: func(status int) bool {
		return status == http.StatusRequestTimeout || status == http.StatusConflict ||
			status == http.StatusTooManyRequests || status >= 500 && status < 600
	},
	retryAfter: true,
	backoff: func(n int) time.Duration {
		d := float64(initialDelay) * math.Pow(2, float64(n)) * (1 - rand.Float64()*0.25) //nolint:gosec // jitter
		return min(time.Duration(d), maxDelay)
	},
}

// modelsRetry is how the model lookup Connect sends is retried: on a timeout,
// a rate limit, and the server errors that may pass, and when Google can't be
// reached, waiting 250 ms doubled for each retry plus up to a second, and at
// most 2 s.
//
//nolint:gochecknoglobals // a fixed policy, shared by every client
var modelsRetry = retryPolicy{
	retryable: func(status int) bool {
		switch status {
		case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
			http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			return true
		default:
			return false
		}
	},
	backoff: func(n int) time.Duration {
		d := float64(initialDelay)*math.Pow(2, float64(n)) + rand.Float64()*float64(time.Second) //nolint:gosec // jitter
		return min(time.Duration(d), maxDelay)
	},
}

// send sends one request, retrying it as policy says, and returns the body of
// the reply that succeeded. A reply that failed is an error carrying its status
// and the reason Google gave.
func (c *Client) send(ctx context.Context, method, path string, body []byte, policy retryPolicy) ([]byte, error) {
	for retry := 0; ; retry++ {
		r, err := c.do(ctx, method, path, body)
		last := retry >= c.maxRetries
		switch {
		case err != nil:
			if last || ctx.Err() != nil {
				return nil, err
			}
		case policy.retryable(r.status) && !last:
		case r.status >= 400:
			return nil, failure(r.status, r.data)
		default:
			return r.data, nil
		}
		wait := policy.backoff(retry)
		if err == nil && policy.retryAfter {
			if asked := retryAfter(r.header); asked > 0 {
				wait = asked
			}
		}
		slog.Debug("Google Decisions request failed, retrying", "status", r.status, "error", err, "wait", wait)
		if err := c.sleep(ctx, wait); err != nil {
			return nil, err
		}
	}
}

// response is a reply Google sent, read in full.
type response struct {
	status int
	header http.Header
	data   []byte
}

// do sends one request with the API key and reads its reply.
func (c *Client) do(ctx context.Context, method, path string, body []byte) (response, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return response{}, err
	}
	req.Header.Set("X-Goog-Api-Key", c.apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return response{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		return response{}, err
	}
	return response{status: resp.StatusCode, header: resp.Header, data: buf.Bytes()}, nil
}

// failure is a failed reply as an error: its HTTP status and the reason Google
// gave, if it gave one.
func failure(status int, data []byte) error {
	var body struct {
		Error struct {
			Message any `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(data, &body)
	if message, _ := body.Error.Message.(string); strings.TrimSpace(message) != "" {
		return fmt.Errorf("HTTP %d: %s", status, strings.TrimSpace(message)) //nolint:err113 // carries Google's reason
	}
	return fmt.Errorf("HTTP %d", status) //nolint:err113 // carries the status
}

// retryAfter is the wait a reply asks for before the next attempt, or zero when
// it asks for none: retry-after-ms in milliseconds when present, otherwise
// Retry-After in seconds or as a date.
func retryAfter(header http.Header) time.Duration {
	var wait time.Duration
	if value := header.Get("Retry-After"); value != "" {
		if seconds, err := strconv.ParseFloat(value, 64); err == nil {
			wait = time.Duration(seconds * float64(time.Second))
		} else if at, err := http.ParseTime(value); err == nil {
			wait = max(0, time.Until(at))
		}
	}
	if value := header.Get("Retry-After-Ms"); value != "" {
		if ms, err := strconv.ParseFloat(value, 64); err == nil && ms >= 0 {
			wait = time.Duration(ms * float64(time.Millisecond))
		}
	}
	return wait
}

// timedOut reports whether ctx ended at the client's timeout.
func timedOut(ctx context.Context) bool {
	return errors.Is(context.Cause(ctx), errTimeout)
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
