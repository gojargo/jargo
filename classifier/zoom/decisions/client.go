// Package decisions is a classifier backed by the Zoom Decisions API.
//
// One Client holds one HTTP/2 connection pool, adds the auth and tracking
// headers, retries when Zoom is busy, and counts the tokens every request
// used. Several Classifiers can share one. A Classifier turns each question
// into Zoom's format and each answer into a result, through a Client.
//
// See https://developers.zoom.us/docs/ai-services/decisions/
package decisions

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/internal/ordered"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/internal/zoomaiutil"
	"github.com/gojargo/jargo/retry"
)

const (
	// DefaultURL is where questions are sent.
	DefaultURL = "https://api.zoom.us/v2/aiservices/decisions/decide"
	// DefaultModel is the decision model to ask: "zoom-decisions" or
	// "zoom-decisions-pro".
	DefaultModel = "zoom-decisions"
	// DefaultTimeout is how long to wait for an answer, retries included,
	// before giving up.
	DefaultTimeout = 10 * time.Second
	// defaultMaxRetries is how many times a request Zoom refused because it was
	// busy is retried.
	defaultMaxRetries = 3

	// tooManyRequests is the status Zoom answers with when the caller is rate
	// limited.
	tooManyRequests = 429
	// overloaded is the status Zoom answers with when it is temporarily
	// overloaded.
	overloaded = 503
	// keepaliveExpiry is how long an idle connection is kept open. Questions
	// come with gaps between them, and a connection that has closed in the
	// meantime costs a new TLS handshake on the next one.
	keepaliveExpiry = 240 * time.Second
	// minRetryWait is the wait before the first retry, doubled for each one
	// after it.
	minRetryWait = 250 * time.Millisecond
	// maxRetryWait is the longest backoff between two attempts.
	maxRetryWait = 2 * time.Second
	// retryMultiplier scales the doubling so the first retry waits
	// minRetryWait.
	retryMultiplier = 0.25
)

// The types of a question and of an answer, and the keys they share.
const (
	typePredicate = "predicate"
	typeChoice    = "choice"
	typeScore     = "score"
	typeRefusal   = "refusal"

	keyType = "type"
	keyName = "name"
)

// errTimeout is the cause of a request's context ending at the client's
// timeout.
var errTimeout = errors.New("decisions: timeout")

// errClosed is why a closed client sends nothing.
var errClosed = errors.New("the client is closed")

// ClientConfig configures a Client.
type ClientConfig struct {
	// APIKey is the Zoom AI Services API key. Required.
	APIKey string `validate:"required"`
	// URL is the decide endpoint; empty uses DefaultURL. A plaintext http://
	// URL to a remote host logs a warning.
	URL string
	// Model is the decision model to ask, "zoom-decisions" or
	// "zoom-decisions-pro"; empty uses DefaultModel.
	Model string
	// Timeout is how long to wait for an answer, retries included, before
	// giving up; zero uses DefaultTimeout.
	Timeout time.Duration `validate:"min=0"`
	// MaxRetries is how many times to retry a request Zoom refused because it
	// was busy; nil uses 3.
	MaxRetries *int `validate:"omitempty,min=0"`
	// HTTPClient sends the requests; nil builds one that keeps HTTP/2
	// connections open between questions.
	HTTPClient *http.Client
}

// Usage is the tokens the Decisions API used: for one request, or over every
// request a client sent.
type Usage struct {
	// InputTokens is the tokens sent, as Zoom counts them in input_units.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the tokens received, one per question.
	OutputTokens int `json:"output_tokens"`
	// CachedInputTokens is the tokens sent that Zoom read from its cache.
	CachedInputTokens int `json:"cached_input_tokens"`
}

// Choice is one option of a choice question in Zoom's own format.
type Choice struct {
	// Value is the option, as the answer names it.
	Value string
	// Description is when the option applies, or nil when it is left out.
	Description *string
}

// Question is one question in Zoom's own format.
type Question struct {
	// Type is "predicate", "choice" or "score".
	Type string
	// Name names the answer. Distinct within one request.
	Name string
	// Instructions is what is being asked, as text.
	Instructions string
	// Choices are the options of a choice question, in order.
	Choices []Choice
	// Levels are the labels of a score question's levels, lowest first.
	Levels []string
}

// MarshalJSON implements json.Marshaler, writing the choices of a choice
// question and the levels of a score question.
func (q Question) MarshalJSON() ([]byte, error) {
	obj := ordered.Object{
		{Key: keyType, Value: q.Type}, {Key: keyName, Value: q.Name}, {Key: "instructions", Value: q.Instructions},
	}
	switch q.Type {
	case typeChoice:
		choices := make([]ordered.Object, 0, len(q.Choices))
		for _, choice := range q.Choices {
			c := ordered.Object{{Key: "value", Value: choice.Value}}
			if choice.Description != nil {
				c.Set("description", *choice.Description)
			}
			choices = append(choices, c)
		}
		obj.Set("choices", choices)
	case typeScore:
		levels := make([]ordered.Object, 0, len(q.Levels))
		for _, level := range q.Levels {
			levels = append(levels, ordered.Object{{Key: "label", Value: level}})
		}
		obj.Set("levels", levels)
	}
	return obj.MarshalJSON()
}

// Client is an HTTP client for the Zoom Decisions decide endpoint.
//
// It holds one HTTP/2 connection pool, so many small requests share a
// connection and can be in flight at the same time. The connection is kept open
// between questions, so a gap between them does not cost a new TLS handshake.
// It retries with backoff when Zoom answers 429 or 503, waiting at least as
// long as Zoom's Retry-After asks, within one time bound for the whole request.
// It counts the tokens every request used in Usage. Errors carry the reason
// Zoom gave and the request's x-zm-trackingid, which Zoom support uses to find
// it.
//
// Connect opens the connection ahead of the first question, and Close releases
// it.
type Client struct {
	apiKey     string
	url        string
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
		url:        cfg.URL,
		model:      cfg.Model,
		timeout:    cfg.Timeout,
		maxRetries: defaultMaxRetries,
		http:       cfg.HTTPClient,
		sleep:      sleep,
	}
	if c.url == "" {
		c.url = DefaultURL
	}
	c.url = zoomaiutil.CheckURL(c.url, "Zoom Decisions", "questions")
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

// URL is where the questions are sent.
func (c *Client) URL() string { return c.url }

// Timeout is how long the client waits for an answer, retries included.
func (c *Client) Timeout() time.Duration { return c.timeout }

// Usage is the tokens used so far, over every request.
func (c *Client) Usage() Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

// Connect opens the connection to Zoom.
//
// It sends an OPTIONS request to the endpoint, which opens the connection
// without asking anything. Zoom has no request that checks the key without
// asking a question, so a bad key shows on the first question. It connects
// once: a client shared by several classifiers is asked by each of them, and
// only the first call sends anything.
//
// It fails with an error wrapping classifier.ErrClassifier when Zoom could not
// be reached.
func (c *Client) Connect(ctx context.Context) error {
	c.connectMu.Lock()
	defer c.connectMu.Unlock()
	if c.connected {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodOptions, c.url, nil)
	if err != nil {
		return fmt.Errorf("%w: could not connect to Zoom Decisions: %w", classifier.ErrClassifier, err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%w: could not connect to Zoom Decisions: %w", classifier.ErrClassifier, err)
	}
	// The reply is read to its end so the connection goes back to the pool.
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
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

// Ask sends questions about one state and returns Zoom's answers, by question
// name and in Zoom's own format, and the tokens the request used.
//
// input is what the questions are about, as the request's input: text, or user
// messages. questions are in Zoom's own format, each with a distinct name.
// options are other request fields, such as "enable_image_input": true.
//
// It fails with an error wrapping classifier.ErrClassifier when Zoom rejected
// the request, kept refusing it because it was busy, could not be reached, did
// not answer in time, or left a question unanswered.
func (c *Client) Ask(
	ctx context.Context, input any, questions []Question, options map[string]any,
) (map[string]map[string]any, Usage, error) {
	if c.isClosed() {
		return nil, Usage{}, fmt.Errorf("%w: Zoom Decisions request failed: %w", classifier.ErrClassifier, errClosed)
	}
	request := ordered.Object{
		{Key: "model", Value: c.model},
		{Key: "input", Value: input},
		{Key: "questions", Value: append([]Question{}, questions...)},
	}
	for _, key := range slices.Sorted(maps.Keys(options)) {
		request.Set(key, options[key])
	}
	body, err := ordered.Marshal(request)
	if err != nil {
		return nil, Usage{}, fmt.Errorf("%w: cannot write the request: %w", classifier.ErrClassifier, err)
	}
	ctx, cancel := context.WithTimeoutCause(ctx, c.timeout, errTimeout)
	defer cancel()
	data, err := c.post(ctx, body)
	if err != nil {
		if errors.Is(context.Cause(ctx), errTimeout) {
			return nil, Usage{}, fmt.Errorf("%w: Zoom Decisions did not answer within %gs",
				classifier.ErrClassifier, c.timeout.Seconds())
		}
		return nil, Usage{}, err
	}
	return c.read(data, questions)
}

// read turns Zoom's reply into answers, counting the tokens it used.
func (c *Client) read(data map[string]any, questions []Question) (map[string]map[string]any, Usage, error) {
	usage, _ := data["usage"].(map[string]any)
	details, _ := usage["details"].(map[string]any)
	requestUsage := Usage{
		InputTokens:       count(usage["input_units"]),
		OutputTokens:      count(usage["output_units"]),
		CachedInputTokens: count(details["cached_input_tokens"]),
	}
	c.mu.Lock()
	c.usage.InputTokens += requestUsage.InputTokens
	c.usage.OutputTokens += requestUsage.OutputTokens
	c.usage.CachedInputTokens += requestUsage.CachedInputTokens
	c.mu.Unlock()

	answers, ok := data["answers"].([]any)
	if !ok {
		return nil, Usage{}, fmt.Errorf("%w: Zoom Decisions reply has no answers", classifier.ErrClassifier)
	}
	// Answers come back as a list, each carrying its question's name.
	byName := make(map[string]map[string]any, len(answers))
	for _, a := range answers {
		answer, isObject := a.(map[string]any)
		if !isObject {
			continue
		}
		if name, isString := answer[keyName].(string); isString {
			byName[name] = answer
		}
	}
	out := make(map[string]map[string]any, len(questions))
	var missing []string
	for _, q := range questions {
		answer, answered := byName[q.Name]
		if !answered {
			missing = append(missing, q.Name)
			continue
		}
		out[q.Name] = answer
	}
	if len(missing) > 0 {
		return nil, Usage{}, fmt.Errorf("%w: Zoom Decisions reply has no answer for %s",
			classifier.ErrClassifier, strings.Join(missing, ", "))
	}
	return out, requestUsage, nil
}

// post sends the request, retrying while Zoom is busy, and returns the reply.
func (c *Client) post(ctx context.Context, body []byte) (map[string]any, error) {
	deadline := time.Now().Add(c.timeout)
	for attempt := 1; ; attempt++ {
		trackingID := zoomaiutil.NewTrackingID()
		r, err := c.do(ctx, body, trackingID)
		if err != nil {
			return nil, requestFailed(err, trackingID)
		}
		if r.status == tooManyRequests || r.status == overloaded {
			wait := max(
				retry.ExponentialBackoffTime(attempt, minRetryWait, maxRetryWait, retryMultiplier),
				retryAfter(r.header),
			)
			if attempt > c.maxRetries || wait > time.Until(deadline) {
				return nil, fmt.Errorf("%w: Zoom Decisions is busy (%s) after %d attempts",
					classifier.ErrClassifier, failure(r, trackingID), attempt)
			}
			slog.Debug("Zoom Decisions busy, retrying", "status", r.status, "wait", wait)
			if err := c.sleep(ctx, wait); err != nil {
				return nil, requestFailed(err, trackingID)
			}
			continue
		}
		if r.status != http.StatusOK {
			return nil, fmt.Errorf("%w: Zoom Decisions rejected the request: %s",
				classifier.ErrClassifier, failure(r, trackingID))
		}
		var reply any
		if err := json.Unmarshal(r.data, &reply); err != nil {
			return nil, fmt.Errorf("%w: Zoom Decisions reply is not valid JSON: %w", classifier.ErrClassifier, err)
		}
		obj, ok := reply.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%w: Zoom Decisions reply is not a JSON object", classifier.ErrClassifier)
		}
		return obj, nil
	}
}

// requestFailed reports a request that got no reply, with its tracking ID.
func requestFailed(err error, trackingID string) error {
	return fmt.Errorf("%w: Zoom Decisions request failed: %w (tracking_id=%s)",
		classifier.ErrClassifier, err, trackingID)
}

// response is a reply Zoom sent, read in full.
type response struct {
	status int
	header http.Header
	data   []byte
}

// do sends one question with the auth and tracking headers and reads its
// reply.
func (c *Client) do(ctx context.Context, body []byte, trackingID string) (response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return response{}, err
	}
	zoomaiutil.SetRequestHeaders(req.Header, c.apiKey, trackingID)
	req.Header.Set("Content-Type", "application/json")
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

// failure is a failed reply's HTTP status, the reason Zoom gave, and the
// tracking ID.
func failure(r response, trackingID string) string {
	reason := fmt.Sprintf("HTTP %d", r.status)
	var body map[string]any
	if err := json.Unmarshal(r.data, &body); err == nil {
		if message, _ := body["message"].(string); message != "" {
			reason += ": " + message
		}
	}
	return fmt.Sprintf("%s (tracking_id=%s)", reason, trackingID)
}

// retryAfter is the wait Zoom's Retry-After header asks for, in seconds, or
// zero.
func retryAfter(header http.Header) time.Duration {
	value := header.Get("Retry-After")
	if value == "" {
		return 0
	}
	seconds, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil || math.IsNaN(seconds) || seconds <= 0 {
		return 0
	}
	if seconds >= math.MaxInt64/float64(time.Second) {
		return math.MaxInt64
	}
	return time.Duration(seconds * float64(time.Second))
}

func (c *Client) isClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// count reads a token count from the reply, 0 if there is none.
func count(value any) int {
	f, ok := value.(float64)
	if !ok || f != math.Trunc(f) {
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
