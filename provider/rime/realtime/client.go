package realtime

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultEndpoint is Rime's public Prism endpoint.
	defaultEndpoint = "wss://prism.api.rime.ai/v1/realtime"
	// readLimit bounds one message from the server.
	readLimit = 1024 * 1024
	// maxRetryAfterHeader bounds the Retry-After header that is read.
	maxRetryAfterHeader = 256
	// maxRefusalBody bounds the body of a refused upgrade that is read for the
	// reason.
	maxRefusalBody = 1024
)

// connectOptions is what a conversation opens with.
type connectOptions struct {
	apiKey   string
	endpoint string
	// voice and instructions are sent when they are given.
	voice, instructions *string
	tools               []toolDefinition
	interruptOnSpeech   bool
	turnDetection       *TurnDetection
	speed               *float64
	timeouts            Timeouts
	// requestID is sent as the upgrade's x-request-id when it is not empty.
	requestID        string
	responseMetadata map[string]string
	truncation       *Truncation
}

// retryAfter reads the delay a refusal asked for: a number of seconds, or an
// HTTP date. It returns nil for anything else.
func retryAfter(values []string, now time.Time) *time.Duration {
	if len(values) != 1 || len(values[0]) > maxRetryAfterHeader {
		return nil
	}
	value := strings.TrimSpace(values[0])
	if value != "" && strings.Trim(value, "0123456789") == "" {
		seconds, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return nil
		}
		// A delay too long to represent waits past any deadline.
		delay := time.Duration(math.MaxInt64)
		if seconds < float64(math.MaxInt64)/float64(time.Second) {
			delay = time.Duration(seconds * float64(time.Second))
		}
		return &delay
	}
	date, ok := httpDate(value, now)
	if !ok {
		return nil
	}
	delay := max(date.Sub(now), 0)
	return &delay
}

// httpDate reads a date in one of the three forms HTTP allows.
func httpDate(value string, now time.Time) (time.Time, bool) {
	if date, err := time.Parse(http.TimeFormat, value); err == nil {
		return date, true
	}
	if date, err := time.Parse("Monday, 02-Jan-06 15:04:05 GMT", value); err == nil {
		// A two-digit year more than 50 years ahead denotes the previous
		// century.
		current := now.UTC()
		year := (current.Year()+50)/100*100 + date.Year()%100
		date = time.Date(year, date.Month(), date.Day(), date.Hour(), date.Minute(), date.Second(), 0, time.UTC)
		if date.After(current.AddDate(50, 0, 0)) {
			date = date.AddDate(-100, 0, 0)
		}
		return date, true
	}
	if date, err := time.Parse(time.ANSIC, value); err == nil {
		return date, true
	}
	return time.Time{}, false
}

// single is the header's value when it has exactly one that is not empty.
func single(values []string) string {
	if len(values) == 1 {
		return values[0]
	}
	return ""
}

// refusalDetail reads the reason a refused upgrade gives in its body, and the
// structured kind of the refusal when the body names one.
func refusalDetail(body []byte) (detail, kind string) {
	if len(body) == 0 || len(body) >= maxRefusalBody || !isValidUTF8(body) {
		return "", ""
	}
	detail = strings.TrimSpace(string(body))
	if strings.HasPrefix(detail, "{") || strings.HasPrefix(detail, "[") {
		var payload struct {
			Error struct {
				Kind    any `json:"kind"`
				Message any `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(detail), &payload); err != nil {
			return "", ""
		}
		kind, _ = payload.Error.Kind.(string)
		detail, _ = payload.Error.Message.(string)
	}
	if strings.HasPrefix(strings.TrimLeft(detail, " \t\r\n\v\f"), "<") {
		return "", kind
	}
	for _, c := range detail {
		if c < 32 && c != '\r' && c != '\n' && c != '\t' {
			return "", kind
		}
	}
	return strings.Join(strings.Fields(detail), " "), kind
}

func isValidUTF8(data []byte) bool { return utf8Valid(string(data)) }

// handshakeError is the failure a refused upgrade means.
func handshakeError(refusal *wsutil.HandshakeError, now time.Time) *Error {
	status := refusal.StatusCode
	kind, reason := KindUnavailable, "Realtime endpoint rejected the connection"
	switch status {
	case http.StatusUnauthorized:
		kind, reason = KindAuthentication, "Realtime endpoint rejected credentials"
	case http.StatusPaymentRequired:
		kind, reason = KindBilling, "Realtime access requires available credits"
	case http.StatusForbidden:
		kind, reason = KindPermission, "Realtime access is not permitted"
	case http.StatusTooManyRequests:
		kind, reason = KindResourceLimit, "Realtime connection or rate limit reached"
	case http.StatusServiceUnavailable:
		reason = "Realtime endpoint is unavailable"
	}
	failure := newError(kind, reason+" (HTTP "+strconv.Itoa(status)+")")
	failure.StatusCode = status
	failure.ServerKind = single(refusal.Header.Values("X-Rime-Error-Kind"))
	detail, serverKind := refusalDetail(refusal.Body)
	if serverKind != "" {
		failure.ServerKind = serverKind
	}
	if detail != "" {
		failure.Message += ": " + detail
	}
	failure.RequestID = single(refusal.Header.Values("X-Request-Id"))
	failure.RetryAfter = retryAfter(refusal.Header.Values("Retry-After"), now)
	return failure
}

// checkEndpoint reports whether endpoint is a Prism endpoint credentials can
// be sent to.
func checkEndpoint(endpoint string) error {
	parsed, err := url.Parse(endpoint)
	if err != nil || (parsed.Scheme != "ws" && parsed.Scheme != "wss") || parsed.Hostname() == "" ||
		parsed.User != nil || parsed.Fragment != "" || strings.TrimRight(parsed.Path, "/") != "/v1/realtime" {
		return inputError("endpoint must be a full ws:// or wss:// URL ending in /v1/realtime")
	}
	if parsed.Scheme == "ws" && !slices.Contains([]string{"localhost", "127.0.0.1", "::1"}, parsed.Hostname()) {
		return inputError("Use wss:// outside localhost to protect credentials")
	}
	return nil
}

// printableASCII reports whether value is nonblank printable ASCII.
func printableASCII(value string) bool {
	if strings.TrimSpace(value) == "" {
		return false
	}
	for _, c := range value {
		if c < ' ' || c > '~' {
			return false
		}
	}
	return true
}

// openingSettings is the session configuration a conversation opens with.
func (o connectOptions) openingSettings() (map[string]any, error) {
	tools, err := toolSettings(o.tools)
	if err != nil {
		return nil, err
	}
	turn := map[string]any{"interrupt_response": o.interruptOnSpeech}
	settings := map[string]any{
		"modalities":         []string{"text", "audio"},
		"input_audio_format": "pcm16",
		keyTurn:              turn,
		keyTools:             tools,
	}
	if o.truncation != nil {
		value, err := truncationSettings(*o.truncation)
		if err != nil {
			return nil, err
		}
		settings["truncation"] = value
	}
	if o.speed != nil {
		speed, err := speedSetting(*o.speed)
		if err != nil {
			return nil, err
		}
		settings["speed"] = speed
	}
	if o.turnDetection != nil {
		rules, err := turnSettings(*o.turnDetection)
		if err != nil {
			return nil, err
		}
		maps.Copy(turn, rules)
	}
	if o.responseMetadata != nil {
		settings[keyLabels] = responseMetadata(o.responseMetadata)
	}
	if o.requestID != "" && !printableASCII(o.requestID) {
		return nil, inputError("request_id must be nonblank printable ASCII")
	}
	for name, value := range map[string]*string{"voice": o.voice, "instructions": o.instructions} {
		if value == nil {
			continue
		}
		if strings.TrimSpace(*value) == "" {
			return nil, inputError(name + " must be nonblank")
		}
		settings[name] = *value
	}
	return settings, nil
}

// connect opens a session on a Prism endpoint and configures it. The caller
// closes the session it returns.
//
// Credentials are sent as a bearer token with the API key, so ws:// is taken
// for a local development server only.
func connect(ctx context.Context, o connectOptions) (*session, error) {
	if strings.TrimSpace(o.apiKey) == "" {
		return nil, newError(KindAuthentication, "Provide an API key")
	}
	endpoint := o.endpoint
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	if err := checkEndpoint(endpoint); err != nil {
		return nil, err
	}
	settings, err := o.openingSettings()
	if err != nil {
		return nil, err
	}
	header := http.Header{}
	header.Set("Authorization", "Bearer "+o.apiKey)
	if o.requestID != "" {
		header.Set("X-Request-Id", o.requestID)
	}

	dialCtx, cancel := context.WithTimeout(ctx, o.timeouts.Connect)
	conn, err := wsutil.Dial(dialCtx, endpoint, header, readLimit)
	cancel()
	if err != nil {
		return nil, dialError(ctx, dialCtx, err)
	}
	// Each write is bounded by the request timeout instead.
	conn.SetSendTimeout(0)
	opened := newSession(conn, o.timeouts)
	if err := opened.initialize(ctx, settings); err != nil {
		opened.close()
		return nil, err
	}
	return opened, nil
}

// dialError is the failure a connection that could not be opened means.
func dialError(ctx, dialCtx context.Context, err error) error {
	if refusal, ok := errors.AsType[*wsutil.HandshakeError](err); ok {
		return handshakeError(refusal, time.Now())
	}
	switch {
	case ctx.Err() != nil:
		return ctx.Err()
	case dialCtx.Err() != nil:
		return newError(KindTimeout, "Realtime connection timed out")
	default:
		return newError(KindUnavailable, "Realtime endpoint is unavailable")
	}
}

// StartupParams bounds the opening of a new conversation.
type StartupParams struct {
	// MaxRetries is how many more attempts follow a refusal that may not
	// repeat, from 0 to 10; nil uses 3.
	MaxRetries *int `validate:"omitempty,min=0,max=10"`
	// RetryInterval is the least time between attempts; nil uses 2 s.
	RetryInterval *time.Duration `validate:"omitempty,min=0"`
	// AttemptTimeout bounds one connection and its initialization; 0 uses
	// 30 s.
	AttemptTimeout time.Duration `validate:"min=0"`
	// TotalTimeout bounds all the attempts and the delays between them; 0 uses
	// 60 s.
	TotalTimeout time.Duration `validate:"min=0"`
}

// The default bounds of a startup.
const (
	defaultMaxRetries     = 3
	defaultRetryInterval  = 2 * time.Second
	defaultAttemptTimeout = 30 * time.Second
	defaultTotalTimeout   = 60 * time.Second
)

func (p StartupParams) maxRetries() int {
	if p.MaxRetries == nil {
		return defaultMaxRetries
	}
	return *p.MaxRetries
}

func (p StartupParams) retryInterval() time.Duration {
	if p.RetryInterval == nil {
		return defaultRetryInterval
	}
	return *p.RetryInterval
}

func (p StartupParams) attemptTimeout() time.Duration {
	if p.AttemptTimeout == 0 {
		return defaultAttemptTimeout
	}
	return p.AttemptTimeout
}

func (p StartupParams) totalTimeout() time.Duration {
	if p.TotalTimeout == 0 {
		return defaultTotalTimeout
	}
	return p.TotalTimeout
}

// retryableStatuses are the refusals another attempt may not meet.
//
//nolint:gochecknoglobals // a fixed set
var retryableStatuses = []int{408, 429, 500, 502, 503, 504}

// retryDelay is how long to wait before trying again after an attempt failed
// with err, and whether to: only an HTTP refusal that may not repeat is tried
// again, as many times as the policy allows.
func retryDelay(policy StartupParams, attempt int, err error) (time.Duration, bool) {
	refusal, ok := errors.AsType[*Error](err)
	if !ok || (refusal.Kind != KindUnavailable && refusal.Kind != KindResourceLimit) ||
		attempt == policy.maxRetries() || !slices.Contains(retryableStatuses, refusal.StatusCode) {
		return 0, false
	}
	delay := policy.retryInterval()
	if refusal.RetryAfter != nil {
		delay = max(delay, *refusal.RetryAfter)
	}
	return delay, true
}

// connectWithRetry opens a conversation, trying again after a refusal that
// may not repeat. An established conversation is never replayed.
func connectWithRetry(ctx context.Context, policy StartupParams, o connectOptions) (*session, error) {
	deadline := time.Now().Add(policy.totalTimeout())
	for attempt := 0; ; attempt++ {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return nil, newError(KindTimeout, "Prism startup deadline expired")
		}
		attemptCtx, cancel := context.WithTimeout(ctx, min(remaining, policy.attemptTimeout()))
		opened, err := connect(attemptCtx, o)
		expired := attemptCtx.Err() != nil && ctx.Err() == nil
		cancel()
		if err == nil {
			return opened, nil
		}
		if expired && errors.Is(err, context.DeadlineExceeded) {
			return nil, newError(KindTimeout, "Prism startup attempt timed out")
		}
		delay, retry := retryDelay(policy, attempt, err)
		if !retry || delay >= time.Until(deadline) {
			return nil, err
		}
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}
