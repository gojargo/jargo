// Package sdkretry sends a request the way OpenAI's SDK clients do: a request
// that failed for a reason that may not repeat is sent again, up to twice, after
// a short backoff or the wait the server asked for.
//
// It covers the response arriving, not the body: once a response has started,
// a stream that breaks part way is the caller's to deal with.
package sdkretry

import (
	"context"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"strconv"
	"time"
)

// The SDK's defaults: how many times a request is sent again, and the bounds of
// the backoff between attempts.
const (
	MaxRetries        = 2
	initialRetryDelay = 500 * time.Millisecond
	maxRetryDelay     = 8 * time.Second
	// maxRetryAfter is the longest wait a server may ask for and be obeyed.
	maxRetryAfter = 60 * time.Second
)

// Do sends the request newRequest builds, building a fresh one for each
// attempt. A request is sent again when it could not be sent at all, or when
// the response says it may succeed if repeated (see ShouldRetry). The response
// of the last attempt is returned whatever its status, for the caller to read.
func Do(ctx context.Context, client *http.Client, newRequest func() (*http.Request, error)) (*http.Response, error) {
	return do(ctx, client, newRequest, sleep)
}

// do is Do with the wait between attempts given.
func do(
	ctx context.Context, client *http.Client, newRequest func() (*http.Request, error),
	wait func(context.Context, time.Duration),
) (*http.Response, error) {
	for attempt := 0; ; attempt++ {
		req, err := newRequest()
		if err != nil {
			return nil, err
		}
		remaining := MaxRetries - attempt
		resp, err := client.Do(req)
		if err != nil {
			if ctx.Err() != nil || remaining <= 0 {
				return nil, err
			}
			slog.DebugContext(ctx, "retrying a request that could not be sent", "attempt", attempt+1, "err", err)
			wait(ctx, retryDelay(attempt, nil))
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			continue
		}
		if resp.StatusCode < http.StatusBadRequest || remaining <= 0 || !ShouldRetry(resp) {
			return resp, nil
		}
		slog.DebugContext(ctx, "retrying a request the server may answer next time",
			"attempt", attempt+1, "status", resp.StatusCode)
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
		wait(ctx, retryDelay(attempt, resp.Header))
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
}

// ShouldRetry reports whether a failed response may succeed if the request is
// sent again. The server's x-should-retry header decides when it is set;
// otherwise a request timeout (408), a lock timeout (409), a rate limit (429)
// and any server error (5xx) are retried.
func ShouldRetry(resp *http.Response) bool {
	switch resp.Header.Get("x-should-retry") {
	case "true":
		return true
	case "false":
		return false
	}
	switch resp.StatusCode {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return true
	}
	return resp.StatusCode >= http.StatusInternalServerError
}

// retryDelay is how long to wait before the retry that follows attempt. A wait
// the server asked for is obeyed when it is reasonable; otherwise the delay
// doubles with each retry, up to a ceiling, less up to a quarter at random.
func retryDelay(attempt int, header http.Header) time.Duration {
	if after, ok := retryAfter(header, time.Now()); ok && after > 0 && after <= maxRetryAfter {
		return after
	}
	backoff := min(float64(initialRetryDelay)*math.Pow(2, float64(attempt)), float64(maxRetryDelay))
	jitter := 1 - 0.25*rand.Float64() //nolint:gosec // jitter, not a secret
	return time.Duration(backoff * jitter)
}

// retryAfter reads the wait a server asked for: retry-after-ms in
// milliseconds, else Retry-After in seconds or as a date.
func retryAfter(header http.Header, now time.Time) (time.Duration, bool) {
	if header == nil {
		return 0, false
	}
	if ms, err := strconv.ParseFloat(header.Get("retry-after-ms"), 64); err == nil {
		return time.Duration(ms * float64(time.Millisecond)), true
	}
	value := header.Get("Retry-After")
	if secs, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(secs * float64(time.Second)), true
	}
	if at, err := http.ParseTime(value); err == nil {
		return at.Sub(now), true
	}
	return 0, false
}

// sleep waits for d, or until ctx ends.
func sleep(ctx context.Context, d time.Duration) {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}
