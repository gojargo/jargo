// Package stream reads the streamed responses of the Google LLM services: the
// server-sent events of one request, bounded by how long the model may go quiet
// for, with a request that never began answering re-issued once.
package stream

import (
	"context"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/internal/sse"
	"github.com/gojargo/jargo/service/llm"
)

// Config says how a streamed request is bounded and retried, and what the
// service reading it does around each attempt.
type Config struct {
	// Client sends the requests.
	Client *http.Client
	// IdleTimeout bounds the gap between two events, and the wait for the first
	// one when the request is not retried. Zero waits indefinitely.
	IdleTimeout time.Duration
	// RetryOnTimeout re-issues the request once when its first event does not
	// arrive within RetryTimeout.
	RetryOnTimeout bool
	// RetryTimeout is how long the first event is waited for before the request
	// is re-issued, when RetryOnTimeout is set.
	RetryTimeout time.Duration
	// Begin is called as each attempt is sent, which is where a service starts
	// measuring the time to its first byte. Nil does nothing.
	Begin func()
	// Retrying is called before a stalled request is re-issued, for the service
	// to log it. Nil does nothing.
	Retrying func(ctx context.Context)
	// StatusError is the error a response with a status other than 200 ends the
	// stream with. It must not close the body.
	StatusError func(resp *http.Response) error
}

// WithRetry issues req and hands the data of each event of its streamed
// response to onData, bounding how long the model may go quiet for.
//
// A request the API accepts and then never answers is the case this exists for:
// nothing closes such a stream, so without a bound the turn stays open and the
// caller waits on a bot that will never speak. With RetryOnTimeout set, a first
// event that does not arrive in time re-issues the request once instead, which
// is safe only up to that point: an event that has been handed on is already
// downstream, so re-issuing after one would say everything twice.
//
// A stream the bound cuts short ends with an error wrapping
// llm.ErrCompletionTimeout. onData is given the attempt's context, which the
// bound cancels.
func WithRetry(req *http.Request, cfg Config, onData func(ctx context.Context, data string) error) error {
	if cfg.RetryOnTimeout {
		retried, err := attempt(req, cfg, cfg.RetryTimeout, onData)
		if err == nil || !retried {
			return err
		}
		if cfg.Retrying != nil {
			cfg.Retrying(req.Context())
		}
	}
	_, err := attempt(req, cfg, cfg.IdleTimeout, onData)
	return err
}

// attempt runs one attempt, reporting whether it timed out before the model had
// said anything, which is the only point a request can be re-issued from. Every
// event after the first is bounded by the idle timeout.
func attempt(
	req *http.Request, cfg Config, firstEvent time.Duration,
	onData func(ctx context.Context, data string) error,
) (retriable bool, err error) {
	// The request is rebuilt per attempt: its body has been read by the time an
	// attempt ends, and a re-issue needs one to send.
	sent, err := cloneRequest(req)
	if err != nil {
		return false, err
	}

	// The watchdog cancels the request when the model goes quiet for too long,
	// which is what unblocks the read. The flag tells that cancellation apart
	// from the pipeline canceling the turn.
	ctx, cancel := context.WithCancel(sent.Context())
	defer cancel()
	var timedOut atomic.Bool
	var watchdog *time.Timer
	arm := func(d time.Duration) {
		if d <= 0 {
			return
		}
		if watchdog == nil {
			watchdog = time.AfterFunc(d, func() { timedOut.Store(true); cancel() })
			return
		}
		watchdog.Reset(d)
	}
	defer func() {
		if watchdog != nil {
			watchdog.Stop()
		}
	}()

	if cfg.Begin != nil {
		cfg.Begin()
	}
	arm(firstEvent)
	resp, err := cfg.Client.Do(sent.WithContext(ctx)) //nolint:gosec // G704: the URL is the calling service's own endpoint
	if err != nil {
		return timedOut.Load(), streamError(sent.Context(), cfg, &timedOut, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return false, cfg.StatusError(resp)
	}

	// began records that the model has said something, which is the point past
	// which the request can no longer be re-issued.
	var began bool
	err = sse.Scan(resp.Body, func(data string) error {
		// Any event at all is the model still producing, so the bound is on the
		// gap between events rather than on the response as a whole.
		arm(cfg.IdleTimeout)
		began = true
		return onData(ctx, data)
	})
	if err != nil {
		return timedOut.Load() && !began, streamError(sent.Context(), cfg, &timedOut, err)
	}
	return false, nil
}

// streamError names the failure that ended a stream. A read the watchdog cut
// short is reported as a completion timeout, which is what tells the base to
// report it as one and notify anything watching for it.
func streamError(ctx context.Context, cfg Config, timedOut *atomic.Bool, err error) error {
	if timedOut.Load() {
		return fmt.Errorf("%w: the model stopped producing for %s",
			llm.ErrCompletionTimeout, cfg.IdleTimeout)
	}
	return llm.AsCompletionTimeout(ctx, err)
}

// cloneRequest copies a request along with its body, so an attempt that consumed
// one can be made again.
func cloneRequest(req *http.Request) (*http.Request, error) {
	if req.GetBody == nil {
		return req, nil
	}
	body, err := req.GetBody()
	if err != nil {
		return nil, err
	}
	clone := req.Clone(req.Context())
	clone.Body = body
	return clone, nil
}
