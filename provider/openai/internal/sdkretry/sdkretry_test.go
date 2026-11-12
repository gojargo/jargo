package sdkretry

import (
	"context"
	"maps"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

// statusServer answers each request with the next status in statuses, and the
// last one from then on, setting header on every answer.
func statusServer(t *testing.T, header http.Header, statuses ...int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		i := int(n.Add(1)) - 1
		maps.Copy(w.Header(), header)
		w.WriteHeader(statuses[min(i, len(statuses)-1)])
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

// send runs Do against srv with no real wait between attempts, and returns the
// final status and the waits asked for.
func send(t *testing.T, srv *httptest.Server) (int, []time.Duration) {
	t.Helper()
	var waits []time.Duration
	resp, err := do(t.Context(), srv.Client(), func() (*http.Request, error) {
		return http.NewRequestWithContext(t.Context(), http.MethodPost, srv.URL, nil)
	}, func(_ context.Context, d time.Duration) { waits = append(waits, d) })
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	_ = resp.Body.Close()
	return resp.StatusCode, waits
}

func TestRetriesTheStatusesThatMaySucceed(t *testing.T) {
	for _, status := range []int{408, 409, 429, 500, 503} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			srv, n := statusServer(t, nil, status, http.StatusOK)
			if got, _ := send(t, srv); got != http.StatusOK || n.Load() != 2 {
				t.Errorf("status %d after %d requests, want 200 after a retry", got, n.Load())
			}
		})
	}
}

func TestDoesNotRetryAClientError(t *testing.T) {
	for _, status := range []int{400, 401, 403, 404, 422} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			srv, n := statusServer(t, nil, status, http.StatusOK)
			if got, _ := send(t, srv); got != status || n.Load() != 1 {
				t.Errorf("status %d after %d requests, want %d after one", got, n.Load(), status)
			}
		})
	}
}

func TestGivesUpAfterTwoRetries(t *testing.T) {
	srv, n := statusServer(t, nil, http.StatusServiceUnavailable)
	got, waits := send(t, srv)
	if got != http.StatusServiceUnavailable || n.Load() != 3 {
		t.Errorf("status %d after %d requests, want 503 after three", got, n.Load())
	}
	if len(waits) != 2 {
		t.Fatalf("waited %d times, want 2", len(waits))
	}
	// 0.5 s then 1 s, each less up to a quarter.
	if waits[0] < 375*time.Millisecond || waits[0] > 500*time.Millisecond ||
		waits[1] < 750*time.Millisecond || waits[1] > time.Second {
		t.Errorf("waits = %v, want about 0.5 s then 1 s", waits)
	}
}

func TestShouldRetryHeaderDecides(t *testing.T) {
	srv, n := statusServer(t, http.Header{"X-Should-Retry": {"false"}}, http.StatusServiceUnavailable)
	if got, _ := send(t, srv); got != http.StatusServiceUnavailable || n.Load() != 1 {
		t.Errorf("status %d after %d requests, want no retry when the server says so", got, n.Load())
	}
	srv, n = statusServer(t, http.Header{"X-Should-Retry": {"true"}}, http.StatusBadRequest, http.StatusOK)
	if got, _ := send(t, srv); got != http.StatusOK || n.Load() != 2 {
		t.Errorf("status %d after %d requests, want a retry when the server asks", got, n.Load())
	}
}

func TestObeysTheWaitTheServerAsksFor(t *testing.T) {
	cases := []struct {
		header http.Header
		want   time.Duration
	}{
		{http.Header{"Retry-After": {"2"}}, 2 * time.Second},
		{http.Header{"Retry-After-Ms": {"1500"}}, 1500 * time.Millisecond},
	}
	for _, c := range cases {
		srv, _ := statusServer(t, c.header, http.StatusTooManyRequests, http.StatusOK)
		if _, waits := send(t, srv); len(waits) != 1 || waits[0] != c.want {
			t.Errorf("header %v: waits = %v, want %v", c.header, waits, c.want)
		}
	}
	// A wait longer than a minute is not obeyed: the backoff applies instead.
	srv, _ := statusServer(t, http.Header{"Retry-After": {"120"}}, http.StatusTooManyRequests, http.StatusOK)
	if _, waits := send(t, srv); len(waits) != 1 || waits[0] > 500*time.Millisecond {
		t.Errorf("waits = %v, want the backoff in place of an unreasonable wait", waits)
	}
}

func TestRetriesARequestThatCouldNotBeSent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()
	attempts := 0
	resp, err := do(t.Context(), http.DefaultClient, func() (*http.Request, error) {
		attempts++
		return http.NewRequestWithContext(t.Context(), http.MethodPost, url, nil)
	}, func(context.Context, time.Duration) {})
	if resp != nil {
		_ = resp.Body.Close()
	}
	if err == nil || attempts != 3 {
		t.Errorf("err = %v after %d attempts, want a failure after three", err, attempts)
	}
}

func TestRetryAfterDate(t *testing.T) {
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	h := http.Header{"Retry-After": {now.Add(3 * time.Second).Format(http.TimeFormat)}}
	if got, ok := retryAfter(h, now); !ok || got != 3*time.Second {
		t.Errorf("retryAfter = %v, %v, want 3s", got, ok)
	}
}
