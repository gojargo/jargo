package errors_test

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"syscall"
	"testing"

	errs "github.com/gojargo/jargo/utils/errors"
)

// The failures the tests classify.
var (
	errForbidden = errors.New("forbidden")
	errPlain     = errors.New("nope")
)

// statusError is a provider error carrying a status, standing for the shapes an
// HTTP or websocket library raises.
type statusError struct {
	status int
}

func (e *statusError) Error() string       { return fmt.Sprintf("refused with %d", e.status) }
func (e *statusError) HTTPStatusCode() int { return e.status }

func TestStatusCodesMapToCategories(t *testing.T) {
	t.Parallel()
	cases := map[int]errs.Category{
		400: errs.InvalidRequest,
		401: errs.Authentication,
		402: errs.Quota,
		403: errs.Authorization,
		404: errs.InvalidRequest,
		422: errs.InvalidRequest,
		429: errs.RateLimit,
	}
	for status, want := range cases {
		if got := errs.ClassifyHTTPStatusCode(status); got != want {
			t.Errorf("status %d: got %q, want %q", status, got, want)
		}
	}
}

func TestServerErrorsMapToServerCategory(t *testing.T) {
	t.Parallel()
	for _, status := range []int{500, 502, 503, 599} {
		if got := errs.ClassifyHTTPStatusCode(status); got != errs.Server {
			t.Errorf("status %d: got %q, want %q", status, got, errs.Server)
		}
	}
}

func TestUnremarkableStatusCodesAreUnknown(t *testing.T) {
	t.Parallel()
	for _, status := range []int{200, 301, 418, 600} {
		if got := errs.ClassifyHTTPStatusCode(status); got != errs.Unknown {
			t.Errorf("status %d: got %q, want %q", status, got, errs.Unknown)
		}
	}
}

func TestExtractsStatusCodeFromAWrappedError(t *testing.T) {
	t.Parallel()
	wrapped := fmt.Errorf("dialing: %w", &statusError{status: 401})

	status, ok := errs.ExtractHTTPStatusCode(wrapped)

	if !ok || status != 401 {
		t.Fatalf("got (%d, %t), want (401, true)", status, ok)
	}
}

func TestExtractsStatusCodeFromTheSharedError(t *testing.T) {
	t.Parallel()
	err := errs.NewHTTPStatusError(403, errForbidden)

	status, ok := errs.ExtractHTTPStatusCode(err)

	if !ok || status != 403 {
		t.Fatalf("got (%d, %t), want (403, true)", status, ok)
	}
	if err.Error() != "forbidden" {
		t.Errorf("message: got %q, want %q", err.Error(), "forbidden")
	}
}

func TestAStatusOfZeroCarriesNothing(t *testing.T) {
	t.Parallel()
	if _, ok := errs.ExtractHTTPStatusCode(&statusError{status: 0}); ok {
		t.Error("a zero status was reported as one the error carries")
	}
}

func TestErrorsWithoutAStatusCode(t *testing.T) {
	t.Parallel()
	err := errPlain

	if _, ok := errs.ExtractHTTPStatusCode(err); ok {
		t.Error("a plain error reported a status")
	}
	if got := errs.ClassifyError(err); got != errs.Unknown {
		t.Errorf("got %q, want %q", got, errs.Unknown)
	}
}

func TestConnectivityErrors(t *testing.T) {
	t.Parallel()
	cases := map[string]error{
		"dial refused":   &net.OpError{Op: "dial", Err: syscall.ECONNREFUSED},
		"name lookup":    &net.DNSError{Err: "no such host", Name: "example.invalid"},
		"bare errno":     syscall.ECONNRESET,
		"deadline":       context.DeadlineExceeded,
		"os deadline":    os.ErrDeadlineExceeded,
		"wrapped refuse": fmt.Errorf("connecting: %w", syscall.ECONNREFUSED),
	}
	for name, err := range cases {
		if got := errs.ClassifyError(err); got != errs.Connectivity {
			t.Errorf("%s: got %q, want %q", name, got, errs.Connectivity)
		}
	}
}

func TestACancellationIsNotConnectivity(t *testing.T) {
	t.Parallel()
	// It says nothing about whether the service was reachable.
	if got := errs.ClassifyError(context.Canceled); got != errs.Unknown {
		t.Errorf("got %q, want %q", got, errs.Unknown)
	}
}

func TestAPlainOSErrorIsUnknown(t *testing.T) {
	t.Parallel()
	if got := errs.ClassifyError(os.ErrNotExist); got != errs.Unknown {
		t.Errorf("got %q, want %q", got, errs.Unknown)
	}
}

func TestClassifyingAStatusCarryingError(t *testing.T) {
	t.Parallel()
	if got := errs.ClassifyError(&statusError{status: 429}); got != errs.RateLimit {
		t.Errorf("got %q, want %q", got, errs.RateLimit)
	}
}

func TestAStatusBeatsConnectivity(t *testing.T) {
	t.Parallel()
	// The handshake reached the server, which refused it; the dial failure it
	// wraps is how the library reports that, not a failure to reach anything.
	err := errs.NewHTTPStatusError(401, &net.OpError{Op: "dial", Err: syscall.ECONNRESET})

	if got := errs.ClassifyError(err); got != errs.Authentication {
		t.Errorf("got %q, want %q", got, errs.Authentication)
	}
}

func TestClassifyingNothing(t *testing.T) {
	t.Parallel()
	if got := errs.ClassifyError(nil); got != errs.Unknown {
		t.Errorf("got %q, want %q", got, errs.Unknown)
	}
}

func TestPermanentCategories(t *testing.T) {
	t.Parallel()
	for _, c := range []errs.Category{errs.Authentication, errs.Authorization, errs.InvalidRequest} {
		if !c.IsPermanent() {
			t.Errorf("%q should be permanent", c)
		}
	}
	for _, c := range []errs.Category{
		errs.Unset, errs.Unknown, errs.RateLimit, errs.Quota,
		errs.Connectivity, errs.Server, errs.Application,
	} {
		if c.IsPermanent() {
			t.Errorf("%q should not be permanent", c)
		}
	}
}

// sdkError stands for the error a provider SDK reports a refusal through: the
// status is on a field, not behind a method, so nothing can be asked for it.
type sdkError struct {
	StatusCode int
	Response   *http.Response
}

func (e *sdkError) Error() string { return "sdk refused" }

// respOnlyError carries its status solely on the response, as an SDK reporting
// the refusal no further than the round trip does.
type respOnlyError struct {
	Response *http.Response
}

func (e *respOnlyError) Error() string { return "sdk refused" }

// textStatusError reports its status as text, which names a refusal but is not
// the code.
type textStatusError struct {
	Status string
}

func (e *textStatusError) Error() string { return "sdk refused" }

func TestExtractsStatusCodeFromAnSDKErrorsFields(t *testing.T) {
	t.Parallel()
	cases := map[string]struct {
		err  error
		want int
	}{
		"on the error": {&sdkError{StatusCode: http.StatusUnauthorized}, http.StatusUnauthorized},
		"on the response": {
			&respOnlyError{Response: &http.Response{StatusCode: http.StatusTooManyRequests}},
			http.StatusTooManyRequests,
		},
		"wrapped": {
			fmt.Errorf("generate: %w", &sdkError{StatusCode: http.StatusServiceUnavailable}),
			http.StatusServiceUnavailable,
		},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			got, ok := errs.ExtractHTTPStatusCode(c.err)
			if !ok || got != c.want {
				t.Fatalf("status = %d (%t), want %d", got, ok, c.want)
			}
		})
	}
}

func TestAStatusReportedAsTextCarriesNothing(t *testing.T) {
	t.Parallel()
	if _, ok := errs.ExtractHTTPStatusCode(&textStatusError{Status: "503 Service Unavailable"}); ok {
		t.Error("a status reported as text was read as the code")
	}
}

func TestAnSDKErrorClassifiesLikeAnyOther(t *testing.T) {
	t.Parallel()
	err := fmt.Errorf("generate: %w", &sdkError{StatusCode: http.StatusUnauthorized})
	if got := errs.ClassifyError(err); got != errs.Authentication {
		t.Errorf("got %q, want %q", got, errs.Authentication)
	}
}
