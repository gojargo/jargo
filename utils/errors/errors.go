// Package errors classifies failures the way every service shares: it gives a
// provider-agnostic vocabulary for why an operation failed, so a caller can tell
// a transient failure (worth retrying) from a permanent one (retrying will never
// help).
//
// Import it under a name of its own where the standard library's errors package
// is also needed, for example:
//
//	import errs "github.com/gojargo/jargo/utils/errors"
package errors

import (
	"errors"
	"net"
	"net/http"
	"reflect"
	"syscall"
)

// Category is why an operation failed, independent of the provider that failed
// it. The zero value means nobody has said yet, which invites the category to be
// worked out from the error.
type Category string

// The categories a failure can fall into.
const (
	// Unset is the zero value: the cause has not been decided yet.
	Unset Category = ""
	// Unknown means the cause could not be determined.
	Unknown Category = "unknown"
	// Authentication means credentials are missing or invalid.
	Authentication Category = "authentication"
	// Authorization means credentials are valid but lack access to the resource.
	Authorization Category = "authorization"
	// InvalidRequest means the request itself is malformed or names something
	// that does not exist, such as an unknown model or voice.
	InvalidRequest Category = "invalid_request"
	// RateLimit means too many requests were sent in too short a window.
	RateLimit Category = "rate_limit"
	// Quota means the account's credit or usage allowance is exhausted.
	Quota Category = "quota"
	// Connectivity means the service could not be reached.
	Connectivity Category = "connectivity"
	// Server means the provider reported an internal failure.
	Server Category = "server"
	// Application means application code failed, not the provider. Reported by a
	// service on behalf of code it invoked (a tool handler, say) whose failures
	// say nothing about the service's own health.
	Application Category = "application"
)

// IsPermanent reports whether the failure will keep recurring until something
// changes.
//
// A permanent failure gives the same result every time it is retried: the
// credentials stay rejected, the request stays malformed. Only new credentials
// or settings can clear it, so retrying is pointless.
func (c Category) IsPermanent() bool {
	switch c {
	case Authentication, Authorization, InvalidRequest:
		return true
	case Unset, Unknown, RateLimit, Quota, Connectivity, Server, Application:
		return false
	default:
		return false
	}
}

// String implements fmt.Stringer.
func (c Category) String() string { return string(c) }

// statusCategories maps the status codes that carry a provider-independent
// meaning. Anything else is decided by range.
//
//nolint:gochecknoglobals // a fixed table
var statusCategories = map[int]Category{
	http.StatusBadRequest:          InvalidRequest,
	http.StatusUnauthorized:        Authentication,
	http.StatusPaymentRequired:     Quota,
	http.StatusForbidden:           Authorization,
	http.StatusNotFound:            InvalidRequest,
	http.StatusUnprocessableEntity: InvalidRequest,
	http.StatusTooManyRequests:     RateLimit,
}

// ClassifyHTTPStatusCode classifies an HTTP status code, returning Unknown for a
// code that carries no provider-independent meaning.
func ClassifyHTTPStatusCode(status int) Category {
	if c, ok := statusCategories[status]; ok {
		return c
	}
	if status >= 500 && status < 600 {
		return Server
	}
	return Unknown
}

// StatusCoder is implemented by an error carrying the HTTP status a provider
// refused with. It is what ExtractHTTPStatusCode looks for as it walks an error
// chain, so a provider that wants its refusals classified reports them through
// an error implementing this.
type StatusCoder interface {
	error
	// HTTPStatusCode returns the status the provider refused with.
	HTTPStatusCode() int
}

// HTTPStatusError is an error carrying the HTTP status a provider refused with,
// for a caller that has a status and no error type of its own to put it on.
type HTTPStatusError struct {
	// Status is the HTTP status the provider refused with.
	Status int
	// Err is the underlying failure, whose message this error carries.
	Err error
}

// NewHTTPStatusError builds an HTTPStatusError carrying status and wrapping err.
func NewHTTPStatusError(status int, err error) *HTTPStatusError {
	return &HTTPStatusError{Status: status, Err: err}
}

// Error implements error.
func (e *HTTPStatusError) Error() string {
	if e.Err == nil {
		return http.StatusText(e.Status)
	}
	return e.Err.Error()
}

// Unwrap returns the underlying failure.
func (e *HTTPStatusError) Unwrap() error { return e.Err }

// HTTPStatusCode implements StatusCoder.
func (e *HTTPStatusError) HTTPStatusCode() int { return e.Status }

// ExtractHTTPStatusCode returns the HTTP status an error carries, walking the
// wrapped chain for the first error that reports one. The second result is false
// when nothing in the chain carries a status.
//
// An error implementing StatusCoder is asked. A provider SDK reports a refusal
// through a type of its own instead, so the rest of the chain is read for the
// status such a type puts on it: the field names below, on a response the error
// carries and then on the error itself.
func ExtractHTTPStatusCode(err error) (int, bool) {
	var sc StatusCoder
	if errors.As(err, &sc) {
		if status := sc.HTTPStatusCode(); status != 0 {
			return status, true
		}
	}
	for e := err; e != nil; e = errors.Unwrap(e) {
		if status, ok := statusFromFields(e); ok {
			return status, true
		}
	}
	return 0, false
}

// statusFieldNames are the fields a status is reported on, in the order they are
// read.
//
//nolint:gochecknoglobals // a fixed table
var statusFieldNames = []string{"StatusCode", "Status"}

// statusFromFields reads the status off an error's own fields, preferring the
// one on a response it carries to the one on the error itself.
func statusFromFields(err error) (int, bool) {
	v := structOf(reflect.ValueOf(err))
	if !v.IsValid() {
		return 0, false
	}
	if resp := v.FieldByName("Response"); resp.IsValid() {
		if status, ok := statusField(structOf(resp)); ok {
			return status, true
		}
	}
	return statusField(v)
}

// structOf dereferences v down to the struct it points at, returning the zero
// Value when it does not point at one.
func structOf(v reflect.Value) reflect.Value {
	for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return reflect.Value{}
		}
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return reflect.Value{}
	}
	return v
}

// statusField reads the first status field carrying a number. A status reported
// as text is not one: net/http spells Response.Status "503 Service Unavailable",
// which names the same refusal but is not the code.
func statusField(v reflect.Value) (int, bool) {
	if !v.IsValid() {
		return 0, false
	}
	for _, name := range statusFieldNames {
		f := v.FieldByName(name)
		if !f.IsValid() || !f.CanInterface() || !f.CanInt() {
			continue
		}
		if status := int(f.Int()); status != 0 {
			return status, true
		}
	}
	return 0, false
}

// ClassifyError classifies an error by the HTTP status it carries, falling back
// to Connectivity for a failure to reach the service at all, and to Unknown for
// anything unrecognized. A provider raising failures its own way needs its own
// classification.
func ClassifyError(err error) Category {
	if err == nil {
		return Unknown
	}
	if status, ok := ExtractHTTPStatusCode(err); ok {
		return ClassifyHTTPStatusCode(status)
	}
	if isConnectivity(err) {
		return Connectivity
	}
	return Unknown
}

// connectionErrnos are the bare syscall errors that mean the service could not
// be reached, for a failure raised outside a net.OpError to be recognized by.
//
//nolint:gochecknoglobals // a fixed table
var connectionErrnos = []error{
	syscall.ECONNREFUSED, syscall.ECONNRESET,
	syscall.EHOSTUNREACH, syscall.ENETUNREACH, syscall.EPIPE,
}

// isConnectivity reports whether err is a failure to reach the service at all.
//
// net.Error covers the dial, name-resolution and timeout failures, since the net
// package wraps each of them in an error reporting it. A bare syscall error is
// matched separately: one raised outside a net.OpError carries no such wrapper.
// A cancellation is deliberately not matched: it says nothing about whether the
// service was reachable.
func isConnectivity(err error) bool {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	for _, errno := range connectionErrnos {
		if errors.Is(err, errno) {
			return true
		}
	}
	return false
}
