package realtime

import (
	"fmt"
	"strconv"
	"time"
)

// ErrorKind is what a Rime failure means, the way Rime's own client sorts its
// failures.
type ErrorKind int

// The kinds of failure.
const (
	// KindStream is a broken or invalid stream: a lost connection, an event
	// that does not follow the protocol.
	KindStream ErrorKind = iota
	// KindAuthentication is rejected credentials.
	KindAuthentication
	// KindBilling is an account with no credits left.
	KindBilling
	// KindPermission is credentials that do not grant access.
	KindPermission
	// KindInput is a request the client refused to send, or a session used
	// in a way it does not allow.
	KindInput
	// KindResourceLimit is a bound reached: a rate limit, or one of the
	// client's own retention limits.
	KindResourceLimit
	// KindUnavailable is an endpoint that could not be reached or refused the
	// connection.
	KindUnavailable
	// KindTimeout is a deadline that passed.
	KindTimeout
	// KindAudioFormat is audio Rime does not take.
	KindAudioFormat
)

// String names the kind as Rime's own client names the failure.
func (k ErrorKind) String() string {
	switch k {
	case KindAuthentication:
		return "RimeAuthenticationError"
	case KindBilling:
		return "RimeBillingError"
	case KindPermission:
		return "RimePermissionError"
	case KindInput:
		return "RimeInputError"
	case KindResourceLimit:
		return "RimeResourceLimitError"
	case KindUnavailable:
		return "RimeUnavailableError"
	case KindTimeout:
		return "RimeTimeoutError"
	case KindAudioFormat:
		return "RimeAudioFormatError"
	case KindStream:
		return "RimeStreamError"
	default:
		return "RimeStreamError"
	}
}

// Error is a failure of the Prism client, with the connection diagnostics that
// are known. The diagnostics cause no retry, and do not make replaying a
// request that was accepted safe.
type Error struct {
	// Kind is what the failure means.
	Kind ErrorKind
	// Message describes the failure.
	Message string
	// RequestID is the identity of the request that failed, when it has one.
	RequestID string
	// StatusCode is the HTTP status a refused upgrade was refused with, 0 when
	// there is none.
	StatusCode int
	// CloseCode is the WebSocket close code received, 0 when there is none.
	CloseCode int
	// RetryAfter is the delay the server asked for, zero included; nil when it
	// asked for none.
	RetryAfter *time.Duration
	// ServerKind is the service's structured reason, such as
	// "insufficient_credits"; empty when it gave none.
	ServerKind string
}

// Error implements error.
func (e *Error) Error() string { return e.Message }

// HTTPStatusCode is the status a refused upgrade was refused with, which the
// shared error classification reads.
func (e *Error) HTTPStatusCode() int { return e.StatusCode }

func newError(kind ErrorKind, message string) *Error {
	return &Error{Kind: kind, Message: message}
}

func inputError(message string) *Error { return newError(KindInput, message) }

func streamError(message string) *Error { return newError(KindStream, message) }

func resourceLimitError(message string) *Error { return newError(KindResourceLimit, message) }

// invalidField is the failure for a server event carrying a field that does
// not follow the protocol.
func invalidField(name string) *Error {
	return streamError("Invalid realtime field: " + name)
}

// AdmissionTimeoutError reports that Prism did not become ready for another
// reply in time. No request was outstanding, so the session remains usable.
type AdmissionTimeoutError struct {
	// Message describes what was waited for.
	Message string
}

// Error implements error.
func (e *AdmissionTimeoutError) Error() string { return e.Message }

// IdleTimeoutError reports that Prism closed an idle conversation.
type IdleTimeoutError struct {
	// IdleTimeout is how long the conversation was idle for.
	IdleTimeout time.Duration
}

// Error implements error.
func (e *IdleTimeoutError) Error() string {
	seconds := strconv.FormatFloat(e.IdleTimeout.Seconds(), 'g', 6, 64)
	return "Realtime session idle for " + seconds + " seconds"
}

// Fault is a failure Prism reported in an error event.
type Fault struct {
	// Code is Prism's code for the failure.
	Code string
	// Message describes the failure.
	Message string
	// Scope is what the failure ended: "event", "utterance", "response",
	// "tool_roundtrip", "review", "session" or "unknown".
	Scope string
	// RequestID is the event id of the request the failure answers, empty for
	// a failure no request caused.
	RequestID string
	// ResponseID is the response the failure belongs to, when it has one.
	ResponseID string
	// ReviewID is the reviewed chunk the failure belongs to, when it has one.
	ReviewID string
	// ItemID is the conversation item the failure belongs to, when it has one.
	ItemID string
	// CallIDs are the tool calls the failure belongs to.
	CallIDs []string
	// CorrelationID is Prism's identifier for the failure in its own logs.
	CorrelationID string
	// Parameter names the request field at fault, when one is.
	Parameter string
}

// The scopes of a fault the client acts on.
const (
	scopeEvent   = "event"
	scopeSession = "session"
	scopeUnknown = "unknown"
)

// Retryable reports whether the same request, unchanged, may succeed on
// another attempt: only for the codes capacity_exceeded, unavailable and
// response_timeout. Nothing is replayed on its account.
func (f Fault) Retryable() bool {
	switch f.Code {
	case "capacity_exceeded", "unavailable", "response_timeout":
		return true
	default:
		return false
	}
}

// FaultError is a failure Prism reported.
type FaultError struct {
	// Fault is what Prism reported.
	Fault Fault
}

// Error implements error.
func (e *FaultError) Error() string { return e.Fault.Code + ": " + e.Fault.Message }

// HistoryUpdateError is a rejected, partially applied or uncertain write to
// the conversation's history. Its message leaves the conversation's text out.
type HistoryUpdateError struct {
	// Status is "rejected" when nothing was written, "partial" when some of
	// the items were, or "uncertain" when the outcome is unknown.
	Status string
	// Err is the failure that ended the write.
	Err error
}

// The statuses of a failed history write.
const (
	historyRejected  = "rejected"
	historyPartial   = "partial"
	historyUncertain = "uncertain"
)

// Error implements error.
func (e *HistoryUpdateError) Error() string { return "Prism history update " + e.Status }

// Unwrap returns the failure that ended the write.
func (e *HistoryUpdateError) Unwrap() error { return e.Err }

// Recoverable reports whether the conversation can continue.
func (e *HistoryUpdateError) Recoverable() bool { return e.Status != historyUncertain }

// SettingsError is a setting or a request Prism does not take, refused before
// anything was sent.
type SettingsError struct {
	// Message says what was refused.
	Message string
}

// Error implements error.
func (e *SettingsError) Error() string { return e.Message }

func settingsError(format string, args ...any) *SettingsError {
	return &SettingsError{Message: fmt.Sprintf(format, args...)}
}

// failureError is a failure of the service itself, which ends the
// conversation.
type failureError struct{ message string }

func (e *failureError) Error() string { return e.message }
