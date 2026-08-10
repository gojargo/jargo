package flows

import "errors"

// The errors a flow reports. They form a hierarchy: every one of them wraps
// ErrFlow, so a caller that only wants to know the flow failed matches on that,
// and one that wants to know how matches on the specific error.
//
//nolint:gochecknoglobals // sentinel errors
var (
	// ErrFlow is the error every flow failure wraps. Match it to catch them all.
	ErrFlow = errors.New("flows: flow error")

	// ErrFlowInitialization reports that the flow could not be entered:
	// its opening node failed to be set up.
	ErrFlowInitialization = errors.New("flows: initialization failed")

	// ErrFlowTransition reports that a transition could not be made, which is
	// almost always a manager that was asked to move before it was initialized.
	ErrFlowTransition = errors.New("flows: transition failed")

	// ErrInvalidFunction reports a function a node cannot offer: one with no
	// name, no handler, or a shape the manager cannot call.
	ErrInvalidFunction = errors.New("flows: invalid function")

	// ErrAction reports that an action failed to run.
	ErrAction = errors.New("flows: action failed")
)

// flowError joins one of the specific errors above to ErrFlow, so a caller may
// match either. Every error the package returns is built through here.
//
// A failure reported at a boundary keeps the failure underneath it as its cause,
// so a caller may match the outer kind or what actually went wrong: a bad action
// that stops initialization matches both ErrFlowInitialization and ErrAction.
type flowError struct {
	kind  error
	msg   string
	cause error
}

// Error implements error.
func (e *flowError) Error() string {
	parts := e.kind.Error()
	if e.msg != "" {
		parts += ": " + e.msg
	}
	if e.cause != nil {
		parts += ": " + e.cause.Error()
	}
	return parts
}

// Is reports the error as both its own kind and a flow error.
func (e *flowError) Is(target error) bool {
	return target == e.kind || target == ErrFlow //nolint:errorlint // sentinel identity
}

// Unwrap exposes the cause, so matching continues down to what actually failed.
func (e *flowError) Unwrap() error { return e.cause }

// errf builds a flow error of the given kind carrying msg.
func errf(kind error, msg string) error { return &flowError{kind: kind, msg: msg} }

// wrapErr reports cause as a failure of the given kind, keeping cause matchable.
func wrapErr(kind error, cause error) error { return &flowError{kind: kind, cause: cause} }
