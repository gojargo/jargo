package processor

import (
	"context"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// The events every processor raises.
const (
	// EventUsableChanged fires with the new value of Usable whenever a
	// processor stops or starts being able to do its job.
	//
	//	events.On(p.Events(), processor.EventUsableChanged,
	//	    func(ctx context.Context, usable bool) { … })
	EventUsableChanged = "on_usable_changed"
	// EventError fires with the error frame a processor reports, before the
	// frame travels. A handler reading Source.Usable() therefore sees the
	// verdict that came with the error it is handling.
	EventError = "on_error"
	// EventBeforeProcessFrame fires with a frame this processor is about to
	// handle, before it has been handled.
	EventBeforeProcessFrame = "on_before_process_frame"
	// EventAfterProcessFrame fires with a frame this processor has handled. It
	// does not fire for a frame whose handling failed, which raises EventError
	// instead.
	EventAfterProcessFrame = "on_after_process_frame"
	// EventBeforePushFrame fires with a frame this processor is about to send to
	// a neighbor, before the neighbor has it.
	EventBeforePushFrame = "on_before_push_frame"
	// EventAfterPushFrame fires with a frame this processor has sent to a
	// neighbor.
	EventAfterPushFrame = "on_after_push_frame"
)

// usability is the part of a Base that tracks whether it can still work. It is
// separate so a processor embedding Base gets it without the field noise.
type usability struct {
	mu     sync.Mutex
	usable bool
}

// ErrorClassifier is implemented by a processor that knows the shape of the
// failures its provider raises, so it can say what one means where the shared
// classification cannot.
//
// Implement it on a service whose provider signals failures through errors of
// its own rather than an HTTP status, or whose credentials can be rejected for a
// reason a reconnection would clear. Returning the zero category falls back to
// the shared classification.
type ErrorClassifier interface {
	// ClassifyError returns the category of err, or the zero category to let
	// the shared classification decide.
	ClassifyError(err error) errs.Category
}

// Events returns the registry of events this processor raises, for attaching
// handlers to them.
func (b *Base) Events() *events.Registry { return &b.events }

// Usable reports whether this processor can still do its job.
//
// A processor stays usable through failures it might recover from, and becomes
// unusable once its work can no longer succeed: a provider has rejected its API
// key, model or voice, or it has failed enough times to stop trying. Sending it
// more work would only produce more of the same error, so services stop
// accepting work and stop reconnecting once this is false.
//
// Errors set this as they are reported, so an error handler reading
// ErrorFrame.Source.Usable() sees the verdict that came with the error it is
// handling.
func (b *Base) Usable() bool {
	b.usability.mu.Lock()
	defer b.usability.mu.Unlock()
	return b.usability.usable
}

// SetUsable sets whether this processor can be given work, raising
// EventUsableChanged when the value moves.
//
// Call it to bring back a processor that became unusable, once whatever stopped
// it working has been dealt with: new credentials, or a provider that has come
// back up. Services also do this for themselves when their settings change,
// since new settings may be the fix.
func (b *Base) SetUsable(ctx context.Context, usable bool) {
	b.usability.mu.Lock()
	if usable == b.usability.usable {
		b.usability.mu.Unlock()
		return
	}
	b.usability.usable = usable
	b.usability.mu.Unlock()

	if usable {
		slog.DebugContext(ctx, "processor usable", "processor", b.name)
	} else {
		slog.DebugContext(ctx, "processor no longer usable", "processor", b.name)
	}
	b.events.Call(ctx, EventUsableChanged, b.self, usable)
}

// ErrorOption adjusts what PushError reports beyond the message. The options
// stand for the arguments upstream passes by keyword, so an ordinary failure
// stays a four-argument call.
type ErrorOption func(*errorOptions)

// errorOptions collects what the options set.
type errorOptions struct {
	category              errs.Category
	forceTreatAsPermanent bool
}

// WithErrorCategory reports why the error occurred, when the caller knows.
// Leaving it unset lets the category be worked out from the error; passing
// errors.Unknown reports a failure whose cause cannot be attributed, an
// unexpected one caught by a catch-all, say, which may not have come from this
// processor at all.
func WithErrorCategory(c errs.Category) ErrorOption {
	return func(o *errorOptions) { o.category = c }
}

// ForceTreatAsPermanent reports the error as one that will keep recurring,
// leaving the processor unable to do any more work: having failed too many times
// to keep trying, say. It is only needed for a failure the category does not
// already convey, since a permanent category costs the processor its usability
// on its own.
func ForceTreatAsPermanent() ErrorOption {
	return func(o *errorOptions) { o.forceTreatAsPermanent = true }
}

// PushError builds an ErrorFrame for msg and pushes it upstream.
func (b *Base) PushError(ctx context.Context, msg string, err error, fatal bool, opts ...ErrorOption) {
	var o errorOptions
	for _, opt := range opts {
		opt(&o)
	}
	ef := frames.NewErrorFrame(msg)
	ef.Fatal = fatal
	ef.Err = err
	ef.Source = b.self
	ef.Category = o.category
	b.self.PushErrorFrame(ctx, ef, o.forceTreatAsPermanent)
}

// PushErrorFrame settles the error frame's category and the processor's
// usability, tells the error handlers, and pushes the frame upstream.
//
// forceTreatAsPermanent reports the error as one that will keep recurring. Leaving it
// false does not keep the processor usable: a permanent category costs it its
// usability either way.
func (b *Base) PushErrorFrame(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool) {
	if ef.Source == nil {
		ef.Source = b.self
	}
	// Anything still unset by now is going to stay that way, so settle it here
	// and let handlers read a category off every error they receive.
	if ef.Category == errs.Unset && ef.Err != nil {
		if classifier, ok := b.self.(ErrorClassifier); ok {
			ef.Category = classifier.ClassifyError(ef.Err)
		}
		if ef.Category == errs.Unset {
			ef.Category = errs.ClassifyError(ef.Err)
		}
	}
	if ef.Category == errs.Unset {
		ef.Category = errs.Unknown
	}

	// Before anything sees the error, so that handlers reading
	// ErrorFrame.Source.Usable() get the verdict that came with it.
	if forceTreatAsPermanent || ef.Category.IsPermanent() {
		b.self.SetUsable(ctx, false)
	}

	b.events.Call(ctx, EventError, b.self, ef)

	slog.ErrorContext(ctx, "processor error", "processor", b.name, "msg", ef.Error,
		"err", ef.Err, "fatal", ef.Fatal, "category", ef.Category)
	_ = b.self.PushFrame(ctx, ef, Upstream)
}
