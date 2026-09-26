package observers

import (
	"fmt"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
)

// ErrorEvent is one failure, as the processor that raised it described it.
type ErrorEvent struct {
	// Message is what went wrong, in the words of the processor that failed.
	Message string
	// Category is why it failed, drawn from [errors.Category] and independent of
	// the provider that failed: rejected credentials, an unreachable service, a
	// malformed request and so on.
	Category errs.Category
	// ExceptionType is the type of the error behind the failure, where one
	// caused it, and "" where none did. Failures group by this where a message,
	// carrying the particulars of a single occurrence, is too specific to group
	// by.
	ExceptionType string
	// Processor is the name of the processor that raised the error.
	Processor string
	// ProcessorUsable reports whether that processor can still do its job. A
	// processor that cannot keeps failing for as long as it is given work, so
	// this separates a bad minute from the end of a capability.
	ProcessorUsable bool
	// Timestamp is the wall-clock time of the failure.
	Timestamp time.Time
}

// ErrorConfig configures an Errors observer.
type ErrorConfig struct {
	// MaxFrames is unused.
	//
	// Deprecated: the observer is told about each frame once, so it keeps no
	// window of the frames it has seen.
	MaxFrames int
	// Now reads the current time. Nil uses time.Now. Supplying one lets a test
	// place failures without waiting.
	Now func() time.Time
	// OnError is called for each error, as an ErrorEvent.
	OnError func(e ErrorEvent)
}

// Errors reports each error a pipeline raises, once, where it is raised.
//
// Errors travel upstream from the processor that raised them, and not all of
// them reach the end of that journey: a processor that answers for a failure
// itself (a service switcher that fails over to its next service, say) stops the
// error there. This observer reads each error where it is raised, so a session's
// failure history holds the ones that were recovered from as well as the ones
// that surfaced.
//
// An error is reported at its origin rather than where it ends up, and named for
// the processor that raised it rather than the one that passed it along.
type Errors struct {
	cfg ErrorConfig

	mu sync.Mutex
}

// NewErrors builds an Errors observer.
func NewErrors(cfg ErrorConfig) *Errors {
	return &Errors{cfg: cfg}
}

// ObserveEveryPush implements processor.EveryPushObserver: an error is reported
// once, on its first push.
func (o *Errors) ObserveEveryPush() bool { return false }

// now reads the clock the observer was configured with.
func (o *Errors) now() time.Time {
	if o.cfg.Now != nil {
		return o.cfg.Now()
	}
	return time.Now()
}

// OnPushFrame implements processor.Observer. It reports an error frame. The
// first push of an error comes from the processor that failed.
func (o *Errors) OnPushFrame(data processor.FramePushed) {
	// Match on the reporting interface rather than on ErrorFrame, so an
	// unrecoverable failure, which is reported by a frame embedding it, is not
	// missed.
	report, ok := data.Frame.(frames.ErrorReport)
	if !ok {
		return
	}

	o.mu.Lock()
	at := o.now()
	o.mu.Unlock()

	ef := report.ErrorInfo()

	// An error assembled by hand rather than reported through PushError arrives
	// without the processor that settles it, so attribute it to the processor
	// pushing it.
	var source frames.ErrorSource = data.Source
	if ef.Source != nil {
		source = ef.Source
	}

	// A category is settled by the time the frame travels, but an error frame
	// pushed without going through PushErrorFrame can still carry none.
	category := ef.Category
	if category == errs.Unset {
		category = errs.Unknown
	}

	exceptionType := ""
	if ef.Err != nil {
		exceptionType = fmt.Sprintf("%T", ef.Err)
	}

	if o.cfg.OnError == nil {
		return
	}
	o.cfg.OnError(ErrorEvent{
		Message:         ef.Error,
		Category:        category,
		ExceptionType:   exceptionType,
		Processor:       source.Name(),
		ProcessorUsable: source.Usable(),
		Timestamp:       at,
	})
}

// Compile-time interface check.
var _ processor.Observer = (*Errors)(nil)
