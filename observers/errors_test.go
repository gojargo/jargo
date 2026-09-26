package observers_test

import (
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
)

// errorRecorder collects everything an Errors observer reports.
type errorRecorder struct {
	mu     sync.Mutex
	events []observers.ErrorEvent
}

func (r *errorRecorder) record(e observers.ErrorEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

func (r *errorRecorder) all() []observers.ErrorEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]observers.ErrorEvent(nil), r.events...)
}

// only returns the single event recorded, failing when there is not exactly one.
func (r *errorRecorder) only(t *testing.T) observers.ErrorEvent {
	t.Helper()
	all := r.all()
	if len(all) != 1 {
		t.Fatalf("events = %d, want 1", len(all))
	}
	return all[0]
}

// pushFrom reports one frame to an observer as a handover from source.
func pushFrom(o processor.Observer, f frames.Frame, source processor.Processor) {
	o.OnPushFrame(processor.FramePushed{Frame: f, Source: source, Direction: processor.Upstream})
}

// errNoRoute stands in for the failure a provider's client reports when it
// cannot be reached.
//
//nolint:gochecknoglobals // sentinel error
var errNoRoute = errors.New("no route to host")

// raised builds the error frame a processor's PushError would produce: the
// source and the category are settled by the time the frame travels.
func raised(msg string, source frames.ErrorSource, err error, category errs.Category) *frames.ErrorFrame {
	ef := frames.NewErrorFrame(msg)
	ef.Source = source
	ef.Err = err
	ef.Category = category
	return ef
}

// TestEachErrorIsItsOwnEvent covers a processor that fails twice having failed
// twice, which the deduplication must not flatten into one.
func TestEachErrorIsItsOwnEvent(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	failing := processor.NewIdentityFilter("tts")
	pushFrom(o, raised("first", failing, nil, errs.Server), failing)
	pushFrom(o, raised("second", failing, nil, errs.Server), failing)

	all := r.all()
	if len(all) != 2 || all[0].Message != "first" || all[1].Message != "second" {
		t.Fatalf("messages = %v, want [first second]", all)
	}
}

// TestErrorReportsWhenItHappened covers the configured clock being what dates a
// failure, so a test can place one without waiting.
func TestErrorReportsWhenItHappened(t *testing.T) {
	var r errorRecorder
	at := time.Date(2026, time.September, 13, 10, 0, 0, 0, time.UTC)
	o := observers.NewErrors(observers.ErrorConfig{
		Now:     func() time.Time { return at },
		OnError: r.record,
	})

	pushFrom(o, frames.NewErrorFrame("failed"), processor.NewIdentityFilter("llm"))

	if got := r.only(t).Timestamp; !got.Equal(at) {
		t.Errorf("timestamp = %v, want %v", got, at)
	}
}

// TestErrorAssembledByHandIsAttributedToItsPusher covers the frame that did not
// go through PushError: it carries neither a source nor a category, so the
// observer attributes it to the processor pushing it and reports its cause as
// unknown rather than leaving either blank.
func TestErrorAssembledByHandIsAttributedToItsPusher(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	pusher := processor.NewIdentityFilter("llm")
	pushFrom(o, frames.NewErrorFrame("failed"), pusher)

	e := r.only(t)
	if e.Processor != pusher.Name() {
		t.Errorf("processor = %q, want %q", e.Processor, pusher.Name())
	}
	if e.Category != errs.Unknown {
		t.Errorf("category = %q, want %q", e.Category, errs.Unknown)
	}
	if e.ExceptionType != "" {
		t.Errorf("exception type = %q, want empty: no error caused this one", e.ExceptionType)
	}
}

// TestErrorReportsTheTypeBehindIt covers grouping failures by what went wrong
// rather than by a message carrying the particulars of one occurrence.
func TestErrorReportsTheTypeBehindIt(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	failing := processor.NewIdentityFilter("stt")
	err := &net.OpError{Op: "dial", Err: errNoRoute}
	pushFrom(o, raised("the provider said no", failing, err, errs.Connectivity), failing)

	e := r.only(t)
	if e.ExceptionType != "*net.OpError" {
		t.Errorf("exception type = %q, want *net.OpError", e.ExceptionType)
	}
	if e.Category != errs.Connectivity {
		t.Errorf("category = %q, want %q", e.Category, errs.Connectivity)
	}
}

// TestRecoverableFailureLeavesTheProcessorUsable covers the distinction the
// event carries: an unreachable service may well answer the next request.
func TestRecoverableFailureLeavesTheProcessorUsable(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	failing := processor.NewIdentityFilter("stt")
	pushFrom(o, raised("no route to host", failing, nil, errs.Connectivity), failing)

	if !r.only(t).ProcessorUsable {
		t.Error("processor usable = false, want true: connectivity is recoverable")
	}
}

// TestPermanentFailureCostsTheProcessorItsUsability covers the other half:
// rejected credentials stay rejected, so the capability is gone rather than
// merely having had a bad minute. Usability is settled before the frame travels,
// so the observer reads the verdict that came with the error.
func TestPermanentFailureCostsTheProcessorItsUsability(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	failing := processor.NewIdentityFilter("stt")
	failing.SetUsable(t.Context(), false)
	pushFrom(o, raised("invalid api key", failing, nil, errs.Authentication), failing)

	if r.only(t).ProcessorUsable {
		t.Error("processor usable = true, want false: rejected credentials stay rejected")
	}
}

// TestFatalErrorIsReported covers the frame that reports an unrecoverable
// failure. It embeds ErrorFrame rather than being one, so an observer matching
// on the concrete type would miss the worst failures a pipeline has.
func TestFatalErrorIsReported(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	failing := processor.NewIdentityFilter("llm")
	ff := frames.NewFatalErrorFrame("the session cannot continue")
	ff.Source = failing
	ff.Category = errs.Server
	pushFrom(o, ff, failing)

	if got := r.only(t).Message; got != "the session cannot continue" {
		t.Errorf("message = %q, want the fatal error's", got)
	}
}

// TestNonErrorFramesAreNotReported covers the observer ignoring the rest of the
// stream.
func TestNonErrorFramesAreNotReported(t *testing.T) {
	var r errorRecorder
	o := observers.NewErrors(observers.ErrorConfig{OnError: r.record})

	pushFrom(o, frames.NewTextFrame("hello"), processor.NewIdentityFilter("llm"))

	if got := r.all(); len(got) != 0 {
		t.Errorf("events = %v, want none", got)
	}
}
