package processor_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// errUnclassifiable is a failure nothing can attribute.
var errUnclassifiable = errors.New("nope")

// rejection is the error a provider raises when it refuses a request, standing
// for the shapes an HTTP or websocket library raises.
type rejection struct{ status int }

func (e *rejection) Error() string       { return "refused" }
func (e *rejection) HTTPStatusCode() int { return e.status }

// reporting is a processor that reports a given error whenever it sees a text
// frame, and forwards everything else.
type reporting struct {
	*processor.Base
	err      error
	category errs.Category
	// permanent asks for the error to be reported as one that will keep
	// recurring, whatever its category says.
	permanent bool
	// classify, when set, is what this processor's own classification returns.
	classify func(error) errs.Category
}

func newReporting(err error) *reporting {
	r := &reporting{err: err}
	r.Base = processor.New("Reporting", r)
	return r
}

func (r *reporting) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); !ok {
		return r.PushFrame(ctx, f, dir)
	}
	var opts []processor.ErrorOption
	if r.category != errs.Unset {
		opts = append(opts, processor.WithErrorCategory(r.category))
	}
	if r.permanent {
		opts = append(opts, processor.TreatAsPermanent())
	}
	r.PushError(ctx, "service failed", r.err, false, opts...)
	return nil
}

// ClassifyError implements processor.ErrorClassifier when the test asked for it.
func (r *reporting) ClassifyError(err error) errs.Category {
	if r.classify == nil {
		return errs.Unset
	}
	return r.classify(err)
}

// reportOnce runs p over one text frame and returns the error frames that
// reached the processor above it.
func reportOnce(t *testing.T, p processor.Processor) []*frames.ErrorFrame {
	t.Helper()
	up, _ := linkAndStart(t, p)
	if err := p.QueueFrame(context.Background(), frames.NewTextFrame("hello"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	return waitForErrors(t, up, 1)
}

// waitForErrors drains up until it has seen want error frames, or gives up.
func waitForErrors(t *testing.T, up *capture, want int) []*frames.ErrorFrame {
	t.Helper()
	var got []*frames.ErrorFrame
	deadline := time.After(2 * time.Second)
	for len(got) < want {
		select {
		case f := <-up.got:
			if ef, ok := f.(frames.ErrorReport); ok {
				got = append(got, ef.ErrorInfo())
			}
		case <-deadline:
			t.Fatalf("got %d error frames, want %d", len(got), want)
		}
	}
	return got
}

func TestProcessorsStartUsable(t *testing.T) {
	if !newReporting(nil).Usable() {
		t.Error("a processor should start able to do its job")
	}
}

func TestRejectedCredentialsMakeTheServiceUnusable(t *testing.T) {
	p := newReporting(&rejection{status: 401})

	got := reportOnce(t, p)

	if got[0].Category != errs.Authentication {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Authentication)
	}
	if p.Usable() {
		t.Error("rejected credentials should cost the service its usability")
	}
}

func TestServerErrorsLeaveTheServiceUsable(t *testing.T) {
	p := newReporting(&rejection{status: 503})

	got := reportOnce(t, p)

	if got[0].Category != errs.Server {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Server)
	}
	if !p.Usable() {
		t.Error("a provider having a bad moment should not write the service off")
	}
}

func TestUnclassifiableErrorsLeaveTheServiceUsable(t *testing.T) {
	p := newReporting(errUnclassifiable)

	got := reportOnce(t, p)

	if got[0].Category != errs.Unknown {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Unknown)
	}
	if !p.Usable() {
		t.Error("an unclassifiable failure should not write the service off")
	}
}

func TestErrorsWithoutAnErrorAreNotClassified(t *testing.T) {
	p := newReporting(nil)

	got := reportOnce(t, p)

	if got[0].Category != errs.Unknown {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Unknown)
	}
	if !p.Usable() {
		t.Error("a message with no error behind it should not write the service off")
	}
}

func TestServiceSpecificClassificationCanKeepAServiceUsable(t *testing.T) {
	p := newReporting(&rejection{status: 401})
	// A provider whose credentials can be rejected for a reason a reconnection
	// would clear.
	p.classify = func(error) errs.Category { return errs.Connectivity }

	got := reportOnce(t, p)

	if got[0].Category != errs.Connectivity {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Connectivity)
	}
	if !p.Usable() {
		t.Error("the service's own classification should have kept it usable")
	}
}

func TestServiceSpecificClassificationTakesPrecedence(t *testing.T) {
	p := newReporting(&rejection{status: 503})
	p.classify = func(error) errs.Category { return errs.Authorization }

	got := reportOnce(t, p)

	if got[0].Category != errs.Authorization {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Authorization)
	}
	if p.Usable() {
		t.Error("a permanent category from the service should cost it its usability")
	}
}

func TestExplicitCategoryNeedsNoOptIn(t *testing.T) {
	p := newReporting(nil)
	p.category = errs.Authentication

	got := reportOnce(t, p)

	if got[0].Category != errs.Authentication {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Authentication)
	}
	if p.Usable() {
		t.Error("an explicit permanent category should cost the service its usability")
	}
}

func TestAnErrorReportedAsTerminalNeedsNoCategory(t *testing.T) {
	p := newReporting(nil)
	p.permanent = true

	got := reportOnce(t, p)

	// Nothing is misconfigured; the service just ran out of attempts.
	if got[0].Category != errs.Unknown {
		t.Errorf("category: got %q, want %q", got[0].Category, errs.Unknown)
	}
	if p.Usable() {
		t.Error("a service that gave up retrying should not be given more work")
	}
}

func TestTheVerdictIsInBeforeTheErrorTravels(t *testing.T) {
	p := newReporting(&rejection{status: 401})
	var mu sync.Mutex
	var seen []bool
	events.On(p.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		src, ok := ef.Source.(processor.Processor)
		if !ok {
			t.Errorf("the error names %T, want the processor that reported it", ef.Source)
			return
		}
		mu.Lock()
		seen = append(seen, src.Usable())
		mu.Unlock()
	})

	reportOnce(t, p)

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 || seen[0] {
		t.Errorf("handlers saw %v, want [false]", seen)
	}
}

func TestBecomingUnusableNotifiesListeners(t *testing.T) {
	p := newReporting(&rejection{status: 401})
	changed := make(chan bool, 4)
	events.On(p.Events(), processor.EventUsableChanged, func(_ context.Context, usable bool) {
		changed <- usable
	})

	reportOnce(t, p)

	select {
	case usable := <-changed:
		if usable {
			t.Error("the change reported usable, want no longer usable")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("losing usability was never announced")
	}
}

func TestRepeatedErrorsNotifyOnce(t *testing.T) {
	p := newReporting(&rejection{status: 401})
	changed := make(chan bool, 8)
	events.On(p.Events(), processor.EventUsableChanged, func(_ context.Context, usable bool) {
		changed <- usable
	})

	up, _ := linkAndStart(t, p)
	ctx := context.Background()
	for _, text := range []string{"one", "two", "three"} {
		if err := p.QueueFrame(ctx, frames.NewTextFrame(text), processor.Downstream); err != nil {
			t.Fatal(err)
		}
	}
	waitForErrors(t, up, 3)

	// Give any further announcement time to arrive before counting.
	time.Sleep(100 * time.Millisecond)
	if got := len(changed); got != 1 {
		t.Errorf("announced %d times, want 1", got)
	}
}

func TestAProcessorCanBeBroughtBack(t *testing.T) {
	p := newReporting(nil)
	ctx := context.Background()

	p.SetUsable(ctx, false)
	p.SetUsable(ctx, true)

	if !p.Usable() {
		t.Error("a processor brought back should be able to take work again")
	}
}

func TestAnErrorFrameKeepsTheCategoryItWasGiven(t *testing.T) {
	p := newReporting(nil)
	ctx := context.Background()
	ef := frames.NewErrorFrame("boom")
	ef.Category = errs.Application
	ef.Err = &rejection{status: 401}

	p.PushErrorFrame(ctx, ef, false)

	// The category the caller set stands: the exception is not consulted, so
	// application code that failed cannot cost the service its usability.
	if ef.Category != errs.Application {
		t.Errorf("category: got %q, want %q", ef.Category, errs.Application)
	}
	if !p.Usable() {
		t.Error("an application failure should leave the service usable")
	}
}

func TestErrorFrameStringOmitsAnUnsetCategory(t *testing.T) {
	if got := frames.NewErrorFrame("boom").String(); strings.Contains(got, "category") {
		t.Errorf("got %q, want no category", got)
	}
}

func TestErrorFrameStringOmitsAnUnknownCategory(t *testing.T) {
	ef := frames.NewErrorFrame("boom")
	ef.Category = errs.Unknown
	if got := ef.String(); strings.Contains(got, "category") {
		t.Errorf("got %q, want no category", got)
	}
}

func TestErrorFrameStringIncludesAKnownCategory(t *testing.T) {
	ef := frames.NewErrorFrame("boom")
	ef.Category = errs.Authentication
	if got := ef.String(); !strings.Contains(got, "category: authentication") {
		t.Errorf("got %q, want the category named", got)
	}
}
