package processor_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// TestBeforeAndAfterFrameEvents checks the four events a processor raises around
// each frame it handles and each frame it sends on. They are what lets something
// watch one processor without sitting in the pipeline or observing all of it.
func TestBeforeAndAfterFrameEvents(t *testing.T) {
	ctx := t.Context()
	p := processor.NewIdentityFilter("Identity")
	down := newSink("Down")

	var mu sync.Mutex
	fired := map[string][]string{}
	record := func(name string) events.Handler {
		return func(_ context.Context, _ any, args ...any) {
			f, ok := args[0].(frames.Frame)
			if !ok {
				t.Errorf("%s handler got %T, want a frame", name, args[0])
				return
			}
			mu.Lock()
			fired[name] = append(fired[name], f.Name())
			mu.Unlock()
		}
	}
	for _, name := range []string{
		processor.EventBeforeProcessFrame, processor.EventAfterProcessFrame,
		processor.EventBeforePushFrame, processor.EventAfterPushFrame,
	} {
		if id := p.Events().Add(name, record(name)); id == 0 {
			t.Fatalf("%s is not a registered event", name)
		}
	}

	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, p, down)
	queue(t, ctx, p, frames.NewStartFrame())

	text := frames.NewTextFrame("hello")
	queue(t, ctx, p, text)

	if !waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(fired[processor.EventAfterPushFrame]) == 2
	}) {
		t.Fatalf("events fired: %v", fired)
	}

	mu.Lock()
	defer mu.Unlock()
	// The StartFrame and the text frame, each handled and each sent on.
	want := []string{"StartFrame", "TextFrame"}
	for name, got := range fired {
		if len(got) != 2 {
			t.Errorf("%s fired %d times (%v), want twice", name, len(got), got)
			continue
		}
		for i := range want {
			if got[i][:len(want[i])] != want[i] {
				t.Errorf("%s fired for %v, want %v", name, got, want)
				break
			}
		}
	}
}

// TestAfterProcessFrameDoesNotFireOnFailure checks that a frame whose handling
// failed does not report as handled. The failure is reported as an error
// instead, so a handler counting what got through is not told it did.
func TestAfterProcessFrameDoesNotFireOnFailure(t *testing.T) {
	ctx := t.Context()
	p := newFailer()
	up := newSink("Up")
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, up, p)

	var mu sync.Mutex
	after, errored := 0, 0
	p.Events().Add(processor.EventAfterProcessFrame,
		func(context.Context, any, ...any) { mu.Lock(); after++; mu.Unlock() })
	p.Events().Add(processor.EventError,
		func(context.Context, any, ...any) { mu.Lock(); errored++; mu.Unlock() })

	queue(t, ctx, up, frames.NewStartFrame())
	queue(t, ctx, up, frames.NewTextFrame("boom"))

	if !waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return errored == 1 }) {
		t.Fatal("the failure was never reported")
	}
	mu.Lock()
	defer mu.Unlock()
	// Only the StartFrame was handled without failing.
	if after != 1 {
		t.Errorf("after-process fired %d times, want once (the StartFrame alone)", after)
	}
}

// errBoom is what the failing processor below returns.
var errBoom = errors.New("boom")

// failer fails on every text frame.
type failer struct{ *processor.Base }

func newFailer() *failer {
	p := &failer{}
	p.Base = processor.New("Failer", p)
	return p
}

func (p *failer) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		return errBoom
	}
	return p.PushFrame(ctx, f, dir)
}
