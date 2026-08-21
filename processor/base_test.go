package processor_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Tests for what the processor base gives every processor built on it: the
// pipeline it is wired into, the flags the StartFrame carries, and the pushes a
// processor makes that are not just forwarding the frame it was handed.

// flusher stands in for the task a processor belongs to and counts the flushes
// asked of it.
type flusher struct {
	mu    sync.Mutex
	calls int
}

func (f *flusher) Flush(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	return nil
}

// TurnTracker reports no tracker: this stand-in is a pipeline only as far as
// flushing goes.
func (f *flusher) TurnTracker() processor.TurnTracker { return nil }

func (f *flusher) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// holder blocks on the first TextFrame until it is released, so a test can look
// at what is still waiting in the queue behind it.
type holder struct {
	*processor.Base
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func newHolder() *holder {
	h := &holder{entered: make(chan struct{}), release: make(chan struct{})}
	h.Base = processor.New("Holder", h)
	return h
}

func (h *holder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := h.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		h.once.Do(func() { close(h.entered) })
		select {
		case <-h.release:
		case <-ctx.Done():
		}
		return nil
	}
	return h.PushFrame(ctx, f, dir)
}

// linkChain wires processors in order and sets them all up.
func linkChain(t *testing.T, ctx context.Context, s processor.Setup, ps ...processor.Processor) {
	t.Helper()
	for i := range ps[:len(ps)-1] {
		ps[i].Link(ps[i+1])
	}
	for _, p := range ps {
		if err := p.Setup(ctx, s); err != nil {
			t.Fatalf("setting up %s: %v", p.Name(), err)
		}
		t.Cleanup(func() { _ = p.Cleanup(ctx) })
	}
}

func TestDirectionString(t *testing.T) {
	if got := processor.Downstream.String(); got != "downstream" {
		t.Errorf("Downstream.String() = %q, want downstream", got)
	}
	if got := processor.Upstream.String(); got != "upstream" {
		t.Errorf("Upstream.String() = %q, want upstream", got)
	}
}

// TestBaseIdentityAndLinks checks the identity a processor carries and the
// neighbors it is linked to. Name appends an instance number, so two processors
// of the same kind are told apart on a span or in a log while TypeName still
// names the kind.
func TestBaseIdentityAndLinks(t *testing.T) {
	first, second := newEcho(), newEcho()

	if first.ID() == second.ID() {
		t.Errorf("two processors share the id %d", first.ID())
	}
	if first.TypeName() != "Echo" {
		t.Errorf("TypeName() = %q, want Echo", first.TypeName())
	}
	if first.Name() == second.Name() {
		t.Errorf("two processors share the name %q", first.Name())
	}

	if first.Next() != nil || first.Prev() != nil {
		t.Error("an unlinked processor reports a neighbor")
	}

	first.Link(second)
	if first.Next() != processor.Processor(second) {
		t.Error("Next() is not the processor that was linked")
	}
	if second.Prev() != processor.Processor(first) {
		t.Error("Link did not set the previous processor on the far side")
	}

	// A plain processor contains nothing and measures nothing; the compound
	// processors override these.
	if first.Processors() != nil || first.EntryProcessors() != nil {
		t.Error("a plain processor reports contained processors")
	}
	if first.ProcessorsWithMetrics() != nil {
		t.Error("a plain processor reports processors with metrics")
	}
	if first.CanGenerateMetrics() {
		t.Error("a plain processor claims it generates metrics")
	}

	// Self is the concrete value passed to New, so a push goes through whatever
	// the outer type does on its way out.
	if first.Self() != processor.Processor(first) {
		t.Error("Self() is not the processor the base belongs to")
	}
}

// TestBaseSetupState checks the shared components Setup hands down and the flags
// the StartFrame carries, which a service reads before it measures anything.
func TestBaseSetupState(t *testing.T) {
	e := newEcho()
	c := newCapture()

	ctx := context.Background()
	clk := clock.NewSystem()
	fl := &flusher{}
	linkChain(t, ctx, processor.Setup{Clock: clk, Running: fl}, e, c)

	if e.Clock() != clk {
		t.Error("Clock() is not the clock Setup was given")
	}

	// An untraced pipeline has no tracing state, and a span opened on one is a
	// no-op that records nothing.
	if e.TracingEnabled() {
		t.Error("an untraced pipeline reports tracing enabled")
	}
	if e.Tracing() != nil {
		t.Error("an untraced pipeline carries tracing state")
	}
	spanCtx, span := e.StartSpan(ctx, "work")
	if spanCtx != ctx {
		t.Error("StartSpan on an untraced pipeline returned a different context")
	}
	if span.SpanContext().IsValid() {
		t.Error("StartSpan on an untraced pipeline returned a recording span")
	}
	span.End()

	if err := e.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.StartFrame](t, c.got, "StartFrame")

	// The flush is the task's to do, and a processor outside a task has nothing
	// to drain.
	if err := e.FlushPipeline(ctx); err != nil {
		t.Fatalf("FlushPipeline: %v", err)
	}
	if fl.count() != 1 {
		t.Fatalf("the task was asked to flush %d times, want 1", fl.count())
	}

	lone := newEcho()
	if err := lone.Setup(ctx, processor.Setup{Clock: clk}); err != nil {
		t.Fatal(err)
	}
	defer lone.Cleanup(ctx)
	if err := lone.FlushPipeline(ctx); err != nil {
		t.Fatalf("flushing a processor outside a task: %v", err)
	}
}

// TestBaseBroadcast checks that a broadcast reaches processors on either side of
// the one making it, as two distinct frames that name each other, so a consumer
// seeing both halves recognizes the pair rather than reporting the event twice.
func TestBaseBroadcast(t *testing.T) {
	up, mid, down := newCapture(), newEcho(), newCapture()

	ctx := context.Background()
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, up, mid, down)

	if err := mid.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.StartFrame](t, down.got, "StartFrame")

	if err := mid.Broadcast(ctx, func() frames.Frame {
		return frames.NewTextFrame("both ways")
	}); err != nil {
		t.Fatalf("Broadcast: %v", err)
	}

	gotDown := mustReceive[*frames.TextFrame](t, down.got, "TextFrame")
	gotUp := mustReceive[*frames.TextFrame](t, up.got, "TextFrame")

	if gotDown.ID() == gotUp.ID() {
		t.Fatal("both directions were sent the same frame")
	}
	sibling, ok := gotDown.BroadcastSiblingID()
	if !ok || sibling != gotUp.ID() {
		t.Errorf("the downstream half names sibling %d, want %d", sibling, gotUp.ID())
	}
	sibling, ok = gotUp.BroadcastSiblingID()
	if !ok || sibling != gotDown.ID() {
		t.Errorf("the upstream half names sibling %d, want %d", sibling, gotDown.ID())
	}
}

// TestBasePushTokenUsage checks the usage report a realtime service makes
// outside the LLM base: it goes downstream as a MetricsFrame so an in-band
// consumer sees it.
func TestBasePushTokenUsage(t *testing.T) {
	mid, down := newEcho(), newCapture()

	ctx := context.Background()
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, mid, down)

	if err := mid.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.StartFrame](t, down.got, "StartFrame")

	usage := frames.LLMTokenUsage{PromptTokens: 12, CompletionTokens: 34, TotalTokens: 46}
	if err := mid.PushTokenUsage(ctx, "some-model", usage); err != nil {
		t.Fatalf("PushTokenUsage: %v", err)
	}

	mf := mustReceive[*frames.MetricsFrame](t, down.got, "MetricsFrame")
	if len(mf.Data) != 1 {
		t.Fatalf("the frame carries %d measurements, want 1", len(mf.Data))
	}
	data, ok := mf.Data[0].(frames.LLMUsageMetricsData)
	if !ok {
		t.Fatalf("the frame carries %T, want LLMUsageMetricsData", mf.Data[0])
	}
	if data.Model != "some-model" {
		t.Errorf("Model = %q, want some-model", data.Model)
	}
	if data.Processor != mid.Name() {
		t.Errorf("Processor = %q, want %q", data.Processor, mid.Name())
	}
	if data.Value != usage {
		t.Errorf("Value = %+v, want %+v", data.Value, usage)
	}
}

// TestBasePushError checks that an error goes upstream, toward the task that
// decides what to do about it, naming the processor it came from.
func TestBasePushError(t *testing.T) {
	up, mid, down := newCapture(), newEcho(), newCapture()

	ctx := context.Background()
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, up, mid, down)

	if err := mid.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.StartFrame](t, down.got, "StartFrame")

	cause := context.DeadlineExceeded
	mid.PushError(ctx, "the provider gave up", cause, true)

	ef := mustReceive[*frames.ErrorFrame](t, up.got, "ErrorFrame")
	if ef.Error != "the provider gave up" {
		t.Errorf("Error = %q, want the provider gave up", ef.Error)
	}
	if !errors.Is(ef.Err, cause) {
		t.Errorf("Err = %v, want %v", ef.Err, cause)
	}
	if !ef.Fatal {
		t.Error("Fatal is false on an error pushed as fatal")
	}
	if ef.Source != processor.Processor(mid) {
		t.Error("Source is not the processor that raised the error")
	}
}

// recorder is an observer that records every report it is given. It implements
// ProcessObserver too, so it sees a frame reaching a processor as well as the
// handovers between them.
type recorder struct {
	mu        sync.Mutex
	pushed    []processor.FramePushed
	processed []processor.FrameProcessed
}

func (r *recorder) OnPushFrame(data processor.FramePushed) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pushed = append(r.pushed, data)
}

func (r *recorder) OnProcessFrame(data processor.FrameProcessed) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.processed = append(r.processed, data)
}

func (r *recorder) reports() ([]processor.FramePushed, []processor.FrameProcessed) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]processor.FramePushed(nil), r.pushed...),
		append([]processor.FrameProcessed(nil), r.processed...)
}

// TestBaseNotifiesObservers checks that every handover is reported, and not only
// what reaches the ends of the pipeline: an observer sees where each frame came
// from, which is what lets it tell the same frame apart at two points in the
// pipeline. It also sees a frame arriving, before the processor has handled it.
func TestBaseNotifiesObservers(t *testing.T) {
	rec := &recorder{}
	mid, down := newEcho(), newCapture()

	ctx := context.Background()
	clk := clock.NewSystem()
	clk.Start()
	linkChain(t, ctx, processor.Setup{Clock: clk, Observers: []processor.Observer{rec}}, mid, down)

	if err := mid.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.StartFrame](t, down.got, "StartFrame")

	tf := frames.NewTextFrame("watched")
	if err := mid.QueueFrame(ctx, tf, processor.Downstream); err != nil {
		t.Fatal(err)
	}
	mustReceive[*frames.TextFrame](t, down.got, "TextFrame")

	pushed, processed := rec.reports()

	var handover *processor.FramePushed
	for i, p := range pushed {
		if p.Frame == frames.Frame(tf) {
			handover = &pushed[i]
			break
		}
	}
	if handover == nil {
		t.Fatal("the handover of the text frame was not reported")
	}
	if handover.Source != processor.Processor(mid) {
		t.Errorf("Source = %v, want the processor that pushed", handover.Source)
	}
	if handover.Destination != processor.Processor(down) {
		t.Errorf("Destination = %v, want the processor it went to", handover.Destination)
	}
	if handover.Direction != processor.Downstream {
		t.Errorf("Direction = %s, want downstream", handover.Direction)
	}
	if handover.Timestamp <= 0 {
		t.Error("the report carries no time off the pipeline clock")
	}

	var arrival *processor.FrameProcessed
	for i, p := range processed {
		if p.Frame == frames.Frame(tf) && p.Processor == processor.Processor(down) {
			arrival = &processed[i]
			break
		}
	}
	if arrival == nil {
		t.Fatal("the frame reaching the far processor was not reported")
	}
	if arrival.Direction != processor.Downstream {
		t.Errorf("Direction = %s, want downstream", arrival.Direction)
	}
	if arrival.Timestamp <= 0 {
		t.Error("the report carries no time off the pipeline clock")
	}
}

// TestBaseHasQueuedFrame checks that a processor can tell more of the same work
// is already on its way, which is what lets it hold off on an action until the
// last of it arrives rather than repeating it once per frame.
func TestBaseHasQueuedFrame(t *testing.T) {
	h := newHolder()

	ctx := context.Background()
	if err := h.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatal(err)
	}
	defer h.Cleanup(ctx)

	isText := func(f frames.Frame) bool {
		_, ok := f.(*frames.TextFrame)
		return ok
	}

	if err := h.QueueFrame(ctx, frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if err := h.QueueFrame(ctx, frames.NewTextFrame("first"), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	select {
	case <-h.entered:
	case <-time.After(time.Second):
		t.Fatal("the processor never began the first frame")
	}

	// Nothing is waiting behind the frame being handled now.
	if h.HasQueuedFrame(isText) {
		t.Fatal("a frame is reported queued while only the held one exists")
	}

	if err := h.QueueFrame(ctx, frames.NewTextFrame("second"), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	// The frame reaches the in-order queue on the processor's own goroutine.
	deadline := time.Now().Add(time.Second)
	for !h.HasQueuedFrame(isText) {
		if time.Now().After(deadline) {
			t.Fatal("the queued frame was never reported")
		}
		time.Sleep(time.Millisecond)
	}

	close(h.release)
}
