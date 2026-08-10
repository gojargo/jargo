package pipeline_test

import (
	"context"
	"errors"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// tagger replaces every downstream TextFrame with a new one whose text is
// prefixed with tag, so each branch of a parallel pipeline produces a distinct
// frame. Other frames pass through unchanged.
type tagger struct {
	*processor.Base
	tag string
}

func newTagger(tag string) *tagger {
	t := &tagger{tag: tag}
	t.Base = processor.New("Tagger", t)
	return t
}

func (t *tagger) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := t.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok && dir == processor.Downstream {
		return t.PushFrame(ctx, frames.NewTextFrame(t.tag+tf.Text), dir)
	}
	return t.PushFrame(ctx, f, dir)
}

// runParallel runs a task wrapping pp, queues frames, stops when done, and
// returns every frame that reached the end of the pipeline.
func runParallel(t *testing.T, pp *pipeline.ParallelPipeline, in []frames.Frame) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewTask(pipeline.New(pp), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrames(in)
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("parallel task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}

func TestParallelFanOutMerge(t *testing.T) {
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newTagger("A")},
		[]processor.Processor{newTagger("B")},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	got := runParallel(t, pp, []frames.Frame{frames.NewTextFrame("hi")})

	var texts []string
	starts, ends := 0, 0
	for _, f := range got {
		switch fr := f.(type) {
		case *frames.TextFrame:
			texts = append(texts, fr.Text)
		case *frames.StartFrame:
			starts++
		case *frames.EndFrame:
			ends++
		}
	}
	sort.Strings(texts)

	if len(texts) != 2 || texts[0] != "Ahi" || texts[1] != "Bhi" {
		t.Errorf("downstream texts = %v, want [Ahi Bhi]", texts)
	}
	// Each branch forwards the same lifecycle frame; the merge must release one.
	if starts != 1 {
		t.Errorf("StartFrame reached downstream %d times, want 1", starts)
	}
	if ends != 1 {
		t.Errorf("EndFrame reached downstream %d times, want 1", ends)
	}
}

func TestParallelPassthroughDeduplicated(t *testing.T) {
	// Both branches forward the input frame unchanged, so it shares a frame id at
	// both sinks and must escape exactly once.
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newEcho()},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	got := runParallel(t, pp, []frames.Frame{frames.NewTextFrame("once")})

	n := 0
	for _, f := range got {
		if tf, ok := f.(*frames.TextFrame); ok && tf.Text == "once" {
			n++
		}
	}
	if n != 1 {
		t.Errorf("passthrough TextFrame reached downstream %d times, want 1", n)
	}
}

func TestParallelNoBranches(t *testing.T) {
	if _, err := pipeline.NewParallel(); err == nil {
		t.Fatal("NewParallel() with no branches: want error, got nil")
	}
}

func TestParallelLifecycleEndToEnd(t *testing.T) {
	// A parallel pipeline with branches of differing length still completes its
	// run: the EndFrame is released only after both branches have flushed.
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newEcho()},
		[]processor.Processor{newEcho(), newEcho(), newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	task := pipeline.NewTask(pipeline.New(pp), pipeline.TaskParams{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTextFrame("x"))
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("parallel task did not finish")
	}
	if !task.HasFinished() {
		t.Error("HasFinished() = false, want true")
	}
}

// emitOnStart forwards every frame and, on a StartFrame, pushes a TextFrame
// downstream as well: output produced while the pipeline is still starting.
type emitOnStart struct {
	*processor.Base
	text string
}

func newEmitOnStart(text string) *emitOnStart {
	p := &emitOnStart{text: text}
	p.Base = processor.New("EmitOnStart", p)
	return p
}

func (p *emitOnStart) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		return p.PushFrame(ctx, frames.NewTextFrame(p.text), processor.Downstream)
	}
	return nil
}

func TestParallelInternalFramesBufferedDuringStart(t *testing.T) {
	// A frame a branch pushes while the StartFrame is still synchronizing is held
	// back and released after it, ahead of the frames that follow.
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newEmitOnStart("from start")},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	var order []string
	for _, f := range runParallel(t, pp, []frames.Frame{frames.NewTextFrame("hello")}) {
		switch fr := f.(type) {
		case *frames.TextFrame:
			order = append(order, fr.Text)
		case *frames.StartFrame:
			order = append(order, "start")
		}
	}

	want := []string{"start", "from start", "hello"}
	if !equalStrings(order, want) {
		t.Errorf("downstream order = %v, want %v", order, want)
	}
}

// endEmitter turns a TextFrame carrying "bye" into a new EndFrame pushed
// downstream. It stands in for a processor inside a branch that ends the session
// on its own, an idle monitor or a transport hanging up.
type endEmitter struct {
	*processor.Base
}

func newEndEmitter() *endEmitter {
	p := &endEmitter{}
	p.Base = processor.New("EndEmitter", p)
	return p
}

func (p *endEmitter) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok && tf.Text == "bye" && dir == processor.Downstream {
		return p.PushFrame(ctx, frames.NewEndFrame(), processor.Downstream)
	}
	return p.PushFrame(ctx, f, dir)
}

func TestParallelBranchGeneratedLifecycleFrameEscapes(t *testing.T) {
	// A lifecycle frame a branch raises itself was never fanned out, so it has no
	// synchronization counter. It still has to leave the parallel pipeline.
	pp, err := pipeline.NewParallel(
		[]processor.Processor{newEndEmitter()},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	var mu sync.Mutex
	sawEnd := make(chan struct{})
	var once sync.Once
	task := pipeline.NewTask(pipeline.New(pp), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			if _, ok := f.(*frames.EndFrame); ok {
				once.Do(func() { close(sawEnd) })
			}
		},
	})

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTextFrame("bye"))

	select {
	case <-sawEnd:
	case <-time.After(3 * time.Second):
		t.Error("branch-generated EndFrame never left the parallel pipeline")
	}

	task.Cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not stop")
	}
}

// slowEnd holds the EndFrame back, and notes whether an InterruptionFrame
// reached it while it still had that EndFrame in hand.
type slowEnd struct {
	*processor.Base
	delay time.Duration

	mu             sync.Mutex
	holdingEnd     bool
	interruptedMid bool
}

func newSlowEnd(delay time.Duration) *slowEnd {
	p := &slowEnd{delay: delay}
	p.Base = processor.New("SlowEnd", p)
	return p
}

func (p *slowEnd) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if _, ok := f.(*frames.InterruptionFrame); ok {
		// System frames are handled on the input goroutine, so this runs while
		// the EndFrame is still being held on the process goroutine.
		p.mu.Lock()
		if p.holdingEnd {
			p.interruptedMid = true
		}
		p.mu.Unlock()
	}

	if _, ok := f.(*frames.EndFrame); ok {
		p.mu.Lock()
		p.holdingEnd = true
		p.mu.Unlock()

		select {
		case <-time.After(p.delay):
		case <-ctx.Done():
		}

		p.mu.Lock()
		p.holdingEnd = false
		p.mu.Unlock()
	}
	return p.PushFrame(ctx, f, dir)
}

// interruptedMidEnd reports whether an InterruptionFrame arrived while this
// processor still had the EndFrame in hand.
func (p *slowEnd) interruptedMidEnd() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.interruptedMid
}

// interruptAfterEnd forwards the EndFrame and then, once the parallel pipeline
// is synchronizing it, pushes an InterruptionFrame downstream behind it.
type interruptAfterEnd struct {
	*processor.Base
	delay time.Duration
}

func newInterruptAfterEnd(delay time.Duration) *interruptAfterEnd {
	p := &interruptAfterEnd{delay: delay}
	p.Base = processor.New("InterruptAfterEnd", p)
	return p
}

func (p *interruptAfterEnd) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.EndFrame); ok {
		go func() {
			select {
			case <-time.After(p.delay):
			case <-ctx.Done():
				return
			}
			_ = p.PushFrame(ctx, frames.NewInterruptionFrame(), processor.Downstream)
		}()
	}
	return nil
}

func TestParallelHoldsSystemFramesWhileSynchronizing(t *testing.T) {
	// A system frame arriving while an EndFrame is synchronizing must wait: fanned
	// out early it would reach the branches mid-shutdown and flush the output they
	// still had queued to send ahead of the EndFrame.
	slow := newSlowEnd(600 * time.Millisecond)
	pp, err := pipeline.NewParallel(
		[]processor.Processor{slow},
		[]processor.Processor{newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	var mu sync.Mutex
	var endReleased bool
	task := pipeline.NewTask(
		pipeline.New(newInterruptAfterEnd(150*time.Millisecond), pp),
		pipeline.TaskParams{
			ReachedDownstreamFilter: pipeline.AnyFrame,
			OnReachedDownstream: func(f frames.Frame) {
				if _, ok := f.(*frames.EndFrame); ok {
					mu.Lock()
					endReleased = true
					mu.Unlock()
				}
			},
		},
	)

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("parallel task did not finish")
	}

	mu.Lock()
	released := endReleased
	mu.Unlock()
	if !released {
		t.Fatal("EndFrame never left the parallel pipeline")
	}
	if slow.interruptedMidEnd() {
		t.Error("InterruptionFrame reached the branch while it still held the EndFrame: " +
			"a system frame must wait for the synchronization to finish")
	}
}
