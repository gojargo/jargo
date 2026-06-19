package pipeline

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Default sample rates used for the StartFrame when not overridden.
const (
	defaultAudioInSampleRate  = 16000
	defaultAudioOutSampleRate = 24000
)

// TaskParams configures a Task.
type TaskParams struct {
	// Clock is the pipeline clock; a system clock is used when nil.
	Clock clock.Clock
	// AudioInSampleRate is the StartFrame input sample rate; default 16000.
	AudioInSampleRate int
	// AudioOutSampleRate is the StartFrame output sample rate; default 24000.
	AudioOutSampleRate int
	// OnReachedDownstream, if set, is called for every frame that reaches the
	// end of the pipeline.
	OnReachedDownstream func(frames.Frame)
	// OnReachedUpstream, if set, is called for every frame that reaches the
	// start of the pipeline.
	OnReachedUpstream func(frames.Frame)
}

// Task runs a pipeline for a single session. It drives the lifecycle: it sends
// the StartFrame, waits for the pipeline to be ready, pushes queued frames, and
// shuts the pipeline down once an EndFrame or CancelFrame has traveled all the
// way through.
type Task struct {
	pipeline *Pipeline
	source   processor.Processor
	sink     processor.Processor
	params   TaskParams
	clk      clock.Clock

	pushQueue *frameQueue

	startOnce sync.Once
	startSig  chan struct{}
	endOnce   sync.Once
	endSig    chan struct{}

	mu        sync.Mutex
	finished  bool
	canceling bool
}

// NewTask wraps pipe in a Task. pipe is usually a *Pipeline but may be any
// processor.
func NewTask(pipe processor.Processor, params TaskParams) *Task {
	if params.AudioInSampleRate == 0 {
		params.AudioInSampleRate = defaultAudioInSampleRate
	}
	if params.AudioOutSampleRate == 0 {
		params.AudioOutSampleRate = defaultAudioOutSampleRate
	}
	t := &Task{
		params:    params,
		clk:       params.Clock,
		pushQueue: newFrameQueue(),
		startSig:  make(chan struct{}),
		endSig:    make(chan struct{}),
	}
	if t.clk == nil {
		t.clk = clock.NewSystem()
	}
	// The source observes upstream frames, the sink observes downstream frames.
	// They bracket the user pipeline so the task can inject and observe frames.
	t.source = processor.NewSource("Task::Source", t.sourcePush)
	t.sink = processor.NewSink("Task::Sink", t.sinkPush)
	t.pipeline = build(t.source, t.sink, []processor.Processor{pipe})
	return t
}

// QueueFrame queues a frame to be pushed downstream through the pipeline.
func (t *Task) QueueFrame(f frames.Frame) { t.pushQueue.push(f) }

// QueueFrames queues several frames to be pushed downstream, in order.
func (t *Task) QueueFrames(fs []frames.Frame) {
	for _, f := range fs {
		t.pushQueue.push(f)
	}
}

// StopWhenDone schedules the pipeline to stop once all queued frames have been
// processed, by queueing an EndFrame.
func (t *Task) StopWhenDone() { t.QueueFrame(frames.NewEndFrame()) }

// Cancel stops the pipeline immediately by queueing a CancelFrame.
func (t *Task) Cancel() {
	t.mu.Lock()
	if t.canceling {
		t.mu.Unlock()
		return
	}
	t.canceling = true
	t.mu.Unlock()
	t.QueueFrame(frames.NewCancelFrame())
}

// HasFinished reports whether the task has finished running.
func (t *Task) HasFinished() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.finished
}

// Run sets up the pipeline and drives it until an EndFrame or CancelFrame
// completes its journey through the pipeline, or ctx is canceled. It then
// cleans up the pipeline. Run blocks until the pipeline has finished.
func (t *Task) Run(ctx context.Context) error {
	t.mu.Lock()
	if t.finished {
		t.mu.Unlock()
		return nil
	}
	t.mu.Unlock()

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	if err := t.pipeline.Setup(runCtx, processor.Setup{Clock: t.clk}); err != nil {
		return err
	}

	runErr := t.runLoop(runCtx)

	// Clean up with a fresh context so a canceled runCtx does not abort the
	// goroutine shutdown.
	_ = t.pipeline.Cleanup(context.Background())

	t.mu.Lock()
	t.finished = true
	t.mu.Unlock()
	return runErr
}

// runLoop sends the StartFrame, waits for the pipeline to be ready, then pushes
// queued frames until a pipeline-ending frame has traveled through.
func (t *Task) runLoop(ctx context.Context) error {
	t.clk.Start()

	start := frames.NewStartFrame()
	start.AudioInSampleRate = t.params.AudioInSampleRate
	start.AudioOutSampleRate = t.params.AudioOutSampleRate
	if err := t.pipeline.QueueFrame(ctx, start, processor.Downstream); err != nil {
		return err
	}

	select {
	case <-t.startSig:
	case <-ctx.Done():
		return ctx.Err()
	}

	for {
		f, ok := t.pushQueue.get(ctx)
		if !ok {
			return ctx.Err()
		}
		if err := t.pipeline.QueueFrame(ctx, f, processor.Downstream); err != nil {
			return err
		}
		if isPipelineEnd(f) {
			select {
			case <-t.endSig:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		}
	}
}

// sinkPush observes frames reaching the end of the pipeline and signals the
// lifecycle events the run loop waits on.
func (t *Task) sinkPush(_ context.Context, f frames.Frame, _ processor.Direction) error {
	switch f.(type) {
	case *frames.StartFrame:
		t.startOnce.Do(func() { close(t.startSig) })
	case *frames.EndFrame, *frames.CancelFrame:
		t.endOnce.Do(func() { close(t.endSig) })
	}
	if t.params.OnReachedDownstream != nil {
		t.params.OnReachedDownstream(f)
	}
	return nil
}

// sourcePush observes frames reaching the start of the pipeline. A fatal error
// cancels the pipeline.
func (t *Task) sourcePush(_ context.Context, f frames.Frame, _ processor.Direction) error {
	if ef, ok := f.(*frames.ErrorFrame); ok && ef.Fatal {
		t.Cancel()
	}
	if t.params.OnReachedUpstream != nil {
		t.params.OnReachedUpstream(f)
	}
	return nil
}

func isPipelineEnd(f frames.Frame) bool {
	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		return true
	}
	return false
}

// frameQueue is an unbounded, concurrency-safe FIFO of frames with a single
// consumer, used for frames the user queues for the task to push.
type frameQueue struct {
	mu     sync.Mutex
	items  []frames.Frame
	notify chan struct{}
}

func newFrameQueue() *frameQueue {
	return &frameQueue{notify: make(chan struct{}, 1)}
}

func (q *frameQueue) push(f frames.Frame) {
	q.mu.Lock()
	q.items = append(q.items, f)
	q.mu.Unlock()
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

func (q *frameQueue) get(ctx context.Context) (frames.Frame, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			f := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return f, true
		}
		q.mu.Unlock()

		select {
		case <-ctx.Done():
			return nil, false
		case <-q.notify:
		}
	}
}
