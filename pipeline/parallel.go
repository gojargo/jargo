package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errNoBranches is returned by NewParallel when no branch is supplied.
//
//nolint:gochecknoglobals // sentinel error
var errNoBranches = errors.New("pipeline: parallel pipeline needs at least one branch")

// ParallelPipeline runs several sub-pipelines, the branches, concurrently. Every
// frame entering the parallel pipeline is fanned out to all branches, and frames
// emerging from the branches are merged back out. A frame that more than one
// branch forwards unchanged escapes only once: the merge deduplicates by frame
// id.
//
// Lifecycle frames (StartFrame, EndFrame and CancelFrame) are synchronized. On
// one of those the parallel pipeline pauses its own frame handling and lets a
// single copy continue only once every branch has processed it, buffering any
// other frames a branch emits in the meantime (flushing them after a StartFrame,
// before an EndFrame or CancelFrame). Three things go wrong without this:
//
//   - StartFrame: a fast branch completing first would let processors in the
//     other branches receive frames before their StartFrame.
//   - EndFrame: an EndFrame escaping from a fast branch would shut downstream
//     processors down while a slower branch still had output to flush.
//   - CancelFrame: the Task waits for the CancelFrame to reach the pipeline
//     sink, so one escaping early would make it consider cancellation complete
//     while slower branches were still running.
//
// A ParallelPipeline is itself a processor, so it nests inside a Pipeline.
//
// The deduplication set retains the id of every frame that escapes the parallel
// pipeline for the lifetime of the session; this mirrors the upstream design and
// is bounded only by the session's frame count.
type ParallelPipeline struct {
	*processor.Base
	setupTracker
	branches []*Pipeline

	mu       sync.Mutex
	seen     map[uint64]struct{}
	counter  map[uint64]int
	syncing  bool
	buffered []bufferedFrame
}

// bufferedFrame is a frame held back, with its direction, while the parallel
// pipeline synchronizes a lifecycle frame.
type bufferedFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// NewParallel builds a ParallelPipeline from one or more branches, each a list of
// processors connected in order. It returns an error if no branch is given.
func NewParallel(branches ...[]processor.Processor) (*ParallelPipeline, error) {
	return newParallelAs(nil, "ParallelPipeline", branches...)
}

// newParallelAs builds a ParallelPipeline on behalf of self, the processor
// embedding it, under the given name. A nil self means the parallel pipeline is
// the outermost processor and stands for itself.
//
// Naming the outer value matters because the frames a branch produces leave
// through it: an embedding type overrides PushFrame to decide what escapes, and
// the pushes below go through Self so that override is honored.
func newParallelAs(
	self processor.Processor, name string, branches ...[]processor.Processor,
) (*ParallelPipeline, error) {
	if len(branches) == 0 {
		return nil, errNoBranches
	}
	p := &ParallelPipeline{
		seen:    map[uint64]struct{}{},
		counter: map[uint64]int{},
	}
	if self == nil {
		self = p
	}
	// Not direct mode: the synchronization pauses frame handling, which needs the
	// processor's queues.
	p.Base = processor.New(name, self)
	for i, procs := range branches {
		// A source and sink bracket each branch so the parallel pipeline controls
		// the frames pushed out of it: the source handles upstream frames, the
		// sink handles downstream frames.
		src := processor.NewSource(fmt.Sprintf("%s::Source%d", p.Name(), i), p.emit)
		sink := processor.NewSink(fmt.Sprintf("%s::Sink%d", p.Name(), i), p.sinkPush)
		p.branches = append(p.branches, build(src, sink, procs))
	}
	return p, nil
}

// ProcessFrame fans a frame out to every branch. For a lifecycle frame it first
// arms the synchronization counter and pauses its own frame handling, so nothing
// else is fanned out until the last branch reports the frame back and the sink
// resumes it.
func (p *ParallelPipeline) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if isLifecycle(f) {
		p.mu.Lock()
		p.counter[f.ID()] = len(p.branches)
		p.syncing = true
		p.mu.Unlock()
		p.PauseProcessingSystemFrames()
		p.PauseProcessingFrames()
	}

	for _, b := range p.branches {
		if err := b.QueueFrame(ctx, f, dir); err != nil {
			return err
		}
	}
	return nil
}

// emit pushes a frame out of the parallel pipeline, dropping duplicates and
// buffering frames that arrive while a lifecycle frame is being synchronized. It
// is the branch sources' upstream handler and the path for non-lifecycle frames
// from the sinks.
func (p *ParallelPipeline) emit(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	p.mu.Lock()
	if _, ok := p.seen[f.ID()]; ok {
		p.mu.Unlock()
		return nil
	}
	p.seen[f.ID()] = struct{}{}
	if p.syncing {
		p.buffered = append(p.buffered, bufferedFrame{frame: f, dir: dir})
		p.mu.Unlock()
		return nil
	}
	p.mu.Unlock()
	return p.Self().PushFrame(ctx, f, dir)
}

// sinkPush is the branch sinks' downstream handler. Lifecycle frames decrement
// the synchronization counter and are released once the last branch reports in;
// every other frame goes out through emit.
//
// A lifecycle frame the parallel pipeline never fanned out has no counter, which
// counts as zero: a frame a branch raised on its own (an EndFrame from something
// inside it that ends the session) is released the same way rather than dropped.
func (p *ParallelPipeline) sinkPush(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if !isLifecycle(f) {
		return p.emit(ctx, f, dir)
	}

	p.mu.Lock()
	n := p.counter[f.ID()]
	if n > 0 {
		n--
		p.counter[f.ID()] = n
	}
	if n > 0 {
		p.mu.Unlock()
		return nil
	}
	// Last branch in: release the synchronized frame and any buffered frames.
	delete(p.counter, f.ID())
	p.mu.Unlock()

	// A StartFrame must precede the frames buffered behind it; an EndFrame or
	// CancelFrame must follow them.
	//
	// The synchronizing flag stays set until the buffer has drained, and the
	// lifecycle frame goes out past it. Clearing the flag first would open a
	// window in which a frame another branch pushes goes straight out, ahead of
	// the StartFrame it is supposed to follow. The branches here are goroutines,
	// so that window is real.
	if _, isStart := f.(*frames.StartFrame); isStart {
		_ = p.emitSynchronized(ctx, f, dir)
		p.flushBuffered(ctx)
	} else {
		p.flushBuffered(ctx)
		_ = p.emitSynchronized(ctx, f, dir)
	}

	p.ResumeProcessingSystemFrames()
	p.ResumeProcessingFrames()
	return nil
}

// emitSynchronized pushes the lifecycle frame itself, past the buffer still
// holding everything else back. It deduplicates like emit, since a frame the
// parallel pipeline fanned out comes back once per branch.
func (p *ParallelPipeline) emitSynchronized(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	p.mu.Lock()
	if _, ok := p.seen[f.ID()]; ok {
		p.mu.Unlock()
		return nil
	}
	p.seen[f.ID()] = struct{}{}
	p.mu.Unlock()
	return p.Self().PushFrame(ctx, f, dir)
}

// flushBuffered pushes out the frames held back while a lifecycle frame was
// synchronizing. They were deduplicated on the way in, so they go straight out.
// Synchronizing ends here, once there is nothing left held back, so a frame
// arriving meanwhile joins the queue rather than overtaking what is in it.
func (p *ParallelPipeline) flushBuffered(ctx context.Context) {
	for {
		p.mu.Lock()
		if len(p.buffered) == 0 {
			p.syncing = false
			p.mu.Unlock()
			return
		}
		bf := p.buffered[0]
		p.buffered = p.buffered[1:]
		p.mu.Unlock()

		_ = p.Self().PushFrame(ctx, bf.frame, bf.dir)
	}
}

// Processors returns the branches.
func (p *ParallelPipeline) Processors() []processor.Processor { return branchList(p.branches) }

// EntryProcessors returns the branches: a frame entering the parallel pipeline
// is fanned out to all of them.
func (p *ParallelPipeline) EntryProcessors() []processor.Processor {
	return branchList(p.branches)
}

// ProcessorsWithMetrics returns the processors across every branch that report
// metrics.
func (p *ParallelPipeline) ProcessorsWithMetrics() []processor.Processor {
	var out []processor.Processor
	for _, b := range p.branches {
		out = append(out, b.ProcessorsWithMetrics()...)
	}
	return out
}

// branchList widens a list of branches to the processor interface.
func branchList(branches []*Pipeline) []processor.Processor {
	out := make([]processor.Processor, 0, len(branches))
	for _, b := range branches {
		out = append(out, b)
	}
	return out
}

// Setup sets up the parallel pipeline and every branch.
func (p *ParallelPipeline) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	setupProcessors(ctx, &p.setupTracker, branchProcessors(p.branches), s)
	return nil
}

// Cleanup cleans up the parallel pipeline and every branch.
func (p *ParallelPipeline) Cleanup(ctx context.Context) error {
	_ = p.Base.Cleanup(ctx)
	cleanupProcessors(ctx, &p.setupTracker, branchProcessors(p.branches))
	return nil
}

// branchProcessors views the branches as the processors they are, so a parallel
// pipeline sets them up and cleans them up the way a linear one does its chain.
func branchProcessors(branches []*Pipeline) []processor.Processor {
	procs := make([]processor.Processor, len(branches))
	for i, b := range branches {
		procs[i] = b
	}
	return procs
}

// isLifecycle reports whether f is one of the frames whose propagation across
// branches the parallel pipeline synchronizes.
func isLifecycle(f frames.Frame) bool {
	switch f.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame:
		return true
	default:
		return false
	}
}
