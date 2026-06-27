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
// Lifecycle frames — StartFrame, EndFrame and CancelFrame — are synchronized.
// The parallel pipeline waits for every branch to process the frame before
// letting a single copy continue, and buffers any other frames a branch emits in
// the meantime (flushing them after a StartFrame, before an EndFrame or
// CancelFrame). This stops a fast branch from leaking an EndFrame and shutting
// downstream processors down while a slower branch still has output to flush, or
// from emitting data before every branch has been started.
//
// A ParallelPipeline is itself a processor, so it nests inside a Pipeline.
//
// The deduplication set retains the id of every frame that escapes the parallel
// pipeline for the lifetime of the session; this mirrors the upstream design and
// is bounded only by the session's frame count.
type ParallelPipeline struct {
	*processor.Base
	branches []*Pipeline

	mu       sync.Mutex
	seen     map[uint64]struct{}
	counter  map[uint64]int
	syncDone map[uint64]chan struct{}
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
	if len(branches) == 0 {
		return nil, errNoBranches
	}
	p := &ParallelPipeline{
		seen:     map[uint64]struct{}{},
		counter:  map[uint64]int{},
		syncDone: map[uint64]chan struct{}{},
	}
	p.Base = processor.New("ParallelPipeline", p)
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
// arms the synchronization counter, then blocks until every branch has processed
// the frame (or the context is canceled), so the next frame is not fanned out
// until the lifecycle frame has fully propagated.
func (p *ParallelPipeline) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	var done chan struct{}
	if isLifecycle(f) {
		done = make(chan struct{})
		p.mu.Lock()
		p.counter[f.ID()] = len(p.branches)
		p.syncDone[f.ID()] = done
		p.syncing = true
		p.mu.Unlock()
	}

	for _, b := range p.branches {
		if err := b.QueueFrame(ctx, f, dir); err != nil {
			return err
		}
	}

	if done == nil {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
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
	return p.PushFrame(ctx, f, dir)
}

// sinkPush is the branch sinks' downstream handler. Lifecycle frames decrement
// the synchronization counter and are released once the last branch reports in;
// every other frame goes out through emit.
func (p *ParallelPipeline) sinkPush(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if !isLifecycle(f) {
		return p.emit(ctx, f, dir)
	}

	p.mu.Lock()
	n, ok := p.counter[f.ID()]
	if !ok || n == 0 {
		p.mu.Unlock()
		return nil
	}
	n--
	p.counter[f.ID()] = n
	if n > 0 {
		p.mu.Unlock()
		return nil
	}
	// Last branch in: release the synchronized frame and any buffered frames.
	delete(p.counter, f.ID())
	doneCh := p.syncDone[f.ID()]
	delete(p.syncDone, f.ID())
	p.syncing = false
	buffered := p.buffered
	p.buffered = nil
	_, seen := p.seen[f.ID()]
	first := !seen
	if first {
		p.seen[f.ID()] = struct{}{}
	}
	p.mu.Unlock()

	// A StartFrame must precede the frames buffered behind it; an EndFrame or
	// CancelFrame must follow them.
	_, isStart := f.(*frames.StartFrame)
	if isStart && first {
		_ = p.PushFrame(ctx, f, dir)
	}
	for _, bf := range buffered {
		_ = p.PushFrame(ctx, bf.frame, bf.dir)
	}
	if !isStart && first {
		_ = p.PushFrame(ctx, f, dir)
	}

	if doneCh != nil {
		close(doneCh)
	}
	return nil
}

// Setup sets up the parallel pipeline and every branch.
func (p *ParallelPipeline) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	for _, b := range p.branches {
		if err := b.Setup(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// Cleanup cleans up the parallel pipeline and every branch.
func (p *ParallelPipeline) Cleanup(ctx context.Context) error {
	_ = p.Base.Cleanup(ctx)
	for _, b := range p.branches {
		_ = b.Cleanup(ctx)
	}
	return nil
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
