package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errNoSyncBranches is returned by NewSyncParallel when no branch is supplied.
//
//nolint:gochecknoglobals // sentinel error
var errNoSyncBranches = errors.New("pipeline: sync parallel pipeline needs at least one branch")

// FrameOrder is the order a SyncParallelPipeline pushes the output of one input
// frame in, once every branch has produced it.
type FrameOrder int

const (
	// FrameOrderArrival pushes frames in the order they arrive from any branch.
	// It is the zero value, and so the default.
	FrameOrderArrival FrameOrder = iota
	// FrameOrderPipeline pushes frames in branch order: everything the first
	// branch produced, then the second, and so on. Use it when the order between
	// branches matters, to keep an image ahead of the audio that goes with it.
	FrameOrderPipeline
)

// SyncFrame is the sentinel sent in behind a frame to detect when a branch has
// finished with it. Frames are handled in order, so a SyncFrame coming out of a
// branch means everything that branch produced for the preceding frame has come
// out already.
type SyncFrame struct {
	frames.BaseControlFrame
}

// NewSyncFrame builds a SyncFrame.
func NewSyncFrame() *SyncFrame {
	return &SyncFrame{BaseControlFrame: frames.NewBaseControlFrame("SyncFrame")}
}

// syncEndpoint is one end of a branch together with the queue that end collects
// output in: a source with the queue its branch's sink fills, a sink with the
// queue its branch's source fills.
type syncEndpoint struct {
	proc  processor.Processor
	queue *frameQueue
}

// SyncParallelPipeline runs several sub-pipelines, the branches, and holds the
// output of each input frame until every branch has finished producing it. For
// each frame it:
//
//  1. sends the frame into every branch,
//  2. sends a SyncFrame in behind it,
//  3. waits for every branch to produce its SyncFrame, which means all the
//     output for that input is ready,
//  4. releases the collected frames, deduplicated by frame id since the same
//     frame can come out of more than one branch.
//
// The last processor of each branch has to be synchronous for this to work: the
// SyncFrame only means the branch is done because frames are handled in order.
//
// System frames skip the mechanism and pass straight through, since they are
// meant to race ahead of data frames. Note that a system frame also lands in the
// branch queues on its way through, so it is released a second time by the next
// frame that does synchronize.
//
// Use it where output has to stay together, for instance an image and the speech
// describing it. Where branches are independent, ParallelPipeline is the lighter
// choice.
type SyncParallelPipeline struct {
	*processor.Base
	frameOrder FrameOrder
	branches   []*Pipeline
	sources    []syncEndpoint
	sinks      []syncEndpoint

	// The queues the collected output of one input frame is gathered in, shared
	// by every branch, one per direction.
	upQueue   *frameQueue
	downQueue *frameQueue
}

// NewSyncParallel builds a SyncParallelPipeline from one or more branches, each
// a list of processors connected in order, pushing output in the given order. It
// returns an error if no branch is given.
func NewSyncParallel(order FrameOrder, branches ...[]processor.Processor) (*SyncParallelPipeline, error) {
	if len(branches) == 0 {
		return nil, errNoSyncBranches
	}
	p := &SyncParallelPipeline{
		frameOrder: order,
		upQueue:    newFrameQueue(),
		downQueue:  newFrameQueue(),
	}
	p.Base = processor.New("SyncParallelPipeline", p)

	for i, procs := range branches {
		// A source and sink bracket each branch. Rather than pushing frames out
		// of the pipeline they collect them in a queue, which is what lets the
		// output be held back until every branch is done.
		upQ, downQ := newFrameQueue(), newFrameQueue()
		src := processor.NewSource(fmt.Sprintf("%s::Source%d", p.Name(), i),
			func(_ context.Context, f frames.Frame, _ processor.Direction) error {
				upQ.push(f)
				return nil
			})
		sink := processor.NewSink(fmt.Sprintf("%s::Sink%d", p.Name(), i),
			func(_ context.Context, f frames.Frame, _ processor.Direction) error {
				downQ.push(f)
				return nil
			})

		// A source collects what comes out of its branch's sink, and vice versa.
		p.sources = append(p.sources, syncEndpoint{proc: src, queue: downQ})
		p.sinks = append(p.sinks, syncEndpoint{proc: sink, queue: upQ})
		p.branches = append(p.branches, build(src, sink, procs))
	}
	return p, nil
}

// ProcessFrame sends a frame through every branch and releases the output once
// they have all finished with it.
func (p *SyncParallelPipeline) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	// System frames pass through every branch without the queued output being
	// drained. Draining here would let a system frame's wait take the frames a
	// data frame's wait is collecting.
	if _, isSystem := f.(frames.SystemFrame); isSystem {
		for _, e := range p.endpoints(dir) {
			if err := e.proc.ProcessFrame(ctx, f, dir); err != nil {
				return err
			}
		}
		return p.PushFrame(ctx, f, dir)
	}

	endpoints := p.endpoints(dir)
	main := p.downQueue
	if dir == processor.Upstream {
		main = p.upQueue
	}

	perBranch := make([][]frames.Frame, len(endpoints))
	var wg sync.WaitGroup
	for i, e := range endpoints {
		wg.Go(func() { perBranch[i] = p.waitForSync(ctx, e, main, f, dir) })
	}
	wg.Wait()

	if p.frameOrder == FrameOrderPipeline {
		seen := map[uint64]struct{}{}
		for _, branchFrames := range perBranch {
			for _, bf := range branchFrames {
				if _, dup := seen[bf.ID()]; dup {
					continue
				}
				seen[bf.ID()] = struct{}{}
				_ = p.PushFrame(ctx, bf, dir)
			}
		}
		return nil
	}

	p.drain(ctx, p.upQueue, processor.Upstream)
	p.drain(ctx, p.downQueue, processor.Downstream)
	return nil
}

// endpoints returns the ends of the branches a frame flowing in dir enters by: a
// downstream frame goes in at the sources, an upstream frame at the sinks.
func (p *SyncParallelPipeline) endpoints(dir processor.Direction) []syncEndpoint {
	if dir == processor.Upstream {
		return p.sinks
	}
	return p.sources
}

// waitForSync sends a frame into one branch and collects everything that comes
// back out of it, up to the sentinel that marks the branch done. In arrival
// order the frames go straight onto the shared queue; in branch order they are
// returned so the caller can release them branch by branch.
func (p *SyncParallelPipeline) waitForSync(
	ctx context.Context,
	e syncEndpoint,
	main *frameQueue,
	f frames.Frame,
	dir processor.Direction,
) []frames.Frame {
	var out []frames.Frame
	collect := func(fr frames.Frame) {
		if p.frameOrder == FrameOrderPipeline {
			out = append(out, fr)
			return
		}
		main.push(fr)
	}

	if err := e.proc.ProcessFrame(ctx, f, dir); err != nil {
		return out
	}

	// An EndFrame is its own sentinel: it is the last frame a branch handles, so
	// there is no point sending a SyncFrame in behind it.
	if _, isEnd := f.(*frames.EndFrame); isEnd {
		next, ok := e.queue.get(ctx)
		if !ok {
			return out
		}
		if _, isEndOut := next.(*frames.EndFrame); isEndOut {
			collect(next)
			return out
		}
		for {
			if _, isEndOut := next.(*frames.EndFrame); isEndOut {
				return out
			}
			collect(next)
			if next, ok = e.queue.get(ctx); !ok {
				return out
			}
		}
	}

	if err := e.proc.ProcessFrame(ctx, NewSyncFrame(), dir); err != nil {
		return out
	}
	for {
		next, ok := e.queue.get(ctx)
		if !ok {
			return out
		}
		if _, isSync := next.(*SyncFrame); isSync {
			return out
		}
		collect(next)
	}
}

// drain empties a shared queue, pushing what it holds in dir and dropping the
// duplicates that come of the same frame emerging from several branches.
func (p *SyncParallelPipeline) drain(ctx context.Context, q *frameQueue, dir processor.Direction) {
	seen := map[uint64]struct{}{}
	for {
		f, ok := q.tryGet()
		if !ok {
			return
		}
		if _, dup := seen[f.ID()]; dup {
			continue
		}
		seen[f.ID()] = struct{}{}
		_ = p.PushFrame(ctx, f, dir)
	}
}

// Processors returns the branches.
func (p *SyncParallelPipeline) Processors() []processor.Processor { return branchList(p.branches) }

// EntryProcessors returns the branch sources, which a frame entering the sync
// parallel pipeline is handed to directly.
func (p *SyncParallelPipeline) EntryProcessors() []processor.Processor {
	out := make([]processor.Processor, 0, len(p.sources))
	for _, e := range p.sources {
		out = append(out, e.proc)
	}
	return out
}

// ProcessorsWithMetrics returns the processors across every branch that report
// metrics.
func (p *SyncParallelPipeline) ProcessorsWithMetrics() []processor.Processor {
	var out []processor.Processor
	for _, b := range p.branches {
		out = append(out, b.ProcessorsWithMetrics()...)
	}
	return out
}

// Setup sets up the sync parallel pipeline and every branch.
func (p *SyncParallelPipeline) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	return p.eachBranch(func(b *Pipeline) error { return b.Setup(ctx, s) })
}

// Cleanup cleans up the sync parallel pipeline and every branch.
func (p *SyncParallelPipeline) Cleanup(ctx context.Context) error {
	_ = p.Base.Cleanup(ctx)
	return p.eachBranch(func(b *Pipeline) error { return b.Cleanup(ctx) })
}

// eachBranch runs fn against every branch at once and returns the first error.
func (p *SyncParallelPipeline) eachBranch(fn func(*Pipeline) error) error {
	errs := make([]error, len(p.branches))
	var wg sync.WaitGroup
	for i, b := range p.branches {
		wg.Go(func() { errs[i] = fn(b) })
	}
	wg.Wait()
	return errors.Join(errs...)
}
