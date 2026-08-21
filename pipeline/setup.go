package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// setupTracker remembers which processors finished being set up.
//
// Setting up is abandoned when it takes too long, and the pipeline is torn down
// straight after. A processor still inside its own setup then has nothing
// ordering that work against its cleanup, so cleaning it up would run the two
// against each other. The tracker is what tells them apart: a processor that
// finished is cleaned up, and one still stuck is left where it is.
//
// Embed it in a pipeline so its setup and its cleanup share one.
type setupTracker struct {
	mu    sync.Mutex
	began map[uint64]bool
	done  map[uint64]bool
}

// begin records that procs are about to be set up.
func (t *setupTracker) begin(procs []processor.Processor) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.began = make(map[uint64]bool, len(procs))
	t.done = make(map[uint64]bool, len(procs))
	for _, p := range procs {
		t.began[p.ID()] = true
	}
}

// finish records that p has finished being set up.
func (t *setupTracker) finish(p processor.Processor) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.done[p.ID()] = true
}

// settled reports whether p can safely be cleaned up.
//
// It can when it finished being set up, and when it was never set up here at
// all, in which case there is no setup to race. A processor holding others is
// settled too: the state its own setup touches is guarded, and the rule is what
// its cleanup applies in turn to the processors below it, so the ones that did
// come up are still released.
func (t *setupTracker) settled(p processor.Processor) bool {
	if len(p.Processors()) > 0 {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return !t.began[p.ID()] || t.done[p.ID()]
}

// setupProcessors sets procs up concurrently.
//
// A processor connects and does its other slow start-up work while it is set up,
// so running them together means the pipeline pays for the slowest processor
// rather than for all of them added together.
//
// A processor that fails to set up reports it as an error and leaves the rest of
// the pipeline to carry on, the same way a failure while handling a frame is
// reported. Every failure is reported, not just the first, so one never hides
// another.
//
// The failure is permanent whatever caused it, since setting up is not attempted
// again: the processor is left half-built for the rest of the session, so it
// loses its usability and a switcher can move off it before the pipeline starts.
func setupProcessors(
	ctx context.Context, t *setupTracker, procs []processor.Processor, s processor.Setup,
) {
	t.begin(procs)

	var wg sync.WaitGroup
	for _, p := range procs {
		wg.Go(func() {
			startedAt := time.Now()
			if err := p.Setup(ctx, s); err != nil {
				ef := frames.NewErrorFrame(fmt.Sprintf("error setting up the processor: %v", err))
				ef.Err = err
				ef.Source = p
				p.PushErrorFrame(ctx, ef, true)
			}
			t.finish(p)
			processor.NotifyProcessorSetup(s.Observers, processor.ProcessorSetUp{
				Processor:  p,
				StartedAt:  startedAt,
				FinishedAt: time.Now(),
			})
		})
	}
	wg.Wait()
}

// cleanupProcessors cleans procs up concurrently.
//
// Tearing down is best effort: a processor that fails is logged and the rest are
// still released. Cleaning up runs after a processor failed to set up, or was
// abandoned part way through being set up, so some of them are released from a
// state they never finished reaching, and one still inside its own setup is left
// alone rather than torn down underneath itself.
func cleanupProcessors(ctx context.Context, t *setupTracker, procs []processor.Processor) {
	var wg sync.WaitGroup
	for _, p := range procs {
		if !t.settled(p) {
			slog.WarnContext(ctx, "leaving a processor that never finished setting up",
				"processor", p.Name())
			continue
		}
		wg.Go(func() {
			if err := p.Cleanup(ctx); err != nil {
				slog.ErrorContext(ctx, "cleaning a processor up failed",
					"processor", p.Name(), "error", err)
			}
		})
	}
	wg.Wait()
}
