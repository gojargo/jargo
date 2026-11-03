// Package run implements each kind of benchmark against jargo's public API. The
// jbench command and the testing.B wrappers both drive it, so the numbers the
// comparison reports and the ones a regression check reports come from the same
// code.
package run

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// stopGrace bounds how long a session that was told to stop is waited for
// before it is reported as stuck. A benchmark must end rather than hang.
const stopGrace = 10 * time.Second

// ErrStuck is returned when a session does not finish within stopGrace of being
// told to.
var ErrStuck = errors.New("session did not finish")

// passthrough forwards every frame to its neighbor and does no work of its own,
// so a chain of them costs only what the framework adds: the queue, the
// goroutine handoff and the dispatch at each hop.
type passthrough struct {
	*processor.Base
}

func newPassthrough() *passthrough {
	p := &passthrough{}
	p.Base = processor.New("Passthrough", p)
	return p
}

// ProcessFrame applies the base handling every processor needs, then forwards
// the frame.
func (p *passthrough) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// Session is one bot, run the way an application runs one: a runner holding a
// pipeline worker, whose pipeline is a chain of pass-through processors. Every
// TextFrame reaching the end of the chain is reported on the arrival channel.
//
// The worker is configured to match the Python side exactly: no turn tracking,
// no idle timeout, no RTVI, no observers, and only TextFrames reported.
type Session struct {
	runner *workers.Runner
	worker *pipeline.Worker

	arrived chan<- struct{}
	started chan struct{}
	done    chan struct{}
	// quit is closed when the session is abandoned, so a reporting handler
	// blocked on a full arrival channel lets the pipeline shut down.
	quit     chan struct{}
	quitOnce sync.Once
	runErr   error
}

// NewSession builds a session with a chain of depth processors, reporting each
// arrival on arrived. Several sessions may share one channel. Nothing runs
// until Start.
func NewSession(depth int, arrived chan<- struct{}) *Session {
	procs := make([]processor.Processor, depth)
	for i := range procs {
		procs[i] = newPassthrough()
	}
	off := false
	worker := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		EnableTurnTracking:      &off,
		EnableRTVI:              &off,
		IdleTimeout:             -1,
		ReachedDownstreamFilter: pipeline.FrameTypes(&frames.TextFrame{}),
	})
	s := &Session{
		runner:  workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &off}),
		worker:  worker,
		arrived: arrived,
		started: make(chan struct{}),
		done:    make(chan struct{}),
		quit:    make(chan struct{}),
	}

	var startedOnce sync.Once
	events.On(worker.Events(), pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) {
		startedOnce.Do(func() { close(s.started) })
	})
	events.On(worker.Events(), pipeline.EventFrameReachedDownstream, func(context.Context, frames.Frame) {
		select {
		case s.arrived <- struct{}{}:
		case <-s.quit:
		}
	})
	return s
}

// Start adds the worker to the runner and runs it in the background. It returns
// at once; WaitStarted reports when the pipeline is ready.
func (s *Session) Start(ctx context.Context) {
	s.runner.AddWorkers(ctx, s.worker)
	go func() {
		defer close(s.done)
		s.runErr = s.runner.Run(context.WithoutCancel(ctx), workers.RunOptions{})
	}()
}

// WaitStarted waits for the StartFrame to have reached the end of the pipeline.
func (s *Session) WaitStarted(ctx context.Context) error {
	select {
	case <-s.started:
		return nil
	case <-s.done:
		return fmt.Errorf("session finished before it started: %w", s.runErr)
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Send queues a frame at the head of the pipeline.
func (s *Session) Send(f frames.Frame) {
	s.worker.QueueFrame(f)
}

// Stop ends the session gracefully: an EndFrame follows whatever is in flight,
// and the runner ends with the worker. A session that does not finish within
// stopGrace is canceled.
func (s *Session) Stop(ctx context.Context) error {
	s.worker.StopWhenDone()
	if err := s.wait(ctx); !errors.Is(err, ErrStuck) {
		return err
	}
	return s.Cancel(ctx)
}

// Cancel abandons the session: it stops reporting arrivals and cancels the
// runner, for a sample that ran out of time.
func (s *Session) Cancel(ctx context.Context) error {
	s.quitOnce.Do(func() { close(s.quit) })
	s.runner.Cancel(ctx, "benchmark budget exceeded")
	return s.wait(ctx)
}

// wait waits for the runner to return, for at most stopGrace.
func (s *Session) wait(ctx context.Context) error {
	timer := time.NewTimer(stopGrace)
	defer timer.Stop()
	select {
	case <-s.done:
		return s.runErr
	case <-timer.C:
		return ErrStuck
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Wait waits for one arrival on ch.
func Wait(ctx context.Context, ch <-chan struct{}) error {
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
