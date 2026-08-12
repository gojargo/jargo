package processor

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
)

// IdleFrameProcessor calls back when nothing it is watching for has come
// through for a while.
//
// It forwards every frame untouched. What it adds is a clock: each frame it
// watches for restarts it, and the callback runs whenever the clock runs out.
// Watching for nothing in particular means any frame restarts it, which measures
// the pipeline going quiet; naming frame types measures the absence of those,
// which is how a bot notices a caller who has stopped speaking to it.
//
// The clock keeps running after the callback, so a pipeline that stays idle
// calls back once per timeout rather than only the first time.
type IdleFrameProcessor struct {
	*Base
	callback func(*IdleFrameProcessor)
	timeout  time.Duration
	types    []func(frames.Frame) bool

	mu     sync.Mutex
	seen   chan struct{}
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewIdleFrameProcessor builds a processor calling back when timeout passes with
// none of types coming through. With no types given, any frame at all restarts
// the clock. Build a matcher with FrameIs.
func NewIdleFrameProcessor(
	name string,
	timeout time.Duration,
	callback func(*IdleFrameProcessor),
	types ...func(frames.Frame) bool,
) *IdleFrameProcessor {
	p := &IdleFrameProcessor{
		callback: callback,
		timeout:  timeout,
		types:    types,
		seen:     make(chan struct{}, 1),
	}
	p.Base = New(name, p)
	return p
}

// ProcessFrame implements Processor.
func (p *IdleFrameProcessor) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir Direction,
) error {
	if err := p.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}
	if _, ok := frame.(*frames.StartFrame); ok {
		p.startWatching()
	}
	if err := p.PushFrame(ctx, frame, dir); err != nil {
		return err
	}

	// The clock is restarted after the frame has gone on, so what is measured is
	// the gap between frames leaving rather than the time this processor spent
	// on one.
	if len(p.types) == 0 {
		p.restart()
		return nil
	}
	for _, match := range p.types {
		if match(frame) {
			p.restart()
			return nil
		}
	}
	return nil
}

// Cleanup implements Processor.
func (p *IdleFrameProcessor) Cleanup(ctx context.Context) error {
	p.stopWatching()
	return p.Base.Cleanup(ctx)
}

// restart tells the watching goroutine a frame came through. It never blocks: a
// restart already pending says the same thing.
func (p *IdleFrameProcessor) restart() {
	select {
	case p.seen <- struct{}{}:
	default:
	}
}

// startWatching begins the clock, once.
func (p *IdleFrameProcessor) startWatching() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cancel != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	p.cancel = cancel
	p.wg.Add(1)
	go p.watch(ctx)
}

// stopWatching ends the clock and waits for the goroutine to finish.
func (p *IdleFrameProcessor) stopWatching() {
	p.mu.Lock()
	cancel := p.cancel
	p.cancel = nil
	p.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	p.wg.Wait()
}

// watch runs the clock, calling back each time it runs out.
func (p *IdleFrameProcessor) watch(ctx context.Context) {
	defer p.wg.Done()
	timer := time.NewTimer(p.timeout)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-p.seen:
			// A frame came through, so the wait starts again from here.
		case <-timer.C:
			p.callback(p)
		}
		timer.Reset(p.timeout)
	}
}
