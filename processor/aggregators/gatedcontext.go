package aggregators

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/notify"
)

// GatedContext holds the conversation back from the model until a notifier says
// to let it through.
//
// It is how a turn waits on something the pipeline cannot express as a frame
// order: a classifier deciding whether to answer at all, a lookup that has to
// finish first. Only the most recent context is held, because an older one
// describes a conversation that has since moved on, so what is released when the
// notifier fires is the conversation as it stands rather than a backlog of it.
type GatedContext struct {
	*processor.Base
	notifier notify.Notifier

	mu        sync.Mutex
	startOpen bool
	last      *frames.LLMContextFrame
	cancel    context.CancelFunc
	wg        sync.WaitGroup
}

// GatedContextConfig configures a GatedContext aggregator.
type GatedContextConfig struct {
	// Notifier releases the held conversation each time it is signaled.
	Notifier notify.Notifier
	// StartOpen lets the first conversation through without waiting, so the bot
	// can open the call before anything has had to decide.
	StartOpen bool
}

// NewGatedContext builds a gate over cfg.
func NewGatedContext(name string, cfg GatedContextConfig) *GatedContext {
	g := &GatedContext{notifier: cfg.Notifier, startOpen: cfg.StartOpen}
	g.Base = processor.New(name, g)
	return g
}

// ProcessFrame implements processor.Processor.
func (g *GatedContext) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir processor.Direction,
) error {
	if err := g.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}

	switch f := frame.(type) {
	case *frames.StartFrame:
		if err := g.PushFrame(ctx, frame, dir); err != nil {
			return err
		}
		g.start()
		return nil
	case *frames.EndFrame, *frames.CancelFrame:
		g.stop()
		return g.PushFrame(ctx, frame, dir)
	case *frames.LLMContextFrame:
		g.mu.Lock()
		if g.startOpen {
			g.startOpen = false
			g.mu.Unlock()
			return g.PushFrame(ctx, frame, dir)
		}
		// Only the latest is kept: an earlier one describes a conversation that
		// has already been added to.
		g.last = f
		g.mu.Unlock()
		return nil
	default:
		return g.PushFrame(ctx, frame, dir)
	}
}

// Cleanup implements processor.Processor.
func (g *GatedContext) Cleanup(ctx context.Context) error {
	g.stop()
	return g.Base.Cleanup(ctx)
}

// start begins waiting on the notifier, once.
func (g *GatedContext) start() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.cancel != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	g.cancel = cancel
	g.wg.Add(1)
	go g.wait(ctx)
}

// stop ends the waiting goroutine and waits for it to finish.
func (g *GatedContext) stop() {
	g.mu.Lock()
	cancel := g.cancel
	g.cancel = nil
	g.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	g.wg.Wait()
}

// wait releases the held conversation each time the notifier fires.
func (g *GatedContext) wait(ctx context.Context) {
	defer g.wg.Done()
	for {
		if !g.notifier.Wait(ctx) {
			return
		}
		g.mu.Lock()
		held := g.last
		g.last = nil
		g.mu.Unlock()
		if held == nil {
			continue
		}
		_ = g.PushFrame(ctx, held, processor.Downstream)
	}
}
