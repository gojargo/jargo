package processor_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// timingProcessor counts how often its clocks were stopped.
type timingProcessor struct {
	*processor.Base
	stopped int
}

func (p *timingProcessor) StopAllMetrics(context.Context) { p.stopped++ }

// An interruption stops a processor's clocks, whether it reaches the processor
// as a frame or the processor broadcasts it: what they were measuring will not
// finish.
func TestInterruptionStopsAllMetrics(t *testing.T) {
	ctx := context.Background()
	p := &timingProcessor{}
	p.Base = processor.New("Timing", p)
	if err := p.Setup(ctx, processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	t.Cleanup(func() { _ = p.Cleanup(ctx) })
	startBase(t, p.Base)

	if err := p.Base.ProcessFrame(context.Background(), frames.NewInterruptionFrame(), processor.Downstream); err != nil {
		t.Fatalf("interruption: %v", err)
	}
	if p.stopped != 1 {
		t.Fatalf("clocks stopped %d times on an interruption frame, want 1", p.stopped)
	}
	if err := p.BroadcastInterruption(context.Background()); err != nil {
		t.Fatalf("broadcast: %v", err)
	}
	if p.stopped != 2 {
		t.Errorf("clocks stopped %d times after a broadcast interruption, want 2", p.stopped)
	}
}
