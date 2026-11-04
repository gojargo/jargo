package pipeline

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// slowSetup is a processor still inside its own Setup when the StartFrame can
// first reach it, recording whether it handled the StartFrame before its Setup
// returned.
type slowSetup struct {
	*processor.Base
	release      chan struct{}
	setupDone    atomic.Bool
	startedEarly atomic.Bool
	started      chan struct{}
}

func newSlowSetup() *slowSetup {
	p := &slowSetup{release: make(chan struct{}), started: make(chan struct{})}
	p.Base = processor.New("SlowSetup", p)
	return p
}

func (p *slowSetup) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	<-p.release
	p.setupDone.Store(true)
	return nil
}

func (p *slowSetup) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		p.startedEarly.Store(!p.setupDone.Load())
		close(p.started)
	}
	return p.PushFrame(ctx, f, dir)
}

// The processors are held before the pipeline lets the StartFrame in, so it
// cannot reach one whose Setup has not returned, however soon after the
// release it arrives. Releasing the pipeline first left a window in which the
// StartFrame passed straight into a processor still setting up, which read its
// state while Setup was still writing it.
func TestTheStartFrameWaitsForAProcessorStillSettingUp(t *testing.T) {
	p := newSlowSetup()
	t.Cleanup(func() { _ = p.Cleanup(context.Background()) })
	var tracker setupTracker

	// The release hands the StartFrame on at once, as the pipeline's own input
	// goroutine can the moment the pipeline lets it in.
	release := func() {
		_ = p.QueueFrame(t.Context(), frames.NewStartFrame(), processor.Downstream)
	}
	done := make(chan struct{})
	go func() {
		setupProcessors(t.Context(), &tracker, release, []processor.Processor{p}, processor.Setup{})
		close(done)
	}()

	select {
	case <-p.started:
		t.Fatal("the StartFrame was handled while the processor was still setting up")
	case <-time.After(100 * time.Millisecond):
	}
	close(p.release)
	<-done
	select {
	case <-p.started:
	case <-time.After(3 * time.Second):
		t.Fatal("the StartFrame was never handled once setup was done")
	}
	if p.startedEarly.Load() {
		t.Error("the StartFrame was handled before Setup returned")
	}
}
