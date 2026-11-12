package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// panicPassthrough forwards every frame.
type panicPassthrough struct{ *processor.Base }

func (p *panicPassthrough) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// raising is an observer that panics on the first text it sees.
type raising struct {
	mu    sync.Mutex
	texts []string
}

func (o *raising) ObserveEveryPush() bool { return false }

func (o *raising) OnPushFrame(d processor.FramePushed) {
	tf, ok := d.Frame.(*frames.TextFrame)
	if !ok {
		return
	}
	o.mu.Lock()
	o.texts = append(o.texts, tf.Text)
	n := len(o.texts)
	o.mu.Unlock()
	if n == 1 {
		panic("observer failed") //nolint:forbidigo // the observer's bug under test
	}
}

// TestAnObserverThatPanicsKeepsReceivingFrames checks an observer that panics
// is logged and given the next frame all the same, rather than ending the
// process.
func TestAnObserverThatPanicsKeepsReceivingFrames(t *testing.T) {
	p := &panicPassthrough{}
	p.Base = processor.New("Passthrough", p)
	o := &raising{}
	off := false
	w := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		Name: "test", IdleTimeout: -1, EnableRTVI: &off, Observers: []pipeline.Observer{o},
	})
	done := make(chan struct{})
	go func() { defer close(done); _ = w.Run(context.Background()) }()
	w.QueueFrame(frames.NewTextFrame("one"))
	w.QueueFrame(frames.NewTextFrame("two"))
	time.Sleep(200 * time.Millisecond)
	w.StopWhenDone()
	<-done
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.texts) != 2 {
		t.Errorf("texts = %v, want [one two]", o.texts)
	}
}
