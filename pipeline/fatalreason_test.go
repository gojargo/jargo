package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// fatalOnText fails for good on any text.
type fatalOnText struct{ *processor.Base }

func (p *fatalOnText) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.TextFrame); ok {
		p.PushError(ctx, "Boo!", nil, true)
	}
	return p.PushFrame(ctx, f, dir)
}

// TestAFatalErrorCancelsWithItsReason checks the CancelFrame a fatal error
// reaching the start of the pipeline queues says why: "fatal error: " and the
// error.
func TestAFatalErrorCancelsWithItsReason(t *testing.T) {
	p := &fatalOnText{}
	p.Base = processor.New("ErrorProcessor", p)
	w := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{IdleTimeout: -1})
	var mu sync.Mutex
	var finished frames.Frame
	events.On(&w.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		finished = f
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	w.QueueFrame(frames.NewTextFrame("hello"))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	cf, ok := finished.(*frames.CancelFrame)
	if !ok {
		t.Fatalf("finished with %v, want a CancelFrame", finished)
	}
	if cf.Reason != "fatal error: Boo!" {
		t.Errorf("reason = %q, want %q", cf.Reason, "fatal error: Boo!")
	}
}
