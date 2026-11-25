package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// connectingService is still connecting while it is set up: its setup returns
// only when the context it was handed ends. It records whether that happened,
// and every frame it is handed.
type connectingService struct {
	*processor.Base
	setupStarted chan struct{}
	stopped      chan struct{}

	mu   sync.Mutex
	seen []frames.Frame
}

func newConnectingService() *connectingService {
	c := &connectingService{setupStarted: make(chan struct{}), stopped: make(chan struct{})}
	c.Base = processor.New("ConnectingService", c)
	return c
}

func (c *connectingService) Setup(ctx context.Context, s processor.Setup) error {
	if err := c.Base.Setup(ctx, s); err != nil {
		return err
	}
	close(c.setupStarted)
	<-ctx.Done()
	close(c.stopped)
	return ctx.Err()
}

func (c *connectingService) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	c.mu.Lock()
	c.seen = append(c.seen, f)
	c.mu.Unlock()
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return c.PushFrame(ctx, f, dir)
}

func (c *connectingService) frames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.seen...)
}

// TestCancelTakesEffectWhileProcessorsAreSettingUp checks a cancel during setup
// ends the run at once, as when a caller hangs up while a service is still
// connecting, rather than once every processor has finished setting up. The
// processor still setting up is told to stop, never gets the StartFrame, and is
// only cleaned up: no CancelFrame is pushed into a pipeline that never started.
func TestCancelTakesEffectWhileProcessorsAreSettingUp(t *testing.T) {
	svc := newConnectingService()
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		SetupTimeout:  time.Minute,
		CancelTimeout: 100 * time.Millisecond,
	})
	var finished []frames.Frame
	var mu sync.Mutex
	event.On(&task.Registry, pipeline.EventPipelineFinished, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		finished = append(finished, f)
		mu.Unlock()
	})

	done := runTask(t, task)
	started(t, svc.setupStarted)
	task.Cancel(context.Background(), "the caller hung up")
	waitDoneWithin(t, done, 3*time.Second)

	select {
	case <-svc.stopped:
	default:
		t.Error("the processor still setting up was not told to stop")
	}
	if got := svc.frames(); len(got) != 0 {
		t.Errorf("the processor was handed %v, want nothing: the pipeline never started", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(finished) != 0 {
		t.Errorf("the pipeline finished on %v, want it only cleaned up", finished)
	}
}
