package rtvi_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

// failingService fails for good on the text "FAIL", when told to.
type failingService struct {
	*processor.Base
	fail bool
}

func newFailingService(name string, fail bool) *failingService {
	s := &failingService{fail: fail}
	s.Base = processor.New(name, s)
	return s
}

func (s *failingService) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok && dir == processor.Downstream && s.fail && tf.Text == "FAIL" {
		//nolint:err113 // a one-off cause
		s.PushError(ctx, "primary failed", errors.New("boom"), false, processor.ForceTreatAsPermanent())
		return nil
	}
	return s.PushFrame(ctx, f, dir)
}

// TestAnErrorTheFailoverAbsorbedIsNotSentToTheClient checks the client is told
// of the errors that reach the RTVI processor at the head of the pipeline, not
// of every error raised anywhere: one a switcher recovered from by failing
// over never gets there, so the client is not told of a failure it was spared.
func TestAnErrorTheFailoverAbsorbedIsNotSentToTheClient(t *testing.T) {
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{newFailingService("primary", true), newFailingService("backup", false)},
		pipeline.NewFailoverStrategy)
	if err != nil {
		t.Fatal(err)
	}
	proc := rtvi.NewProcessor()
	var mu sync.Mutex
	var msgs []rtvi.Message
	task := pipeline.NewWorker(pipeline.New(proc, sw, newPlayback()), pipeline.WorkerConfig{
		Observers:               []pipeline.Observer{rtvi.NewObserver(proc)},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				mu.Lock()
				msgs = append(msgs, msg)
				mu.Unlock()
			}
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTextFrame("FAIL"))
	time.Sleep(400 * time.Millisecond)
	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	defer mu.Unlock()
	for _, m := range msgs {
		if m.Type == rtvi.TypeError {
			t.Errorf("the client was sent %+v for an error the failover absorbed", m.Data)
		}
	}
	if !strings.HasPrefix(sw.ActiveService().Name(), "backup") {
		t.Errorf("active %s, want the backup: the failover did not happen", sw.ActiveService().Name())
	}
}
