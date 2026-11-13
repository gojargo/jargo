package stt

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// startProbe passes every frame on and says when the StartFrame went past it.
type startProbe struct {
	*processor.Base
	started chan struct{}
}

func newStartProbe() *startProbe {
	p := &startProbe{started: make(chan struct{})}
	p.Base = processor.New("StartProbe", p)
	return p
}

func (p *startProbe) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		close(p.started)
	}
	return p.PushFrame(ctx, f, dir)
}

// blockedSetup is still setting up when the pipeline gives up on it: once the
// processors ahead of it have started, it waits for the context it was set up
// with to end.
type blockedSetup struct {
	*processor.Base
	after <-chan struct{}
}

func newBlockedSetup(after <-chan struct{}) *blockedSetup {
	b := &blockedSetup{after: after}
	b.Base = processor.New("BlockedSetup", b)
	return b
}

func (b *blockedSetup) Setup(ctx context.Context, s processor.Setup) error {
	if err := b.Base.Setup(ctx, s); err != nil {
		return err
	}
	select {
	case <-b.after:
	case <-ctx.Done():
		return ctx.Err()
	}
	<-ctx.Done()
	return ctx.Err()
}

func (b *blockedSetup) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return b.PushFrame(ctx, f, dir)
}

// A processor after the service times out setting up, so the pipeline the
// service already started in is only cleaned up: no EndFrame or CancelFrame
// ever reaches it. The cleanup still stops the task that transcribes its
// segments.
func TestCleanupStopsTheTranscriptionTask(t *testing.T) {
	t.Parallel()

	svc := NewSegment("FakeSTT", &settingsTranscriber{}, 16000)
	probe := newStartProbe()
	task := pipeline.NewWorker(pipeline.New(svc, probe, newBlockedSetup(probe.started)),
		pipeline.WorkerConfig{SetupTimeout: 200 * time.Millisecond})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	select {
	case <-probe.started:
	case <-time.After(3 * time.Second):
		t.Fatal("the service never started")
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the run did not end after setting up timed out")
	}

	svc.mu.Lock()
	running := svc.segmentCancel != nil
	svc.mu.Unlock()
	if running {
		t.Error("the transcription task is still running after cleanup")
	}
	stopped := make(chan struct{})
	go func() {
		svc.segmentWG.Wait()
		close(stopped)
	}()
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Error("the transcription task did not stop at cleanup")
	}
}
