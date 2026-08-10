package service_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
)

// describer is a minimal service: it says nothing and does nothing but describe
// itself, which is all this test is about.
type describer struct {
	*service.Base
}

func newDescriber(name string) *describer {
	d := &describer{}
	d.Base = service.New(name, d)
	return d
}

func (d *describer) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := d.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return d.PushFrame(ctx, f, dir)
}

// ServiceMetadataFrame implements service.MetadataDescriber.
func (d *describer) ServiceMetadataFrame() frames.ServiceMetadata {
	return frames.NewServiceMetadataFrame(d.Name())
}

// TestASwitchedServiceDescribesItself checks the whole path a real service
// takes: it describes itself once the pipeline starts, only the active one is
// heard, and switching makes the new one describe itself so what the pipeline
// knows is never left describing a service that is no longer in use.
func TestASwitchedServiceDescribesItself(t *testing.T) {
	a, b := newDescriber("A"), newDescriber("B")
	sw, err := pipeline.NewServiceSwitcher(
		[]processor.Processor{a, b}, pipeline.NewManualStrategy,
	)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	var (
		mu   sync.Mutex
		seen []string
	)
	task := pipeline.NewTask(pipeline.New(sw), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if mf, ok := f.(frames.ServiceMetadata); ok {
				mu.Lock()
				seen = append(seen, mf.Service())
				mu.Unlock()
			}
		},
	})
	done := make(chan struct{})
	go func() { _ = task.Run(context.Background()); close(done) }()
	defer func() {
		task.Cancel()
		<-done
	}()

	read := func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), seen...)
	}

	time.Sleep(250 * time.Millisecond)
	if got := read(); len(got) != 1 || got[0] != a.Name() {
		t.Fatalf("metadata at startup = %v, want just the active service %q", got, a.Name())
	}

	if !sw.SwitchTo(b) {
		t.Fatal("SwitchTo(b) = false, want the switch accepted")
	}
	time.Sleep(250 * time.Millisecond)

	got := read()
	if len(got) != 2 || got[1] != b.Name() {
		t.Errorf("metadata after the switch = %v, want it ending with %q", got, b.Name())
	}
}
