package pipeline_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// Ported from upstream (test_getter_returns_constructor_value and
// test_default_app_resources_is_none).
func TestWorkerAppResources(t *testing.T) {
	shared := map[string]string{"db": "orders"}
	w := pipeline.NewWorker(pipeline.New(), pipeline.WorkerConfig{AppResources: shared})
	if got, ok := w.AppResources().(map[string]string); !ok || got["db"] != "orders" {
		t.Errorf("AppResources = %v, want the configured value", w.AppResources())
	}
	if got := pipeline.NewWorker(pipeline.New(), pipeline.WorkerConfig{}).AppResources(); got != nil {
		t.Errorf("AppResources = %v, want nil by default", got)
	}
}

// resourceReader records what its pipeline shares, once it has started.
type resourceReader struct {
	*processor.Base
	seen chan any
}

func (r *resourceReader) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if _, ok := f.(*frames.StartFrame); ok {
		r.seen <- r.Running().AppResources()
	}
	return r.Base.ProcessFrame(ctx, f, dir)
}

// Ported from upstream (test_processor_can_reach_pipeline_task_and_app_resources).
// A processor reaches the worker running it, and through it the resources.
func TestProcessorCanReachAppResources(t *testing.T) {
	shared := &struct{ name string }{"shared"}
	r := &resourceReader{seen: make(chan any, 1)}
	r.Base = processor.New("Reader", r)
	w := pipeline.NewWorker(pipeline.New(r), pipeline.WorkerConfig{AppResources: shared})
	runDone := make(chan error, 1)
	go func() { runDone <- w.Run(context.Background()) }()
	defer func() {
		w.Cancel(context.Background(), "")
		<-runDone
	}()
	select {
	case got := <-r.seen:
		if got != shared {
			t.Errorf("AppResources = %v, want the configured value", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the processor never started")
	}
}
