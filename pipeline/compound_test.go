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

// metered is a stand-in for a service: a processor that reports metrics.
type metered struct {
	*processor.Base
}

func newMetered(name string) *metered {
	m := &metered{}
	m.Base = processor.New(name, m)
	return m
}

func (m *metered) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := m.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return m.PushFrame(ctx, f, dir)
}

func (m *metered) CanGenerateMetrics() bool { return true }

func TestPlainProcessorIsNotCompound(t *testing.T) {
	e := newEcho()
	if got := e.Processors(); got != nil {
		t.Errorf("Processors() = %v, want nil", got)
	}
	if got := e.EntryProcessors(); got != nil {
		t.Errorf("EntryProcessors() = %v, want nil", got)
	}
	if got := e.ProcessorsWithMetrics(); got != nil {
		t.Errorf("ProcessorsWithMetrics() = %v, want nil", got)
	}
	if e.CanGenerateMetrics() {
		t.Error("CanGenerateMetrics() = true, want false")
	}
}

func TestPipelineProcessorsIncludeSourceAndSink(t *testing.T) {
	inner := newEcho()
	p := pipeline.New(inner)

	got := p.Processors()
	if len(got) != 3 {
		t.Fatalf("Processors() has %d entries, want 3 (source, echo, sink)", len(got))
	}
	if got[1] != processor.Processor(inner) {
		t.Errorf("Processors()[1] = %s, want the echo", got[1].Name())
	}

	entry := p.EntryProcessors()
	if len(entry) != 1 || entry[0] != got[0] {
		t.Errorf("EntryProcessors() = %v, want just the source", entry)
	}
}

func TestPipelineProcessorsWithMetricsRecurses(t *testing.T) {
	// A metered processor at the top level and one inside a nested pipeline must
	// both be found.
	deep := newMetered("Deep")
	top := newMetered("Top")
	p := pipeline.New(top, newEcho(), pipeline.New(deep))

	names := metricsNames(p.ProcessorsWithMetrics())
	if len(names) != 2 {
		t.Fatalf("ProcessorsWithMetrics() = %v, want 2 entries", names)
	}
	if names[0] != top.Name() || names[1] != deep.Name() {
		t.Errorf("ProcessorsWithMetrics() = %v, want [%s %s]", names, top.Name(), deep.Name())
	}
}

func TestParallelProcessorsAndMetrics(t *testing.T) {
	a, b := newMetered("A"), newMetered("B")
	pp, err := pipeline.NewParallel(
		[]processor.Processor{a},
		[]processor.Processor{b, newEcho()},
	)
	if err != nil {
		t.Fatalf("NewParallel: %v", err)
	}

	if got := len(pp.Processors()); got != 2 {
		t.Errorf("Processors() has %d entries, want 2 branches", got)
	}
	if got := len(pp.EntryProcessors()); got != 2 {
		t.Errorf("EntryProcessors() has %d entries, want 2 branches", got)
	}

	names := metricsNames(pp.ProcessorsWithMetrics())
	if len(names) != 2 || names[0] != a.Name() || names[1] != b.Name() {
		t.Errorf("ProcessorsWithMetrics() = %v, want [%s %s]", names, a.Name(), b.Name())
	}
}

func TestSyncParallelProcessorsAndMetrics(t *testing.T) {
	a, b := newMetered("A"), newMetered("B")
	spp, err := pipeline.NewSyncParallel(pipeline.FrameOrderArrival,
		[]processor.Processor{a},
		[]processor.Processor{b},
	)
	if err != nil {
		t.Fatalf("NewSyncParallel: %v", err)
	}

	if got := len(spp.Processors()); got != 2 {
		t.Errorf("Processors() has %d entries, want 2 branches", got)
	}
	// Unlike a parallel pipeline, a frame is handed straight to the branch
	// sources rather than queued on the branches.
	entry := spp.EntryProcessors()
	if len(entry) != 2 {
		t.Fatalf("EntryProcessors() has %d entries, want 2 sources", len(entry))
	}
	for _, e := range entry {
		if got := len(e.Processors()); got != 0 {
			t.Errorf("entry %s has %d sub-processors, want a source", e.Name(), got)
		}
	}

	names := metricsNames(spp.ProcessorsWithMetrics())
	if len(names) != 2 || names[0] != a.Name() || names[1] != b.Name() {
		t.Errorf("ProcessorsWithMetrics() = %v, want [%s %s]", names, a.Name(), b.Name())
	}
}

func TestTaskSendsInitialEmptyMetrics(t *testing.T) {
	// Two metrics-reporting processors, one of them inside a nested pipeline, so
	// the test covers what the single frame is for: every processor at once.
	svc, deep := newMetered("Svc"), newMetered("Deep")
	p := pipeline.New(svc, newEcho(), pipeline.New(deep))

	var mu sync.Mutex
	var got []*frames.MetricsFrame
	task := pipeline.NewWorker(p, pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics: true,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if mf, ok := f.(*frames.MetricsFrame); ok {
			mu.Lock()
			got = append(got, mf)
			mu.Unlock()
		}
	})

	runToEnd(t, task)

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 {
		t.Fatalf("got %d MetricsFrames, want 1 carrying every processor", len(got))
	}

	// A zeroed time to first byte and processing time for each, in pipeline order.
	want := []struct {
		processor string
		kind      string
	}{
		{svc.Name(), "ttfb"},
		{svc.Name(), "processing"},
		{deep.Name(), "ttfb"},
		{deep.Name(), "processing"},
	}
	data := got[0].Data
	if len(data) != len(want) {
		t.Fatalf("frame carries %d measurements, want %d: %+v", len(data), len(want), data)
	}
	for i, w := range want {
		if p := data[i].MetricsProcessor(); p != w.processor {
			t.Errorf("measurement %d processor = %q, want %q", i, p, w.processor)
		}
		switch d := data[i].(type) {
		case frames.TTFBMetricsData:
			if w.kind != "ttfb" {
				t.Errorf("measurement %d is TTFB, want %s", i, w.kind)
			}
			if d.Value != 0 {
				t.Errorf("measurement %d TTFB = %v, want zero", i, d.Value)
			}
		case frames.ProcessingMetricsData:
			if w.kind != "processing" {
				t.Errorf("measurement %d is processing, want %s", i, w.kind)
			}
			if d.Value != 0 {
				t.Errorf("measurement %d processing = %v, want zero", i, d.Value)
			}
		default:
			t.Errorf("measurement %d is %T, want a zeroed TTFB or processing time", i, d)
		}
	}
}

func TestTaskSkipsInitialEmptyMetrics(t *testing.T) {
	off := false
	cases := map[string]pipeline.WorkerConfig{
		"metrics disabled":        {},
		"initial metrics refused": {Params: pipeline.Params{EnableMetrics: true, SendInitialEmptyMetrics: &off}},
	}

	for name, params := range cases {
		t.Run(name, func(t *testing.T) {
			var mu sync.Mutex
			var n int
			worker := pipeline.NewWorker(pipeline.New(newMetered("Svc")), params)
			events.On(&worker.Registry, pipeline.EventFrameReachedDownstream,
				func(_ context.Context, f frames.Frame) {
					if _, ok := f.(*frames.MetricsFrame); ok {
						mu.Lock()
						n++
						mu.Unlock()
					}
				})
			runToEnd(t, worker)

			mu.Lock()
			defer mu.Unlock()
			if n != 0 {
				t.Errorf("got %d MetricsFrames, want none", n)
			}
		})
	}
}

// runToEnd runs a task until it stops.
func runToEnd(t *testing.T, task *pipeline.Worker) {
	t.Helper()

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.StopWhenDone()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("task run error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

// metricsNames is the names of the processors reporting metrics, in order.
func metricsNames(procs []processor.Processor) []string {
	out := make([]string, 0, len(procs))
	for _, p := range procs {
		out = append(out, p.Name())
	}
	return out
}
