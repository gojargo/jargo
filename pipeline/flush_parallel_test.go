package pipeline_test

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// flushLeg is one pass of a flush through a processor: its direction, and
// whether it was the returning probe.
type flushLeg struct {
	dir       processor.Direction
	returning bool
}

// flushLegs is the trip every flush makes past a processor: down, up, and down
// again as the returning probe.
func flushLegs() []flushLeg {
	return []flushLeg{
		{processor.Downstream, false},
		{processor.Upstream, false},
		{processor.Downstream, true},
	}
}

// flushRecorder records the legs of each flush that pass through it.
type flushRecorder struct {
	*processor.Base
	mu    sync.Mutex
	order []uint64
	legs  map[uint64][]flushLeg
}

func newFlushRecorder(name string) *flushRecorder {
	p := &flushRecorder{legs: map[uint64][]flushLeg{}}
	p.Base = processor.New(name, p)
	return p
}

func (p *flushRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if probe, ok := f.(*frames.PipelineFlushFrame); ok {
		p.mu.Lock()
		if _, seen := p.legs[probe.FlushID]; !seen {
			p.order = append(p.order, probe.FlushID)
		}
		p.legs[probe.FlushID] = append(p.legs[probe.FlushID], flushLeg{dir, probe.Returning})
		p.mu.Unlock()
	}
	return p.PushFrame(ctx, f, dir)
}

// trips returns the legs recorded for each flush, in the order the flushes
// first passed.
func (p *flushRecorder) trips() [][]flushLeg {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([][]flushLeg, 0, len(p.order))
	for _, id := range p.order {
		out = append(out, p.legs[id])
	}
	return out
}

// flushThrough runs a pipeline of p, flushes it the given number of times at
// once, and returns each flush's result.
func flushThrough(t *testing.T, p processor.Processor, flushes int) []error {
	t.Helper()
	up := make(chan struct{})
	var once sync.Once
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			once.Do(func() { close(up) })
		}
	})
	done := runTask(t, task)
	started(t, up)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	results := make([]error, flushes)
	var wg sync.WaitGroup
	for i := range flushes {
		wg.Go(func() { results[i] = task.Flush(ctx) })
	}
	wg.Wait()

	task.StopWhenDone()
	waitDone(t, done)
	return results
}

// TestFlushThroughParallelPipeline checks that every branch sees every leg of
// each flush. A parallel pipeline passes on the first copy of a frame its
// branches emit and drops the rest, so a probe that went down through it would
// have been dropped on its way back up, and the flush would never settle.
func TestFlushThroughParallelPipeline(t *testing.T) {
	branches := []*flushRecorder{newFlushRecorder("BranchA"), newFlushRecorder("BranchB")}
	par, err := pipeline.NewParallel([]processor.Processor{branches[0]}, []processor.Processor{branches[1]})
	if err != nil {
		t.Fatal(err)
	}

	results := flushThrough(t, par, 2)

	for i, err := range results {
		if err != nil {
			t.Errorf("flush %d = %v, want it settled", i, err)
		}
	}
	for _, branch := range branches {
		if got := branch.trips(); !reflect.DeepEqual(got, [][]flushLeg{flushLegs(), flushLegs()}) {
			t.Errorf("%s saw %v, want each of the two flushes down, up and down again", branch.Name(), got)
		}
	}
}

// TestFlushThroughNestedServiceSwitchers checks that only the active service
// sees the flush.
func TestFlushThroughNestedServiceSwitchers(t *testing.T) {
	active, inactive, outer := newFlushRecorder("Active"), newFlushRecorder("Inactive"), newFlushRecorder("Outer")
	inner, err := pipeline.NewServiceSwitcher([]processor.Processor{active, inactive}, nil)
	if err != nil {
		t.Fatal(err)
	}
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{inner, outer}, nil)
	if err != nil {
		t.Fatal(err)
	}

	results := flushThrough(t, sw, 1)

	if results[0] != nil {
		t.Errorf("flush = %v, want it settled", results[0])
	}
	if got := active.trips(); !reflect.DeepEqual(got, [][]flushLeg{flushLegs()}) {
		t.Errorf("active service saw %v, want the flush down, up and down again", got)
	}
	if got := inactive.trips(); len(got) != 0 {
		t.Errorf("inactive service saw %v, want nothing", got)
	}
	if got := outer.trips(); len(got) != 0 {
		t.Errorf("outer inactive service saw %v, want nothing", got)
	}
}

// TestFlushWithProbeReportsItFlushed checks a flush started with the caller's
// probe settles under its FlushID, and the worker says so once.
func TestFlushWithProbeReportsItFlushed(t *testing.T) {
	up := make(chan struct{})
	var once sync.Once
	task := pipeline.NewWorker(pipeline.New(processor.NewIdentityFilter("Identity")), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			once.Do(func() { close(up) })
		}
	})
	flushed := make(chan uint64, 4)
	event.On(&task.Registry, pipeline.EventPipelineFlushed, func(_ context.Context, f *frames.PipelineFlushFrame) {
		flushed <- f.FlushID
	})
	done := runTask(t, task)
	started(t, up)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	probe := frames.NewPipelineFlushFrame()
	err := task.FlushWithProbe(ctx, probe)

	task.StopWhenDone()
	waitDone(t, done)
	if err != nil {
		t.Fatalf("flush = %v, want it settled", err)
	}
	close(flushed)
	var ids []uint64
	for id := range flushed {
		ids = append(ids, id)
	}
	if !reflect.DeepEqual(ids, []uint64{probe.FlushID}) {
		t.Errorf("flushed %v, want the probe's flush %d once", ids, probe.FlushID)
	}
}
