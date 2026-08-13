package pipeline_test

import (
	"testing"

	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
)

// A client talks to a session over RTVI, so a worker adds the processor and its
// observer unless the pipeline already carries one or the caller turns it off.

func TestWorkerAddsRTVIByDefault(t *testing.T) {
	t.Parallel()
	w := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{IdleTimeout: -1})
	if w.RTVI() == nil {
		t.Error("the worker added no RTVI processor, want one by default")
	}
}

func TestWorkerRTVICanBeTurnedOff(t *testing.T) {
	t.Parallel()
	off := false
	w := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		IdleTimeout: -1,
		EnableRTVI:  &off,
	})
	if w.RTVI() != nil {
		t.Error("the worker added an RTVI processor, want none when it is turned off")
	}
}

func TestWorkerKeepsTheRTVIThePipelineCarries(t *testing.T) {
	t.Parallel()
	own := rtvi.NewProcessor()
	w := pipeline.NewWorker(pipeline.New(own, newEcho()), pipeline.WorkerConfig{
		IdleTimeout: -1,
		Observers:   []pipeline.Observer{rtvi.NewObserver(own)},
	})
	// Adding a second would put one in the frame path twice.
	if w.RTVI() != own {
		t.Error("the worker replaced the RTVI processor the pipeline already carried")
	}
}

func TestWorkerUsesTheRTVIProcessorItIsGiven(t *testing.T) {
	t.Parallel()
	own := rtvi.NewProcessor()
	w := pipeline.NewWorker(pipeline.New(newEcho()), pipeline.WorkerConfig{
		IdleTimeout:   -1,
		RTVIProcessor: own,
	})
	if w.RTVI() != own {
		t.Error("the worker built its own RTVI processor, want the one it was given")
	}
}

func TestWorkerFindsTheRTVIInsideANestedPipeline(t *testing.T) {
	t.Parallel()
	own := rtvi.NewProcessor()
	inner := pipeline.New(own, newEcho())
	w := pipeline.NewWorker(pipeline.New(inner), pipeline.WorkerConfig{
		IdleTimeout: -1,
		Observers:   []pipeline.Observer{rtvi.NewObserver(own)},
	})
	if w.RTVI() != own {
		t.Error("the worker did not find the RTVI processor nested in the pipeline")
	}
}
