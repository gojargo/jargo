package dtmf_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/dtmf"
	"github.com/gojargo/jargo/utils/events"
)

func TestAggregatorFlushesOnTerminator(t *testing.T) {
	got := make(chan string, 1)
	agg := dtmf.NewAggregator(dtmf.AggregatorConfig{Prefix: "digits: "})
	task := pipeline.NewWorker(pipeline.New(agg), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			select {
			case got <- tf.Text:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for _, b := range []frames.KeypadEntry{frames.KeypadFour, frames.KeypadTwo, frames.KeypadPound} {
		task.QueueFrame(frames.NewInputDTMFFrame(b))
	}

	select {
	case text := <-got:
		if text != "digits: 42" {
			t.Fatalf("aggregated text = %q, want %q", text, "digits: 42")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("aggregator never flushed")
	}
	task.StopWhenDone()
	<-runDone
}
