package dtmf_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/dtmf"
)

func TestToneLengthAndSilenceForUnknown(t *testing.T) {
	pcm := dtmf.Tone(frames.KeypadFive, 100, 8000)
	if len(pcm) != 8000*100/1000*2 {
		t.Fatalf("tone length = %d bytes, want %d", len(pcm), 8000*100/1000*2)
	}
	nonzero := false
	for _, b := range pcm {
		if b != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("tone is all silence")
	}
	if dtmf.Tone("nope", 100, 8000) != nil {
		t.Fatal("unknown button should yield nil")
	}
}

func TestAggregatorFlushesOnTerminator(t *testing.T) {
	got := make(chan string, 1)
	agg := dtmf.NewAggregator(dtmf.AggregatorConfig{Prefix: "digits: "})
	task := pipeline.NewTask(pipeline.New(agg), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if tf, ok := f.(*frames.TranscriptionFrame); ok {
				select {
				case got <- tf.Text:
				default:
				}
			}
		},
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
