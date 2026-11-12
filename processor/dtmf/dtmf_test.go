package dtmf_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/dtmf"
	"github.com/gojargo/jargo/utils/events"
)

func TestAggregatorFlushesOnTerminator(t *testing.T) {
	got := make(chan string, 1)
	agg := dtmf.NewAggregator(dtmf.AggregatorConfig{Prefix: new("digits: ")})
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
		if text != "digits: 42#" {
			t.Fatalf("aggregated text = %q, want %q", text, "digits: 42#")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("aggregator never flushed")
	}
	task.StopWhenDone()
	<-runDone
}

// dtmfRun is what reached the end of a pipeline running one aggregator.
type dtmfRun struct {
	mu    sync.Mutex
	names []string
	texts []string
}

// runKeys runs an aggregator built from cfg, feeding it each step: a key, or a
// pause. With end set, the pipeline is ended with an EndFrame once the steps are
// fed; otherwise it is stopped as the test ends.
func runKeys(t *testing.T, cfg dtmf.AggregatorConfig, end bool, steps ...any) *dtmfRun {
	t.Helper()
	r := &dtmfRun{}
	task := pipeline.NewWorker(pipeline.New(dtmf.NewAggregator(cfg)), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		defer r.mu.Unlock()
		switch fr := f.(type) {
		case *frames.InputDTMFFrame:
			r.names = append(r.names, "key")
		case *frames.InterruptionFrame:
			r.names = append(r.names, "interruption")
		case *frames.TranscriptionFrame:
			r.names = append(r.names, "text")
			r.texts = append(r.texts, fr.Text)
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for _, s := range steps {
		switch v := s.(type) {
		case frames.KeypadEntry:
			task.QueueFrame(frames.NewInputDTMFFrame(v))
		case time.Duration:
			time.Sleep(v)
		}
	}
	time.Sleep(50 * time.Millisecond)
	if end {
		task.StopWhenDone()
	} else {
		task.Cancel(context.Background(), "test over")
	}
	<-runDone
	return r
}

func (r *dtmfRun) gotTexts() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.texts...)
}

// TestAggregatorDefaults checks the zero config is the defaults: an entry with
// no terminator is flushed after two idle seconds, and the transcription says it
// came from the keypad.
func TestAggregatorDefaults(t *testing.T) {
	r := runKeys(t, dtmf.AggregatorConfig{}, false, frames.KeypadOne, 2300*time.Millisecond)
	if got := r.gotTexts(); len(got) != 1 || got[0] != "DTMF: 1" {
		t.Errorf("transcriptions = %q, want the entry flushed by the default timeout", got)
	}
}

// TestAggregatorKeepsTheTerminator checks the terminator is part of the entry,
// as the caller pressed it, so a lone terminator is an entry too.
func TestAggregatorKeepsTheTerminator(t *testing.T) {
	r := runKeys(t, dtmf.AggregatorConfig{Terminator: frames.KeypadStar}, true,
		frames.KeypadOne, frames.KeypadTwo, frames.KeypadStar, 50*time.Millisecond, frames.KeypadStar)
	if got := r.gotTexts(); len(got) != 2 || got[0] != "DTMF: 12*" || got[1] != "DTMF: *" {
		t.Errorf("transcriptions = %q, want the terminator kept in each entry", got)
	}
}
