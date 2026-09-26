package pipeline_test

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// A frame is pushed again by every processor that passes it along.

// pushRecorder records every push of a text frame it is told about.
type pushRecorder struct {
	everyPush bool

	mu     sync.Mutex
	pushes []textPush
}

type textPush struct {
	source    string
	firstPush bool
}

func (o *pushRecorder) ObserveEveryPush() bool { return o.everyPush }

func (o *pushRecorder) OnPushFrame(data processor.FramePushed) {
	if _, ok := data.Frame.(*frames.TextFrame); !ok {
		return
	}
	o.mu.Lock()
	o.pushes = append(o.pushes, textPush{source: data.Source.Name(), firstPush: data.FirstPush})
	o.mu.Unlock()
}

func (o *pushRecorder) recorded() []textPush {
	o.mu.Lock()
	defer o.mu.Unlock()
	return slices.Clone(o.pushes)
}

// runThroughThree sends one text frame through three processors in a row.
func runThroughThree(t *testing.T, observers ...pipeline.Observer) []string {
	t.Helper()
	names := []string{"first", "second", "third"}
	procs := make([]processor.Processor, 0, len(names))
	for _, n := range names {
		e := newEcho()
		e.Base = processor.New(n, e)
		procs = append(procs, e)
	}
	off := false
	w := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		IdleTimeout: -1,
		EnableRTVI:  &off,
		Observers:   observers,
	})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	w.QueueFrame(frames.NewTextFrame("hello"))
	time.Sleep(100 * time.Millisecond)
	w.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	return names
}

func TestAnObserverThatHandlesAFrameOnceIsToldOnce(t *testing.T) {
	o := &pushRecorder{everyPush: false}
	runThroughThree(t, o)
	got := o.recorded()
	if len(got) != 1 || !got[0].firstPush {
		t.Fatalf("got %+v, want the first push alone", got)
	}
}

func TestAnObserverGetsEveryHopByDefault(t *testing.T) {
	o := &pushRecorder{everyPush: true}
	names := runThroughThree(t, o)
	got := o.recorded()
	sources := make([]string, 0, len(got))
	for _, p := range got {
		sources = append(sources, p.source)
	}
	for _, n := range names {
		if !slices.ContainsFunc(sources, func(s string) bool { return strings.HasPrefix(s, n+"#") }) {
			t.Errorf("no push from %q in %v", n, sources)
		}
	}
	// Only the first push is marked as such.
	for i, p := range got {
		if p.firstPush != (i == 0) {
			t.Fatalf("push %d first=%v, want only the first marked: %+v", i, p.firstPush, got)
		}
	}
}
