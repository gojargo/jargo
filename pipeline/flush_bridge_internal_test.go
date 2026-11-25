package pipeline

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/workers"
)

// TestTheWorkerAnsweringAProbeLetsGoOfItOnceSettled covers a flush answered in
// another worker's pipeline. Each leg of the trip is a new frame, all of them
// under one flush id, and the worker holding the probe tracks it by that id: a
// holder that kept the probe after the flush settled would go on reporting
// progress on a flush that is over.
func TestTheWorkerAnsweringAProbeLetsGoOfItOnceSettled(t *testing.T) {
	noSignals := false
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &noSignals})

	// The origin's pipeline is on the bus and leaves the probe for the holder
	// to answer.
	origin := NewWorker(New(newPassthrough()), WorkerConfig{
		Name: "origin", Bridged: []string{}, IdleTimeout: -1,
	})
	holder := NewWorker(New(
		bus.NewBridgeProcessor(bus.BridgeConfig{Bus: r.Bus(), WorkerName: "holder"}),
		newPassthrough(),
	), WorkerConfig{Name: "holder", IdleTimeout: -1})

	var wg sync.WaitGroup
	for _, w := range []*Worker{origin, holder} {
		wg.Add(1)
		var once sync.Once
		event.On(&w.Registry, EventPipelineStarted, func(context.Context, *frames.StartFrame) {
			once.Do(wg.Done)
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	r.AddWorkers(ctx, origin, holder)
	done := make(chan error, 1)
	go func() { done <- r.Run(ctx, workers.RunOptions{}) }()
	wg.Wait()

	flushErr := origin.Flush(ctx)
	holder.flushMu.Lock()
	held := len(holder.foreignProbes)
	holder.flushMu.Unlock()

	origin.StopWhenDone()
	holder.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatalf("run: %v", err)
	}

	if flushErr != nil {
		t.Errorf("Flush = %v, want the flush settled by the holder", flushErr)
	}
	if held != 0 {
		t.Errorf("the holder still tracks %d probes after the flush settled, want none", held)
	}
}
