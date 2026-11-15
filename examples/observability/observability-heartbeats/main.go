// Command observability-heartbeats shows heartbeat monitoring.
//
// With heartbeats on, the worker sends a heartbeat through the pipeline at a
// fixed interval and watches for it at the far end. This pipeline is a single
// processor that swallows everything but system frames, heartbeats included,
// to simulate a stalled pipeline: a warning is logged when no heartbeat
// arrives within the timeout, and the pipeline.EventHeartbeatTimeout handler
// runs.
//
//	go run ./examples/observability/observability-heartbeats
//
// Press Ctrl-C to stop it.
package main

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// nullProcessor passes on the system frames only.
type nullProcessor struct {
	*processor.Base
}

func newNullProcessor() *nullProcessor {
	p := &nullProcessor{}
	p.Base = processor.New("NullProcessor", p)
	return p
}

// ProcessFrame passes on the system frames (the start and cancel frames, for
// instance) only, so the heartbeat frames are swallowed, simulating a stalled
// pipeline.
func (p *nullProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(frames.SystemFrame); ok {
		return p.PushFrame(ctx, f, dir)
	}
	return nil
}

func main() {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	ctx := context.Background()

	p := pipeline.New(newNullProcessor())

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params:                  pipeline.Params{EnableHeartbeats: true},
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	r := workers.NewRunner(workers.RunnerConfig{})
	r.AddWorkers(ctx, worker)

	events.OnSignal(&worker.Registry, pipeline.EventHeartbeatTimeout, func(context.Context) {
		slog.Warn("Heartbeat timeout detected: the pipeline may be stalled")
	})

	if err := r.Run(ctx, workers.RunOptions{}); err != nil {
		slog.Error("runner ended", "err", err)
	}
}
