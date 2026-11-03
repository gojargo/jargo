package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// A continuously generating service reports its time to first byte when the
// first output arrives, once, and its processing time when the response ends.
// (Not a port of an upstream test: upstream's processor metrics do this for
// every service.)
func TestContinuousServicesReportTheirTiming(t *testing.T) {
	svc := llm.New("ContinuousLLM", &fakeGen{}, llm.WithContinuousGeneration())
	var mu sync.Mutex
	var ttfb, processing int
	seen := map[frames.Frame]bool{}
	started := make(chan struct{})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{EnableMetrics: true},
	})
	events.On(&task.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) { close(started) })
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		m, ok := f.(*frames.MetricsFrame)
		if !ok {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if seen[m] {
			return
		}
		seen[m] = true
		// The worker reports every metric at zero when it starts; only what
		// was measured counts.
		for _, d := range m.Data {
			switch v := d.(type) {
			case frames.TTFBMetricsData:
				if v.Value > 0 {
					ttfb++
				}
			case frames.ProcessingMetricsData:
				if v.Value > 0 {
					processing++
				}
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	<-started

	ctx := context.Background()
	svc.StartTTFBMetrics()
	svc.StartProcessingMetrics()
	svc.ReportTTFBMetrics(ctx)
	svc.ReportTTFBMetrics(ctx) // Only the first output reports.
	svc.StopAllMetrics(ctx)
	svc.StopProcessingMetrics(ctx) // Nothing left to report.

	time.Sleep(200 * time.Millisecond)
	task.StopWhenDone()
	<-runDone
	mu.Lock()
	defer mu.Unlock()
	if ttfb != 1 || processing != 1 {
		t.Errorf("reported %d TTFB and %d processing measurements, want one of each", ttfb, processing)
	}
}
