package providertest

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// TTFBReported drives one generation through svc and reports whether the
// service recorded a time to first byte for it.
//
// It is how a provider pins where its TTFB ends. TTFB is comparable across
// services only when they all stop it on the same thing, the first output the
// model produces, rather than on an earlier event that merely acknowledges the
// request. Feed the service a stream that stops after such an event and this
// returns false; feed it one that goes on to carry output and it returns true.
//
// The service is run in a pipeline with metrics enabled, since a metrics frame
// is what carries the measurement out of the service. The TTFB goes out on its
// own as the first output arrives, so every metrics frame is read up to the
// one with the processing time, which closes the generation.
func TTFBReported(t *testing.T, svc processor.Processor) bool {
	t.Helper()

	// The zeroed frames the worker sends when the pipeline is ready would arrive
	// first and are not what this is measuring.
	noInitial := false
	var sawTTFB atomic.Bool
	timing := make(chan struct{}, 1)
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mf, ok := f.(*frames.MetricsFrame)
		if !ok {
			return
		}
		for _, d := range mf.Data {
			switch d.(type) {
			case frames.TTFBMetricsData:
				sawTTFB.Store(true)
			case frames.ProcessingMetricsData:
				select {
				case timing <- struct{}{}:
				default:
				}
			}
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()
	t.Cleanup(func() {
		worker.StopWhenDone()
		<-runDone
	})

	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	worker.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-timing:
		return sawTTFB.Load()
	case <-time.After(3 * time.Second):
		t.Fatal("no timing MetricsFrame emitted")
		return false
	}
}
