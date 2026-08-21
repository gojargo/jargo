package providertest

import (
	"context"
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
// The service is run in a pipeline with metrics enabled, since the timing frame
// is what carries the measurement out of the service.
func TTFBReported(t *testing.T, svc processor.Processor) bool {
	t.Helper()

	// The zeroed frames the worker sends when the pipeline is ready would arrive
	// first and are not what this is measuring.
	noInitial := false
	timing := make(chan *frames.MetricsFrame, 4)
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
		// The timing frame is the one carrying the processing time; it carries
		// the TTFB alongside it when the service recorded one.
		for _, d := range mf.Data {
			if _, ok := d.(frames.ProcessingMetricsData); ok {
				select {
				case timing <- mf:
				default:
				}
				return
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
	case mf := <-timing:
		for _, d := range mf.Data {
			if _, ok := d.(frames.TTFBMetricsData); ok {
				return true
			}
		}
		return false
	case <-time.After(3 * time.Second):
		t.Fatal("no timing MetricsFrame emitted")
		return false
	}
}
