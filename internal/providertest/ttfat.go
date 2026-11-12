package providertest

import (
	"context"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// StagedStream serves a streamed body in two stages: first, then rest after
// gap. It is how a test puts time between the output a measurement should end
// at and what follows it.
func StagedStream(contentType, first, rest string, gap time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(first))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(gap)
		_, _ = w.Write([]byte(rest))
	}
}

// TTFATReported drives one generation, with a tool on offer, through svc and
// returns the time to first answer token it reported, and whether it reported
// one.
//
// It is how a provider pins where its TTFAT ends: at the first answer token,
// which on a turn that calls a tool instead of replying is the call's start
// rather than the end of the stream.
func TTFATReported(t *testing.T, svc processor.Processor) (frames.TTFATMetricsData, bool) {
	t.Helper()

	noInitial := false
	var mu sync.Mutex
	var got frames.TTFATMetricsData
	var found bool
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
			switch d := d.(type) {
			case frames.TTFATMetricsData:
				mu.Lock()
				got, found = d, true
				mu.Unlock()
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
	convo.AddUserMessage("What is the weather in Paris?")
	convo.SetTools([]frames.Tool{{Name: "get_weather", Description: "Get the weather."}})
	worker.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-timing:
	case <-time.After(5 * time.Second):
		t.Fatal("no timing MetricsFrame emitted")
	}
	mu.Lock()
	defer mu.Unlock()
	return got, found
}
