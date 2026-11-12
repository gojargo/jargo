package cartesia_test

// A pipeline harness for the turns STT tests: it streams audio in real time
// through the service, against a local fake server, and keeps every frame that
// reaches either end of the pipeline.

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

const frameDuration = 20 * time.Millisecond

func wsURL(httpURL string) string { return "ws" + strings.TrimPrefix(httpURL, "http") }

// result is what a run captured.
type result struct {
	down    []frames.Frame
	up      []frames.Frame
	elapsed time.Duration
}

// injection queues f after the given number of audio frames.
type injection struct {
	after int
	f     frames.Frame
}

// run streams real-time 20 ms frames of silence through svc for d, queueing each
// injection at its frame, and returns every frame that reached either end of the
// pipeline.
func run(t *testing.T, svc *stt.StreamService, d time.Duration, inject ...injection) result {
	t.Helper()
	var mu sync.Mutex
	var res result
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		IdleTimeout:             -1,
		Params:                  pipeline.Params{EnableUsageMetrics: true},
		ReachedUpstreamFilter:   pipeline.AnyFrame,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		res.up = append(res.up, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		res.down = append(res.down, f)
		mu.Unlock()
	})
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for i := range int(d / frameDuration) {
		for _, in := range inject {
			if in.after == i {
				task.QueueFrame(in.f)
			}
		}
		task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
		time.Sleep(frameDuration)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("pipeline did not shut down")
	}
	res.elapsed = time.Since(start)
	// Give a connection still being served time to record what it received.
	time.Sleep(100 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	return res
}

type transcript struct {
	text      string
	finalized bool
}

// turnEnds returns the transcriptions pushed, and the positions of the proposed
// turn stops.
func turnEnds(down []frames.Frame) ([]transcript, []int) {
	var ts []transcript
	var stops []int
	for i, f := range down {
		switch fr := f.(type) {
		case *frames.TranscriptionFrame:
			ts = append(ts, transcript{fr.Text, fr.Finalized})
		case *frames.ProposedUserStoppedSpeakingFrame:
			stops = append(stops, i)
		}
	}
	return ts, stops
}

func errorFrames(fs []frames.Frame) []*frames.ErrorFrame {
	var out []*frames.ErrorFrame
	for _, f := range fs {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, ef)
		}
	}
	return out
}
