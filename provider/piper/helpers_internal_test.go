package piper

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// speakFrame is a TTSSpeakFrame for text, appended to the context or not.
func speakFrame(text string, appendToContext bool) *frames.TTSSpeakFrame {
	f := frames.NewTTSSpeakFrame(text)
	f.AppendToContext = appendToContext
	return f
}

// runTest runs the service in a pipeline, queues the frames, ends the
// pipeline, and returns the frames that left it downstream and upstream.
func runTest(t *testing.T, svc *tts.Base, send ...frames.Frame) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
		// No RTVI, so only what the service pushes reaches the ends.
		EnableRTVI: new(false),
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, f := range send {
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), slices.Clone(up)
}

// frameNames is the name of each frame, in order.
func frameNames(fs []frames.Frame) []string {
	names := make([]string, len(fs))
	for i, f := range fs {
		names[i] = f.Name()
	}
	return names
}

// indexOf is the index of the first frame of type T, or -1.
func indexOf[T frames.Frame](fs []frames.Frame) int {
	return slices.IndexFunc(fs, func(f frames.Frame) bool { _, ok := f.(T); return ok })
}

// audioFrames is the audio frames among fs.
func audioFrames(fs []frames.Frame) []*frames.TTSAudioRawFrame {
	var out []*frames.TTSAudioRawFrame
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a)
		}
	}
	return out
}

// errorsOf is the messages of the error frames among fs.
func errorsOf(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, ef.Error)
		}
	}
	return out
}
