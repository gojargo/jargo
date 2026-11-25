package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

// TestSpokenTextCarriesItsSegmentID checks the text a service without word
// timings reports as spoken names the segment announced for it before synthesis.
// A client renders the announcement and then the completion, and only the shared
// id tells it the two are the same sentence rather than two of them.
func TestSpokenTextCarriesItsSegmentID(t *testing.T) {
	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 1)}

	var (
		mu        sync.Mutex
		announced []*frames.AggregatedTextFrame
		spoken    []*frames.TTSTextFrame
	)
	stopped := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(tts.New("FakeTTS", syn)), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.AggregatedTextFrame:
			announced = append(announced, fr)
		case *frames.TTSTextFrame:
			spoken = append(spoken, fr)
		case *frames.TTSStoppedFrame:
			select {
			case stopped <- struct{}{}:
			default:
			}
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello world."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("TTS base did not finish a synthesis")
	}
	// The spoken text trails the stop frame through the context's queue.
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	<-done

	mu.Lock()
	defer mu.Unlock()
	if len(announced) != 1 || len(spoken) != 1 {
		t.Fatalf("got %d announcements and %d spoken texts, want one of each", len(announced), len(spoken))
	}
	if spoken[0].SegmentID != announced[0].ID() {
		t.Fatalf("spoken text names segment %d, want the announced one, %d",
			spoken[0].SegmentID, announced[0].ID())
	}
}
