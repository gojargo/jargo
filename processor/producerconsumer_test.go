package processor_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// collectDownstream runs procs as a pipeline over feed and returns every frame
// that reached the end of it.
func collectDownstream(
	t *testing.T, procs []processor.Processor, feed func(task *pipeline.Task),
) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewTask(pipeline.New(procs...), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	feed(task)
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), got...)
}

// countType returns how many of the frames are of type T.
func countType[T frames.Frame](got []frames.Frame) int {
	n := 0
	for _, f := range got {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

func isText(f frames.Frame) bool {
	_, ok := f.(*frames.TextFrame)
	return ok
}

// A consumer puts back what the producer picked, and the picked frame carries on
// down the pipeline as well, so the text arrives twice.
//
// The two are counted rather than ordered: the consumer runs on its own
// goroutine, so which of them reaches the end first is not fixed.
func TestProducerPassesThroughAndConsumes(t *testing.T) {
	prod := processor.NewProducerProcessor("Producer", isText)
	cons := processor.NewConsumerProcessor("Consumer", prod)

	got := collectDownstream(t, []processor.Processor{prod, cons}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("Hello!"))
		time.Sleep(100 * time.Millisecond)
	})

	if n := countType[*frames.TextFrame](got); n != 2 {
		t.Fatalf("got %d text frames, want the consumed one and the one passing through", n)
	}
}

// Without passthrough the picked frame reaches the consumer only, so the text
// arrives once.
func TestProducerWithoutPassthroughConsumesOnly(t *testing.T) {
	prod := processor.NewProducerProcessor("Producer", isText, processor.WithoutPassthrough())
	cons := processor.NewConsumerProcessor("Consumer", prod)

	got := collectDownstream(t, []processor.Processor{prod, cons}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("Hello!"))
		time.Sleep(100 * time.Millisecond)
	})

	if n := countType[*frames.TextFrame](got); n != 1 {
		t.Fatalf("got %d text frames, want only the consumed one", n)
	}
}

// Every registered consumer gets the frame, each with its own copy.
func TestProducerFeedsEveryConsumer(t *testing.T) {
	prod := processor.NewProducerProcessor("Producer", isText, processor.WithoutPassthrough())
	one := processor.NewConsumerProcessor("ConsumerOne", prod)
	two := processor.NewConsumerProcessor("ConsumerTwo", prod)

	got := collectDownstream(t, []processor.Processor{prod, one, two}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("Hello!"))
		time.Sleep(150 * time.Millisecond)
	})

	if n := countType[*frames.TextFrame](got); n != 2 {
		t.Fatalf("got %d text frames, want one from each consumer", n)
	}
}

// A frame the predicate did not pick is left alone: it passes through and no
// consumer sees it.
func TestProducerLeavesUnpickedFramesAlone(t *testing.T) {
	prod := processor.NewProducerProcessor("Producer", isText, processor.WithoutPassthrough())
	cons := processor.NewConsumerProcessor("Consumer", prod)

	got := collectDownstream(t, []processor.Processor{prod, cons}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		time.Sleep(100 * time.Millisecond)
	})

	if n := countType[*frames.LLMFullResponseStartFrame](got); n != 1 {
		t.Fatalf("got %d unpicked frames, want the one that passed through", n)
	}
}

// The producer's transformer rewrites what the consumers are handed, and leaves
// the frame carrying on down the pipeline as it was.
func TestProducerTransformsWhatItHandsOver(t *testing.T) {
	toAudio := func(frames.Frame) frames.Frame {
		return frames.NewInputAudioRawFrame(nil, 16000, 1)
	}
	prod := processor.NewProducerProcessor("Producer", isText,
		processor.WithProducerTransformer(toAudio))
	cons := processor.NewConsumerProcessor("Consumer", prod)

	got := collectDownstream(t, []processor.Processor{prod, cons}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("Hello!"))
		time.Sleep(100 * time.Millisecond)
	})

	if n := countType[*frames.InputAudioRawFrame](got); n != 1 {
		t.Fatalf("got %d audio frames, want the rewritten one", n)
	}
	if n := countType[*frames.TextFrame](got); n != 1 {
		t.Fatalf("got %d text frames, want the original still passing through", n)
	}
}

// The consumer's own transformer rewrites the frame on its way back into the
// pipeline.
func TestConsumerTransformsWhatItPutsBack(t *testing.T) {
	toAudio := func(frames.Frame) frames.Frame {
		return frames.NewInputAudioRawFrame(nil, 16000, 1)
	}
	prod := processor.NewProducerProcessor("Producer", isText)
	cons := processor.NewConsumerProcessor("Consumer", prod,
		processor.WithConsumerTransformer(toAudio))

	got := collectDownstream(t, []processor.Processor{prod, cons}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("Hello!"))
		time.Sleep(100 * time.Millisecond)
	})

	if n := countType[*frames.InputAudioRawFrame](got); n != 1 {
		t.Fatalf("got %d audio frames, want the rewritten one", n)
	}
	if n := countType[*frames.TextFrame](got); n != 1 {
		t.Fatalf("got %d text frames, want the original passing through", n)
	}
}

// The callback runs once the pipeline has been quiet for the timeout.
func TestIdleProcessorCallsBackWhenNothingComes(t *testing.T) {
	called := make(chan struct{}, 4)
	idle := processor.NewIdleFrameProcessor("Idle", 50*time.Millisecond,
		func(*processor.IdleFrameProcessor) {
			select {
			case called <- struct{}{}:
			default:
			}
		})

	collectDownstream(t, []processor.Processor{idle}, func(*pipeline.Task) {
		time.Sleep(200 * time.Millisecond)
	})

	select {
	case <-called:
	default:
		t.Fatal("the callback never ran for a pipeline that went quiet")
	}
}

// A frame coming through restarts the clock, so a pipeline kept busy never goes
// idle.
func TestIdleProcessorIsRestartedByAFrame(t *testing.T) {
	called := make(chan struct{}, 4)
	idle := processor.NewIdleFrameProcessor("Idle", 150*time.Millisecond,
		func(*processor.IdleFrameProcessor) {
			select {
			case called <- struct{}{}:
			default:
			}
		})

	collectDownstream(t, []processor.Processor{idle}, func(task *pipeline.Task) {
		for range 4 {
			time.Sleep(50 * time.Millisecond)
			task.QueueFrame(frames.NewTextFrame("hello"))
		}
	})

	select {
	case <-called:
		t.Fatal("the callback ran for a pipeline that was never quiet")
	default:
	}
}

// Naming the frame types to watch for measures the absence of those, so a frame
// of another type does not restart the clock.
func TestIdleProcessorWatchesOnlyTheTypesItWasGiven(t *testing.T) {
	called := make(chan struct{}, 4)
	idle := processor.NewIdleFrameProcessor("Idle", 100*time.Millisecond,
		func(*processor.IdleFrameProcessor) {
			select {
			case called <- struct{}{}:
			default:
			}
		},
		processor.FrameIs[*frames.TranscriptionFrame]())

	collectDownstream(t, []processor.Processor{idle}, func(task *pipeline.Task) {
		time.Sleep(60 * time.Millisecond)
		task.QueueFrame(frames.NewTextFrame("this should not restart it"))
		time.Sleep(150 * time.Millisecond)
	})

	select {
	case <-called:
	default:
		t.Fatal("the callback never ran, so an unwatched frame restarted the clock")
	}
}

// The transformer rewrites the text of every text frame and leaves everything
// else alone.
func TestTextTransformerRewritesText(t *testing.T) {
	up := processor.NewStatelessTextTransformer("Upper", func(s string) string { return s + "!" })

	got := collectDownstream(t, []processor.Processor{up}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("hello"))
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		time.Sleep(50 * time.Millisecond)
	})

	var text *frames.TextFrame
	for _, f := range got {
		if tf, ok := f.(*frames.TextFrame); ok {
			text = tf
		}
	}
	if text == nil {
		t.Fatalf("no text frame reached the end, got %+v", got)
	}
	if text.Text != "hello!" {
		t.Errorf("text = %q, want it rewritten", text.Text)
	}
	if n := countType[*frames.LLMFullResponseStartFrame](got); n != 1 {
		t.Errorf("got %d other frames, want them left alone", n)
	}
}

// A consumer can put what it consumed back the other way, for a frame that has
// to reach what sits ahead of it rather than behind.
func TestConsumerCanPushUpstream(t *testing.T) {
	prod := processor.NewProducerProcessor("Producer", isText, processor.WithoutPassthrough())
	cons := processor.NewConsumerProcessor("Consumer", prod,
		processor.WithConsumerDirection(processor.Upstream))

	got := collectDownstream(t, []processor.Processor{prod, cons}, func(task *pipeline.Task) {
		task.QueueFrame(frames.NewTextFrame("Hello!"))
		time.Sleep(100 * time.Millisecond)
	})

	// It went the other way, so it never reached the end of the pipeline.
	if n := countType[*frames.TextFrame](got); n != 0 {
		t.Fatalf("got %d text frames downstream, want the consumed one sent upstream", n)
	}
}
