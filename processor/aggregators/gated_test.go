package aggregators_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/notify"
)

// collectGated runs procs as a pipeline over feed and returns every frame that
// reached the end of it. feed is responsible for ending the run.
func collectGated(
	t *testing.T, procs []processor.Processor, feed func(task *pipeline.Worker),
) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	// feed ends the run itself: a gate left closed holds the frame that ends it,
	// so those runs have to be canceled instead.
	feed(task)

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), got...)
}

// spoken returns the text of every text frame that came through, in order.
func spoken(got []frames.Frame) []string {
	var out []string
	for _, f := range got {
		if tf, ok := f.(*frames.TextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

func sameOrder(a, b []string) bool { return slices.Equal(a, b) }

// isOpener matches the text that opens the gate in these tests.
func isOpener(f frames.Frame) bool {
	tf, ok := f.(*frames.TextFrame)
	return ok && tf.Text == "open"
}

func isCloser(f frames.Frame) bool {
	tf, ok := f.(*frames.TextFrame)
	return ok && tf.Text == "close"
}

// A closed gate holds what arrives, and the frame that opens it is released
// ahead of everything it was holding.
func TestGatedReleasesTheOpenerFirst(t *testing.T) {
	g := aggregators.NewGated("Gate", aggregators.GatedConfig{
		Open:  isOpener,
		Close: isCloser,
	})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewTextFrame("one"))
		task.QueueFrame(frames.NewTextFrame("two"))
		task.QueueFrame(frames.NewTextFrame("open"))
		time.Sleep(50 * time.Millisecond)
		task.StopWhenDone()
	})

	want := []string{"open", "one", "two"}
	if !sameOrder(spoken(got), want) {
		t.Fatalf("frames = %q, want %q", spoken(got), want)
	}
}

// A gate that starts open lets frames through until one closes it.
func TestGatedStartsOpenAndCloses(t *testing.T) {
	g := aggregators.NewGated("Gate", aggregators.GatedConfig{
		Open:      isOpener,
		Close:     isCloser,
		StartOpen: true,
	})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewTextFrame("one"))
		task.QueueFrame(frames.NewTextFrame("close"))
		task.QueueFrame(frames.NewTextFrame("held"))
		time.Sleep(50 * time.Millisecond)
		// The gate is closed, so the frame that ends a run would be held with
		// the rest. Canceling is a system frame and gets through.
		task.Cancel(t.Context(), "")
	})

	// The frame that closes the gate is held with the rest rather than passing.
	want := []string{"one"}
	if !sameOrder(spoken(got), want) {
		t.Fatalf("frames = %q, want %q", spoken(got), want)
	}
}

// What a close held is released by the next open, the closing frame first,
// because it arrived first.
func TestGatedReleasesWhatACloseHeld(t *testing.T) {
	g := aggregators.NewGated("Gate", aggregators.GatedConfig{
		Open:      isOpener,
		Close:     isCloser,
		StartOpen: true,
	})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewTextFrame("close"))
		task.QueueFrame(frames.NewTextFrame("held"))
		task.QueueFrame(frames.NewTextFrame("open"))
		time.Sleep(50 * time.Millisecond)
		task.StopWhenDone()
	})

	want := []string{"open", "close", "held"}
	if !sameOrder(spoken(got), want) {
		t.Fatalf("frames = %q, want %q", spoken(got), want)
	}
}

// A system frame is never held: a gate that swallowed one would have no way of
// hearing about an interruption or a shutdown while it was closed.
func TestGatedNeverHoldsASystemFrame(t *testing.T) {
	g := aggregators.NewGated("Gate", aggregators.GatedConfig{
		Open:  isOpener,
		Close: isCloser,
	})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewUserStartedSpeakingFrame())
		time.Sleep(50 * time.Millisecond)
		// The gate never opened, so the run has to be canceled.
		task.Cancel(t.Context(), "")
	})

	found := false
	for _, f := range got {
		if _, ok := f.(*frames.UserStartedSpeakingFrame); ok {
			found = true
		}
	}
	if !found {
		t.Fatalf("frames = %+v, want the system frame through a closed gate", got)
	}
}

// The conversation is held until the notifier fires, then released.
func TestGatedContextWaitsForTheNotifier(t *testing.T) {
	n := notify.NewEventNotifier()
	g := aggregators.NewGatedContext("Gate", aggregators.GatedContextConfig{Notifier: n})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		time.Sleep(50 * time.Millisecond)
		// Nothing should have gone through yet.
		n.Notify()
		time.Sleep(50 * time.Millisecond)
		task.StopWhenDone()
	})

	if n := countContexts(got); n != 1 {
		t.Fatalf("got %d contexts, want the held one released", n)
	}
}

// Nothing goes through while the notifier stays quiet.
func TestGatedContextHoldsWhileTheNotifierIsQuiet(t *testing.T) {
	n := notify.NewEventNotifier()
	g := aggregators.NewGatedContext("Gate", aggregators.GatedContextConfig{Notifier: n})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		time.Sleep(100 * time.Millisecond)
		task.StopWhenDone()
	})

	if n := countContexts(got); n != 0 {
		t.Fatalf("got %d contexts, want them all held", n)
	}
}

// Only the latest conversation is held, because an earlier one describes a
// conversation that has since been added to.
func TestGatedContextKeepsOnlyTheLatest(t *testing.T) {
	n := notify.NewEventNotifier()
	g := aggregators.NewGatedContext("Gate", aggregators.GatedContextConfig{Notifier: n})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		time.Sleep(50 * time.Millisecond)
		n.Notify()
		time.Sleep(50 * time.Millisecond)
		task.StopWhenDone()
	})

	if n := countContexts(got); n != 1 {
		t.Fatalf("got %d contexts, want only the latest released", n)
	}
}

// Starting open lets the first conversation through without waiting, so the bot
// can open the call before anything has had to decide.
func TestGatedContextCanStartOpen(t *testing.T) {
	n := notify.NewEventNotifier()
	g := aggregators.NewGatedContext("Gate", aggregators.GatedContextConfig{
		Notifier:  n,
		StartOpen: true,
	})

	got := collectGated(t, []processor.Processor{g}, func(task *pipeline.Worker) {
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("")))
		time.Sleep(80 * time.Millisecond)
		task.StopWhenDone()
	})

	if n := countContexts(got); n != 1 {
		t.Fatalf("got %d contexts, want only the first through", n)
	}
}

func countContexts(got []frames.Frame) int {
	n := 0
	for _, f := range got {
		if _, ok := f.(*frames.LLMContextFrame); ok {
			n++
		}
	}
	return n
}
