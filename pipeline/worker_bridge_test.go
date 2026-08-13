package pipeline_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/registry"
	"github.com/gojargo/jargo/utils/events"
)

// A bridged worker puts its pipeline on the bus: what comes out of either end
// is copied across for the other workers, and what they send arrives back in.
//
// Ported from the upstream edge-to-bus suite.

// generator answers every text frame with one of its own, so a test can tell a
// frame the pipeline produced from the frame that went in.
type generator struct{ *processor.Base }

func newGenerator() *generator {
	p := &generator{}
	p.Base = processor.New("Generator", p)
	return p
}

func (p *generator) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok {
		return p.PushFrame(ctx, frames.NewTextFrame("generated_"+tf.Text), dir)
	}
	return p.PushFrame(ctx, f, dir)
}

// busRecorder keeps the frame messages that cross the bus.
type busRecorder struct {
	mu  sync.Mutex
	got []*bus.FrameMessage
}

func (r *busRecorder) Name() string { return "bus-recorder" }

func (r *busRecorder) OnBusMessage(_ context.Context, m bus.Message) {
	fm, ok := m.(*bus.FrameMessage)
	if !ok {
		return
	}
	r.mu.Lock()
	r.got = append(r.got, fm)
	r.mu.Unlock()
}

// texts are the text frames that crossed, as source and text pairs.
func (r *busRecorder) texts() [][2]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out [][2]string
	for _, m := range r.got {
		if tf, ok := m.Frame.(*frames.TextFrame); ok {
			out = append(out, [2]string{m.Source(), tf.Text})
		}
	}
	return out
}

// newBridgedWorker builds a worker on pipe, bridged as given, with a recorder
// watching the bus it is attached to.
func newBridgedWorker(t *testing.T, pipe processor.Processor, bridged []string) (
	*pipeline.Worker, *bus.AsyncQueueBus, *busRecorder,
) {
	t.Helper()
	w := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
		Name:                    "worker",
		Bridged:                 bridged,
		IdleTimeout:             -1,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	msgBus := bus.NewAsyncQueueBus()
	rec := &busRecorder{}
	msgBus.Subscribe(rec)
	w.Attach(t.Context(), registry.New(runner), msgBus.Bus)
	msgBus.Start(t.Context())
	t.Cleanup(msgBus.Stop)
	return w, msgBus, rec
}

// runBridged runs the worker and waits for its pipeline to be up. The edges
// subscribe to the bus as the pipeline is set up, so a frame sent before that
// reaches nobody.
func runBridged(t *testing.T, w *pipeline.Worker) chan struct{} {
	t.Helper()
	started := make(chan struct{})
	var once sync.Once
	events.On(w.Events(), pipeline.EventPipelineStarted,
		func(_ context.Context, _ *frames.StartFrame) { once.Do(func() { close(started) }) })
	done := runWorker(t, w)
	awaitClosed(t, started, "the pipeline starting")
	return done
}

// sendFromOther puts a frame on the bus as another worker would.
func sendFromOther(t *testing.T, msgBus *bus.AsyncQueueBus, text, bridge string) {
	t.Helper()
	m := &bus.FrameMessage{
		Frame:     frames.NewTextFrame(text),
		Direction: processor.Downstream,
		Bridge:    bridge,
	}
	m.From = "other"
	msgBus.Send(t.Context(), m)
}

// collectReached records the text frames reaching the end of the pipeline.
func collectReached(w *pipeline.Worker) (*sync.Mutex, *[]string) {
	var mu sync.Mutex
	var got []string
	events.On(w.Events(), pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TextFrame); ok {
			mu.Lock()
			got = append(got, tf.Text)
			mu.Unlock()
		}
	})
	return &mu, &got
}

func TestBridgedWorkerTeesGeneratedFramesOntoTheBus(t *testing.T) {
	t.Parallel()
	w, _, rec := newBridgedWorker(t, pipeline.New(newGenerator()), []string{})

	done := runBridged(t, w)
	w.QueueFrame(frames.NewTextFrame("input"))

	eventuallyTrue(t, "the generated frame crosses the bus", func() bool {
		for _, pair := range rec.texts() {
			if pair[0] == "worker" && pair[1] == "generated_input" {
				return true
			}
		}
		return false
	})

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerDoesNotTakeBackWhatItSent(t *testing.T) {
	t.Parallel()
	w, msgBus, rec := newBridgedWorker(t, pipeline.New(newEcho()), []string{})

	done := runBridged(t, w)
	sendFromOther(t, msgBus, "from_bus", "")

	// The frame passes through the pipeline and is teed back onto the bus by
	// the far edge. That is expected; what must not happen is a loop, because
	// the near edge ignores what this worker itself sent.
	eventuallyTrue(t, "the frame crosses both ways", func() bool {
		var fromOther, fromWorker int
		for _, pair := range rec.texts() {
			switch pair[0] {
			case "other":
				fromOther++
			case "worker":
				fromWorker++
			}
		}
		return fromOther == 1 && fromWorker == 1
	})

	time.Sleep(200 * time.Millisecond)
	if got := len(rec.texts()); got != 2 {
		t.Errorf("%d text frames crossed the bus, want exactly 2: the frame is looping", got)
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestUnbridgedWorkerPutsNothingOnTheBus(t *testing.T) {
	t.Parallel()
	w, _, rec := newBridgedWorker(t, pipeline.New(newEcho()), nil)

	done := runBridged(t, w)
	w.QueueFrame(frames.NewTextFrame("root_frame"))
	time.Sleep(200 * time.Millisecond)

	if got := rec.texts(); len(got) != 0 {
		t.Errorf("%v crossed the bus, want nothing from an unbridged worker", got)
	}
	if w.Bridged() {
		t.Error("an unbridged worker reports itself bridged")
	}

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerTakesFramesFromTheBus(t *testing.T) {
	t.Parallel()
	w, msgBus, _ := newBridgedWorker(t, pipeline.New(newEcho()), []string{})
	mu, got := collectReached(w)

	done := runBridged(t, w)
	sendFromOther(t, msgBus, "from_bus", "")

	eventuallyTrue(t, "the frame from the bus reaches the pipeline", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(*got) == 1 && (*got)[0] == "from_bus"
	})

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerKeepsTheDirection(t *testing.T) {
	t.Parallel()
	w, _, rec := newBridgedWorker(t, pipeline.New(newGenerator()), []string{})

	done := runBridged(t, w)
	w.QueueFrame(frames.NewTextFrame("hello"))

	eventuallyTrue(t, "the generated frame crosses the bus", func() bool {
		rec.mu.Lock()
		defer rec.mu.Unlock()
		for _, m := range rec.got {
			tf, ok := m.Frame.(*frames.TextFrame)
			if ok && tf.Text == "generated_hello" {
				return m.Direction == processor.Downstream
			}
		}
		return false
	})

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerTakesItsOwnBridge(t *testing.T) {
	t.Parallel()
	w, msgBus, _ := newBridgedWorker(t, pipeline.New(newEcho()), []string{"voice"})
	mu, got := collectReached(w)

	done := runBridged(t, w)
	sendFromOther(t, msgBus, "voice_frame", "voice")

	eventuallyTrue(t, "the frame from the named bridge arrives", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(*got) == 1 && (*got)[0] == "voice_frame"
	})

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerRefusesAnotherBridge(t *testing.T) {
	t.Parallel()
	w, msgBus, _ := newBridgedWorker(t, pipeline.New(newEcho()), []string{"voice"})
	mu, got := collectReached(w)

	done := runBridged(t, w)
	sendFromOther(t, msgBus, "video_frame", "video")
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	if len(*got) != 0 {
		t.Errorf("took %v, want nothing from a bridge it does not listen on", *got)
	}
	mu.Unlock()

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerWithNoBridgeNamedTakesThemAll(t *testing.T) {
	t.Parallel()
	w, msgBus, _ := newBridgedWorker(t, pipeline.New(newEcho()), []string{})
	mu, got := collectReached(w)

	done := runBridged(t, w)
	sendFromOther(t, msgBus, "voice", "voice")
	sendFromOther(t, msgBus, "video", "video")
	sendFromOther(t, msgBus, "none", "")

	eventuallyTrue(t, "all three arrive", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(*got) == 3
	})

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerTakesEveryBridgeItNames(t *testing.T) {
	t.Parallel()
	w, msgBus, _ := newBridgedWorker(t, pipeline.New(newEcho()), []string{"voice", "video"})
	mu, got := collectReached(w)

	done := runBridged(t, w)
	sendFromOther(t, msgBus, "voice", "voice")
	sendFromOther(t, msgBus, "video", "video")
	sendFromOther(t, msgBus, "other", "other")

	eventuallyTrue(t, "both named bridges arrive", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(*got) == 2
	})
	time.Sleep(200 * time.Millisecond)

	mu.Lock()
	seen := map[string]bool{}
	for _, text := range *got {
		seen[text] = true
	}
	if !seen["voice"] || !seen["video"] || seen["other"] {
		t.Errorf("took %v, want the two named bridges and no other", *got)
	}
	mu.Unlock()

	w.StopWhenDone()
	awaitClosed(t, done, "the run returning")
}

func TestBridgedWorkerAnnouncesItselfBridged(t *testing.T) {
	t.Parallel()
	w, _, _ := newBridgedWorker(t, pipeline.New(newEcho()), []string{})
	if !w.Bridged() {
		t.Error("a bridged worker does not report itself bridged")
	}
}
