package stt_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// ttfbWatcher collects the time-to-first-byte a service reports.
type ttfbWatcher struct {
	mu   sync.Mutex
	seen []time.Duration
	got  chan struct{}
}

func newTTFBWatcher() *ttfbWatcher {
	return &ttfbWatcher{got: make(chan struct{}, 1)}
}

func (w *ttfbWatcher) observe(f frames.Frame) {
	mf, ok := f.(*frames.MetricsFrame)
	if !ok {
		return
	}
	for _, d := range mf.Data {
		td, ok := d.(frames.TTFBMetricsData)
		if !ok {
			continue
		}
		w.mu.Lock()
		w.seen = append(w.seen, td.Value)
		w.mu.Unlock()
		select {
		case w.got <- struct{}{}:
		default:
		}
	}
}

func (w *ttfbWatcher) reported() []time.Duration {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]time.Duration(nil), w.seen...)
}

// newMeasuredTask runs svc with metrics on and the zeroed opening frame off, so
// the only measurements the watcher sees are the ones the service reported.
func newMeasuredTask(svc processor.Processor, w *ttfbWatcher) *pipeline.Worker {
	no := false
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &no,
		},
	})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream,
		func(_ context.Context, f frames.Frame) { w.observe(f) })
	return worker
}

// The wait reported is measured from the moment the speech ended, which is the
// VAD's determination less the silence it required before making it. Timing from
// the determination instead would hide that silence from every measurement.
func TestStreamServiceMeasuresFromTheEndOfSpeech(t *testing.T) {
	const silence = 500 * time.Millisecond

	stream := &answeringStream{told: make(chan struct{}, 1)}
	svc := stt.NewStream("ConfirmingSTT", &answeringConnector{stream: stream}, 16000)
	w := newTTFBWatcher()
	task := newMeasuredTask(svc, w)

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(silence.Seconds(), time.Now()))
	select {
	case <-w.got:
	case <-time.After(3 * time.Second):
		t.Fatal("no time to first byte was reported for the utterance")
	}
	task.StopWhenDone()
	<-runDone

	seen := w.reported()
	if len(seen) != 1 {
		t.Fatalf("reported %d measurements, want 1: %v", len(seen), seen)
	}
	if seen[0] < silence {
		t.Fatalf("ttfb = %v, want at least the %v of silence the VAD waited out", seen[0], silence)
	}
}

// A service that answers nothing has no latency to report. Timing to the moment
// the wait was given up on would report the deadline rather than the service.
func TestStreamServiceReportsNothingWithoutATranscript(t *testing.T) {
	svc := stt.NewStream("SilentSTT", &fakeConnector{stream: &fakeStream{}}, 16000)
	svc.SetTTFBTimeout(50 * time.Millisecond)
	w := newTTFBWatcher()
	task := newMeasuredTask(svc, w)

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	select {
	case <-w.got:
		t.Fatal("a measurement was reported for an utterance nothing was transcribed from")
	case <-time.After(300 * time.Millisecond):
	}
	task.StopWhenDone()
	<-runDone
}

// processingWatcher collects the processing time a service reports.
type processingWatcher struct {
	mu   sync.Mutex
	seen []time.Duration
	got  chan struct{}
}

func newProcessingWatcher() *processingWatcher {
	return &processingWatcher{got: make(chan struct{}, 1)}
}

func (w *processingWatcher) observe(f frames.Frame) {
	mf, ok := f.(*frames.MetricsFrame)
	if !ok {
		return
	}
	for _, d := range mf.Data {
		pd, ok := d.(frames.ProcessingMetricsData)
		if !ok {
			continue
		}
		w.mu.Lock()
		w.seen = append(w.seen, pd.Value)
		w.mu.Unlock()
		select {
		case w.got <- struct{}{}:
		default:
		}
	}
}

// The work a service did on an utterance is reported alongside the wait it cost.
// The two answer different questions: a streaming service is at work for the
// length of the utterance, where the wait it leaves behind is only what follows
// the speech.
func TestStreamServiceReportsTheWorkItDid(t *testing.T) {
	stream := &answeringStream{told: make(chan struct{}, 1)}
	svc := stt.NewStream("ConfirmingSTT", &answeringConnector{stream: stream}, 16000)
	w := newProcessingWatcher()
	no := false
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &no,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream,
		func(_ context.Context, f frames.Frame) { w.observe(f) })

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	select {
	case <-w.got:
	case <-time.After(3 * time.Second):
		t.Fatal("no processing time was reported for the utterance")
	}
	task.StopWhenDone()
	<-runDone

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.seen) != 1 {
		t.Fatalf("reported %d measurements, want 1: %v", len(w.seen), w.seen)
	}
	if w.seen[0] <= 0 {
		t.Fatalf("processing time = %v, want the stretch the service was at work", w.seen[0])
	}
}

// repeatAnsweringStream answers every finalize it is told to make, so a test can
// run more than one utterance through it.
type repeatAnsweringStream struct {
	told chan struct{}
	ctx  context.Context //nolint:containedctx // the session context, set on dial
}

func (s *repeatAnsweringStream) Send([]byte) error { return nil }

func (s *repeatAnsweringStream) Close() error { return nil }

func (s *repeatAnsweringStream) Finalize() error {
	select {
	case s.told <- struct{}{}:
	default:
	}
	return nil
}

func (s *repeatAnsweringStream) Recv() ([]stt.Result, error) {
	select {
	case <-s.told:
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
	return []stt.Result{{Text: "hello world", Final: true, FromFinalize: true}}, nil
}

type repeatAnsweringConnector struct{ stream *repeatAnsweringStream }

func (c *repeatAnsweringConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// Asked for the initial figure alone, a service measures the first utterance and
// leaves the rest alone.
func TestStreamServiceReportsOnlyTheInitialTTFBWhenAsked(t *testing.T) {
	stream := &repeatAnsweringStream{told: make(chan struct{}, 1)}
	svc := stt.NewStream("OnceOnlySTT", &repeatAnsweringConnector{stream: stream}, 16000)
	w := newTTFBWatcher()
	no := false
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &no,
			ReportOnlyInitialTTFB:   true,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream,
		func(_ context.Context, f frames.Frame) { w.observe(f) })

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for range 2 {
		task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
		task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
		select {
		case <-w.got:
		case <-time.After(time.Second):
		}
	}
	task.StopWhenDone()
	<-runDone

	if seen := w.reported(); len(seen) != 1 {
		t.Fatalf("reported %d measurements, want only the initial one: %v", len(seen), seen)
	}
}

// A transcript that does not close the utterance still says the service
// answered. When nothing closes it, the wait reported is the one to that
// transcript rather than the one to the deadline that gave up on it.
func TestStreamServiceMeasuresToTheLastTranscriptWhenNoneCloses(t *testing.T) {
	const (
		silence  = 200 * time.Millisecond
		deadline = 150 * time.Millisecond
	)

	conn := &fakeConnector{stream: &fakeStream{results: [][]stt.Result{
		{{Text: "hello world", Final: true}},
	}}}
	svc := stt.NewStream("OpenEndedSTT", conn, 16000)
	svc.SetTTFBTimeout(deadline)
	w := newTTFBWatcher()
	task := newMeasuredTask(svc, w)

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(silence.Seconds(), time.Now()))
	select {
	case <-w.got:
	case <-time.After(3 * time.Second):
		t.Fatal("the deadline reported nothing for a transcript that had arrived")
	}
	task.StopWhenDone()
	<-runDone

	seen := w.reported()
	if len(seen) != 1 {
		t.Fatalf("reported %d measurements, want 1: %v", len(seen), seen)
	}
	// The transcript arrived at once, so the wait is the silence the VAD waited
	// out and no more. Measuring to the deadline instead would add its own wait
	// on top, and report a service that answered immediately as slow.
	if seen[0] < silence {
		t.Fatalf("ttfb = %v, want at least the %v of silence", seen[0], silence)
	}
	if seen[0] >= silence+deadline {
		t.Fatalf("ttfb = %v, want the wait to the transcript, not to the deadline that gave up on it",
			seen[0])
	}
}

// answeringEmptyStream answers the finalize with nothing left to say, the way a
// provider does when the transcript went out ahead of the confirmation.
type answeringEmptyStream struct{ answeringStream }

func (s *answeringEmptyStream) Recv() ([]stt.Result, error) {
	if !s.sent {
		select {
		case <-s.told:
		case <-s.ctx.Done():
			return nil, s.ctx.Err()
		}
		s.sent = true
		return []stt.Result{{Final: true, FromFinalize: true}}, nil
	}
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

type answeringEmptyConnector struct{ stream *answeringEmptyStream }

func (c *answeringEmptyConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// A provider confirming it has nothing further to send ends the wait there and
// then. Falling to the deadline instead would delay the measurement by it.
func TestStreamServiceReportsOnAnEmptyFinalizeAnswer(t *testing.T) {
	stream := &answeringEmptyStream{answeringStream{told: make(chan struct{}, 1)}}
	svc := stt.NewStream("ConfirmingSTT", &answeringEmptyConnector{stream: stream}, 16000)
	// Long enough that a measurement arriving promptly cannot have come from it.
	svc.SetTTFBTimeout(30 * time.Second)
	w := newTTFBWatcher()
	task := newMeasuredTask(svc, w)

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	select {
	case <-w.got:
	case <-time.After(3 * time.Second):
		t.Fatal("the wait was left to the deadline after the provider said it was done")
	}
	task.StopWhenDone()
	<-runDone
}

// A VAD that does not say how long it waited gives nothing to measure against:
// the silence it required is indistinguishable from the wait for the transcript.
func TestStreamServiceSkipsAVADThatDidNotSayHowLongItWaited(t *testing.T) {
	stream := &answeringStream{told: make(chan struct{}, 1)}
	svc := stt.NewStream("ConfirmingSTT", &answeringConnector{stream: stream}, 16000)
	w := newTTFBWatcher()
	task := newMeasuredTask(svc, w)

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))
	select {
	case <-w.got:
		t.Fatal("a measurement was reported with no end of speech to measure from")
	case <-time.After(300 * time.Millisecond):
	}
	task.StopWhenDone()
	<-runDone

	if seen := w.reported(); len(seen) != 0 {
		t.Fatalf("reported %v, want nothing", seen)
	}
}
