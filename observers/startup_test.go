package observers_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// slowStart holds the StartFrame for a while before passing it on, standing in
// for a service that connects, authenticates or loads a model as it starts.
type slowStart struct {
	*processor.Base
	delay time.Duration
}

func newSlowStart(delay time.Duration) *slowStart {
	p := &slowStart{delay: delay}
	p.Base = processor.New("SlowStart", p)
	return p
}

func (p *slowStart) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		time.Sleep(p.delay)
	}
	return p.PushFrame(ctx, f, dir)
}

// slowSetup takes a while to be set up, standing in for a service that connects
// while it is set up rather than while it handles the StartFrame.
type slowSetup struct {
	*processor.Base
	delay time.Duration
}

func newSlowSetup(delay time.Duration) *slowSetup {
	p := &slowSetup{delay: delay}
	p.Base = processor.New("SlowSetup", p)
	return p
}

func (p *slowSetup) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	time.Sleep(p.delay)
	return nil
}

func (p *slowSetup) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// fastStart passes everything straight on.
type fastStart struct {
	*processor.Base
}

func newFastStart() *fastStart {
	p := &fastStart{}
	p.Base = processor.New("FastStart", p)
	return p
}

func (p *fastStart) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// recorder collects what an observer reported. The reports arrive on the
// observer's own goroutine, and signal so a test can wait for one rather than
// racing the pipeline shutting down.
type recorder[T any] struct {
	mu  sync.Mutex
	got []T
	sig chan struct{}
}

func newRecorder[T any]() *recorder[T] { return &recorder[T]{sig: make(chan struct{}, 8)} }

func (r *recorder[T]) record(v T) {
	r.mu.Lock()
	r.got = append(r.got, v)
	r.mu.Unlock()
	select {
	case r.sig <- struct{}{}:
	default:
	}
}

func (r *recorder[T]) snapshot() []T {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]T(nil), r.got...)
}

// wait blocks until a report arrives, failing if none does.
func (r *recorder[T]) wait(t *testing.T) {
	t.Helper()
	select {
	case <-r.sig:
	case <-time.After(3 * time.Second):
		t.Fatal("nothing was ever reported")
	}
}

// runPipeline runs a worker over procs with o watching, queues send through it,
// then stops it and waits for the run to finish.
func runPipeline(t *testing.T, o pipeline.Observer, procs []processor.Processor, send ...frames.Frame) {
	t.Helper()
	w := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		Observers:   []pipeline.Observer{o},
		IdleTimeout: -1,
	})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	for _, f := range send {
		w.QueueFrame(f)
	}
	w.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("worker run error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the worker never finished")
	}
}

// TestStartupTimingReportsWhatEachProcessorCost covers the measurement itself:
// the gap between a processor being handed the StartFrame and passing it on is
// what that processor's own start cost.
func TestStartupTimingReportsWhatEachProcessorCost(t *testing.T) {
	r := newRecorder[observers.StartupTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnStartupTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newSlowStart(100 * time.Millisecond)},
		frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("reports = %d, want 1", len(got))
	}
	report := got[0]
	if report.TotalDuration <= 0 {
		t.Errorf("TotalDuration = %s, want more than nothing", report.TotalDuration)
	}
	if report.StartTime.IsZero() {
		t.Error("StartTime is unset, want the wall-clock time the pipeline started")
	}

	var slow []observers.ProcessorStartupTiming
	for _, timing := range report.ProcessorTimings {
		if timing.StartOffset < 0 {
			t.Errorf("%s started %s before the pipeline did", timing.ProcessorName, -timing.StartOffset)
		}
		if strings.HasPrefix(timing.ProcessorName, "SlowStart#") {
			slow = append(slow, timing)
		}
	}
	if len(slow) != 1 {
		t.Fatalf("timings naming the slow processor = %+v, want one", slow)
	}
	if slow[0].Duration < 50*time.Millisecond {
		t.Errorf("the slow processor was measured at %s, want at least 50ms", slow[0].Duration)
	}
}

// TestStartupTimingTrackNarrowsTheReport covers the filter: a caller who only
// cares about the services gets only those, not the plumbing around them.
func TestStartupTimingTrackNarrowsTheReport(t *testing.T) {
	r := newRecorder[observers.StartupTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		Track: func(p processor.Processor) bool {
			_, ok := p.(*slowStart)
			return ok
		},
		OnStartupTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newSlowStart(50 * time.Millisecond), newFastStart()},
		frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("reports = %d, want 1", len(got))
	}
	for _, timing := range got[0].ProcessorTimings {
		if !strings.HasPrefix(timing.ProcessorName, "SlowStart#") {
			t.Errorf("report names %s, want only the processors the filter selected", timing.ProcessorName)
		}
	}
}

// TestStartupTimingReportsOnce covers the report being made once for the run.
// Startup happens once, however many frames follow.
func TestStartupTimingReportsOnce(t *testing.T) {
	r := newRecorder[observers.StartupTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnStartupTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newFastStart()},
		frames.NewTextFrame("first"), frames.NewTextFrame("second"), frames.NewTextFrame("third"))
	r.wait(t)

	if got := r.snapshot(); len(got) != 1 {
		t.Fatalf("reports = %d, want 1", len(got))
	}
}

// TestStartupTimingLeavesOutThePipelinePlumbing covers the default filter. A
// pipeline hands the StartFrame to the chain it contains, so timing it would
// count that chain twice, and a source starts nothing of its own.
func TestStartupTimingLeavesOutThePipelinePlumbing(t *testing.T) {
	r := newRecorder[observers.StartupTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnStartupTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newFastStart()}, frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("reports = %d, want 1", len(got))
	}
	for _, timing := range got[0].ProcessorTimings {
		if strings.HasPrefix(timing.ProcessorName, "Pipeline#") || strings.Contains(timing.ProcessorName, "::Source") {
			t.Errorf("report names %s, want the pipeline plumbing left out", timing.ProcessorName)
		}
	}
}

// TestStartupTimingReportsTheClientConnecting covers the transport half: a
// client connecting is the point at which the call can actually happen.
func TestStartupTimingReportsTheClientConnecting(t *testing.T) {
	r := newRecorder[observers.TransportTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newFastStart()},
		frames.NewClientConnectedFrame(), frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("transport reports = %d, want 1", len(got))
	}
	if got[0].StartTime.IsZero() {
		t.Error("StartTime is unset, want the wall-clock time the pipeline started")
	}
	if got[0].ClientConnected <= 0 {
		t.Errorf("ClientConnected = %s, want the time it took to get there", got[0].ClientConnected)
	}
	if got[0].BotConnected != nil {
		t.Errorf("BotConnected = %s, want none: no transport reported the bot joining", *got[0].BotConnected)
	}
}

// TestStartupTimingReportsOnlyTheFirstClient covers a session more than one
// participant joins. The figure describes how long the call took to become
// answerable, which the second arrival says nothing about.
func TestStartupTimingReportsOnlyTheFirstClient(t *testing.T) {
	r := newRecorder[observers.TransportTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newFastStart()},
		frames.NewClientConnectedFrame(), frames.NewClientConnectedFrame(), frames.NewTextFrame("hello"))
	r.wait(t)

	if got := r.snapshot(); len(got) != 1 {
		t.Fatalf("transport reports = %d, want 1", len(got))
	}
}

// Ported from upstream. A client that connects before the StartFrame is still
// measured. A transport connects while it is being set up, so it can report a
// connection before the StartFrame is pushed; timings run from the pipeline
// starting to set up, so there is nothing to wait for.
func TestStartupTimingMeasuresAClientBeforeTheStart(t *testing.T) {
	r := newRecorder[observers.TransportTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: r.record,
	})

	o.OnPushFrame(processor.FramePushed{
		Frame:     frames.NewClientConnectedFrame(),
		Direction: processor.Downstream,
		Timestamp: 250 * time.Millisecond,
	})

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("transport reports = %+v, want one", got)
	}
	if got[0].ClientConnected != 250*time.Millisecond {
		t.Errorf("ClientConnected = %v, want %v", got[0].ClientConnected, 250*time.Millisecond)
	}
}

// TestStartupTimingReportsTheBotConnecting covers an SFU transport, where the
// bot joins the session itself before any client can be heard.
func TestStartupTimingReportsTheBotConnecting(t *testing.T) {
	r := newRecorder[observers.TransportTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newFastStart()},
		frames.NewBotConnectedFrame(), frames.NewClientConnectedFrame(), frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("transport reports = %d, want 1", len(got))
	}
	if got[0].BotConnected == nil {
		t.Fatal("BotConnected is unset, want the time the bot took to join")
	}
	if *got[0].BotConnected <= 0 {
		t.Errorf("BotConnected = %s, want the time it took to get there", *got[0].BotConnected)
	}
	if got[0].ClientConnected < *got[0].BotConnected {
		t.Errorf("the client connected at %s, before the bot did at %s",
			got[0].ClientConnected, *got[0].BotConnected)
	}
}

// TestStartupTimingKeepsTheFirstBotConnection covers a transport reporting the
// bot joining more than once. The first is the one that describes the startup.
func TestStartupTimingKeepsTheFirstBotConnection(t *testing.T) {
	r := newRecorder[observers.TransportTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnTransportTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newFastStart()},
		frames.NewBotConnectedFrame(), frames.NewBotConnectedFrame(),
		frames.NewClientConnectedFrame(), frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("transport reports = %d, want 1", len(got))
	}
	if got[0].BotConnected == nil {
		t.Fatal("BotConnected is unset, want the time the bot took to join")
	}
}

// Ported from upstream. A processor that connects while being set up is
// measured for it. Services connect during setup rather than while handling the
// StartFrame, so a report that only measured starting would show a fast startup
// for a pipeline that spent its time connecting.
func TestStartupTimingCountsWhatSetupCost(t *testing.T) {
	r := newRecorder[observers.StartupTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnStartupTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{newSlowSetup(100 * time.Millisecond)},
		frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("reports = %d, want 1", len(got))
	}

	var slow []observers.ProcessorStartupTiming
	for _, timing := range got[0].ProcessorTimings {
		if strings.HasPrefix(timing.ProcessorName, "SlowSetup#") {
			slow = append(slow, timing)
		}
	}
	if len(slow) != 1 {
		t.Fatalf("timings naming the slow processor = %+v, want one", slow)
	}
	if slow[0].SetupDuration < 50*time.Millisecond {
		t.Errorf("SetupDuration = %s, want at least 50ms", slow[0].SetupDuration)
	}
	// Setting up is what this processor cost, and starting added nothing.
	if slow[0].Duration < slow[0].SetupDuration {
		t.Errorf("Duration %s is less than the setup it contains, %s",
			slow[0].Duration, slow[0].SetupDuration)
	}
}

// Ported from upstream. Processors are set up concurrently, so their cost does
// not add up. Summing what each cost would report three concurrent 100ms
// connections as 300ms, so a pipeline would read as slower the more it
// overlapped.
func TestStartupTimingTotalIsTheSpanNotTheSum(t *testing.T) {
	r := newRecorder[observers.StartupTimingReport]()
	o := observers.NewStartupTiming(observers.StartupTimingConfig{
		OnStartupTimingReport: r.record,
	})

	runPipeline(t, o, []processor.Processor{
		newSlowSetup(100 * time.Millisecond),
		newSlowSetup(100 * time.Millisecond),
		newSlowSetup(100 * time.Millisecond),
	}, frames.NewTextFrame("hello"))
	r.wait(t)

	got := r.snapshot()
	if len(got) != 1 {
		t.Fatalf("reports = %d, want 1", len(got))
	}
	report := got[0]

	var summed time.Duration
	for _, timing := range report.ProcessorTimings {
		summed += timing.Duration
	}
	if summed < 250*time.Millisecond {
		t.Fatalf("the per-processor durations add up to %s, want each one measured", summed)
	}
	if report.TotalDuration >= summed {
		t.Errorf("TotalDuration = %s, want less than the %s they add up to: the setups overlapped",
			report.TotalDuration, summed)
	}
}
