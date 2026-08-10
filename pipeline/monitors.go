package pipeline

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// monitorStopTimeout bounds how long the task waits for its monitor goroutines
// to exit before giving up on them and finishing the run.
const monitorStopTimeout = 3 * time.Second

// monitors is the scope the task's background goroutines run in: the heartbeat
// pusher, the heartbeat monitor and the idle monitor. They are the only work the
// task owns outside the run loop, which runs on the caller's goroutine.
//
// The scope is closed once, at the end of the run and before the pipeline is
// cleaned up, so no monitor is still pushing frames into a pipeline that is
// shutting down.
type monitors struct {
	mu      sync.Mutex
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
	stopped bool
	// started guards against starting the heartbeat goroutines twice, since the
	// StartFrame that triggers them can in principle reach the sink more than
	// once across an interruption.
	started bool
}

// start opens the scope. It returns the context the monitors run under, which
// the run loop uses too so the whole run ends together.
//
// The scope keeps the context rather than handing it to each caller of spawn:
// the heartbeats are started from the sink, where the context in hand belongs to
// the frame passing through and not to the run.
func (m *monitors) start(ctx context.Context) context.Context {
	monCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.ctx, m.cancel = monCtx, cancel
	m.mu.Unlock()
	return monCtx
}

// spawn runs fn on its own goroutine inside the scope, or does nothing once the
// scope has been closed or before it has been opened. It is called from the run
// loop and from the sink's goroutine, so it counts the goroutine in under the
// lock that stop takes.
func (m *monitors) spawn(fn func(context.Context)) {
	m.mu.Lock()
	if m.stopped || m.ctx == nil {
		m.mu.Unlock()
		return
	}
	ctx := m.ctx
	m.wg.Add(1)
	m.mu.Unlock()

	go func() {
		defer m.wg.Done()
		fn(ctx)
	}()
}

// stop closes the scope and waits for the goroutines to exit, bounded so a
// monitor wedged in a call cannot hold the run open for good.
func (m *monitors) stop() {
	m.mu.Lock()
	if m.stopped {
		m.mu.Unlock()
		return
	}
	m.stopped = true
	cancel := m.cancel
	m.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	done := make(chan struct{})
	go func() {
		m.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(monitorStopTimeout):
		slog.Warn("timed out stopping the task's monitor goroutines")
	}
}

// startHeartbeats starts the heartbeat pusher and monitor, once, if heartbeats
// are enabled. It runs when the StartFrame reaches the sink rather than when the
// run begins: every processor has seen its StartFrame by then, so no heartbeat
// is dropped for arriving before one.
func (t *Task) startHeartbeats() {
	if !t.params.EnableHeartbeats {
		return
	}
	t.monitors.mu.Lock()
	if t.monitors.started {
		t.monitors.mu.Unlock()
		return
	}
	t.monitors.started = true
	t.monitors.mu.Unlock()

	t.monitors.spawn(t.heartbeatPushLoop)
	t.monitors.spawn(t.heartbeatMonitorLoop)
}

// heartbeatPushLoop sends a heartbeat into the pipeline every period.
func (t *Task) heartbeatPushLoop(ctx context.Context) {
	for {
		// Straight into the pipeline, not through the task's queue. The run loop
		// stops draining that queue the moment a pipeline-ending frame goes into
		// it, since it then waits for that frame to travel through, so a
		// heartbeat queued behind one would never be pushed at all.
		hb := frames.NewHeartbeatFrame(t.clk.Time())
		if err := t.pipeline.QueueFrame(ctx, hb, processor.Downstream); err != nil {
			return
		}
		timer := time.NewTimer(t.params.HeartbeatPeriod)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			return
		}
	}
}

// idleObserver signals activity whenever a frame the caller counts as activity
// is pushed anywhere in the pipeline. It is how the task tells a conversation
// that has gone quiet from one that is merely between turns.
type idleObserver struct {
	match FrameFilter
	sig   chan<- struct{}
}

// OnPushFrame implements processor.Observer.
func (o *idleObserver) OnPushFrame(data processor.FramePushed) {
	// The StartFrame counts, so the first idle interval is measured from the
	// pipeline coming up rather than from the task being built.
	if _, isStart := data.Frame.(*frames.StartFrame); !isStart && !o.match.selects(data.Frame) {
		return
	}
	// A frame is reported at every handover, so the same one arrives many times.
	// A non-blocking send onto a channel holding one signal absorbs that: the
	// monitor only needs to know something happened, not how often.
	select {
	case o.sig <- struct{}{}:
	default:
	}
}

// idleMonitorLoop watches for the pipeline going quiet for longer than the idle
// timeout.
func (t *Task) idleMonitorLoop(ctx context.Context) {
	for {
		timer := time.NewTimer(t.params.IdleTimeout)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-t.idleSig:
			timer.Stop()
		case <-timer.C:
			if !t.idleTimeoutReached() {
				return
			}
		}
	}
}

// idleTimeoutReached reports the pipeline going quiet and returns whether to
// keep watching.
func (t *Task) idleTimeoutReached() bool {
	// Already on the way out, so there is nothing left to notice.
	if t.isCanceling() {
		return false
	}

	slog.Warn("the pipeline has gone quiet", "timeout", t.params.IdleTimeout)
	if t.params.OnIdleTimeout != nil {
		t.params.OnIdleTimeout()
	}
	if !boolValue(t.params.CancelOnIdleTimeout, true) {
		// The caller wants to hear about it and decide for itself, so keep
		// watching and report the next stretch of quiet too.
		return true
	}

	slog.Warn("canceling the idle pipeline")
	t.cancelWithReason("idle timeout")
	return false
}

// heartbeatMonitorLoop watches the heartbeats arriving at the end of the
// pipeline and reports the interval passing with none.
//
// It keeps reporting for as long as the silence lasts, one report per interval,
// rather than reporting once and going quiet: a pipeline that is still stuck a
// minute later is still worth hearing about.
func (t *Task) heartbeatMonitorLoop(ctx context.Context) {
	for {
		waitCtx, cancel := context.WithTimeout(ctx, t.params.HeartbeatMonitorTimeout)
		f, ok := t.heartbeats.get(waitCtx)
		cancel()

		if ok {
			if hb, isHB := f.(*frames.HeartbeatFrame); isHB {
				slog.Debug("heartbeat crossed the pipeline", "duration", t.clk.Time()-hb.Timestamp)
			}
			continue
		}
		if ctx.Err() != nil {
			return // The scope closed, rather than the wait timing out.
		}
		slog.Warn("no heartbeat reached the end of the pipeline",
			"timeout", t.params.HeartbeatMonitorTimeout)
		if t.params.OnHeartbeatTimeout != nil {
			t.params.OnHeartbeatTimeout()
		}
	}
}
