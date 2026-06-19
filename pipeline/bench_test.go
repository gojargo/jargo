package pipeline

import (
	"context"
	"fmt"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// These benchmarks measure jargo's frame-plumbing and concurrency, the parts of
// the architecture that genuinely differ from Pipecat's asyncio model (there is
// no shared C++ runtime here, unlike the turn-taking benchmarks). They time the
// framework overhead with no per-processor work, so the numbers are the cost of
// the plumbing itself. The matching Pipecat harness is benchmarks/python/bench_pipeline.py.

// passthrough is a data-forwarding processor with no work of its own: it applies
// the base lifecycle handling, then forwards every frame to its neighbor. A
// chain of these isolates the per-hop plumbing cost — queue push, goroutine
// handoff, dispatch — from any real processing.
type passthrough struct{ *processor.Base }

func newPassthrough() *passthrough {
	p := &passthrough{}
	p.Base = processor.New("Passthrough", p)
	return p
}

func (p *passthrough) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	return p.PushFrame(ctx, f, dir)
}

// newChain builds a running Task whose pipeline is depth passthrough processors,
// signaling arrived for every TextFrame that reaches the sink. It returns the
// task, the arrival channel, and a stop func that drains and shuts it down.
func newChain(tb testing.TB, depth int, arrivedBuf int) (*Task, <-chan struct{}, func()) {
	tb.Helper()

	procs := make([]processor.Processor, depth)
	for i := range procs {
		procs[i] = newPassthrough()
	}

	arrived := make(chan struct{}, arrivedBuf)
	task := NewTask(New(procs...), TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TextFrame); ok {
				arrived <- struct{}{}
			}
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = task.Run(ctx); close(done) }()

	stop := func() {
		task.Cancel()
		cancel()
		<-done
	}
	return task, arrived, stop
}

// BenchmarkFramePlumbing measures the latency of one TextFrame traveling through
// a chain of depth pass-through processors, including every goroutine handoff.
// ns/op is the end-to-end chain latency; the reported ns/hop divides it by depth
// for a per-processor figure. A single frame is reused so the measurement is the
// plumbing cost, not frame allocation.
func BenchmarkFramePlumbing(b *testing.B) {
	for _, depth := range []int{1, 4, 16, 64} {
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			task, arrived, stop := newChain(b, depth, 1)
			defer stop()
			frame := frames.NewTextFrame("x")

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				task.QueueFrame(frame)
				<-arrived
			}
			b.StopTimer()
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(depth), "ns/hop")
		})
	}
}

// BenchmarkConcurrentSessions measures aggregate frame throughput as the number
// of independent pipelines (sessions) running at once scales up. Each iteration
// pushes one frame into every session and waits for all of them to arrive, so
// ns/op is the time for one round of N parallel frames and frames/s is the
// aggregate rate. This is where goroutines should scale across cores while a
// single asyncio event loop cannot.
func BenchmarkConcurrentSessions(b *testing.B) {
	const depth = 4
	for _, sessions := range []int{1, 10, 100, 1000} {
		b.Run(fmt.Sprintf("sessions=%d", sessions), func(b *testing.B) {
			tasks := make([]*Task, sessions)
			arrived := make(chan struct{}, sessions)
			stops := make([]func(), sessions)
			for s := range tasks {
				t, a, stop := newChain(b, depth, sessions)
				// Fan every session's arrivals into one shared channel.
				go func(a <-chan struct{}) {
					for range a {
						arrived <- struct{}{}
					}
				}(a)
				tasks[s] = t
				stops[s] = stop
			}
			defer func() {
				for _, stop := range stops {
					stop()
				}
			}()

			frame := frames.NewTextFrame("x")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, t := range tasks {
					t.QueueFrame(frame)
				}
				for range tasks {
					<-arrived
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(b.N)*float64(sessions)/b.Elapsed().Seconds(), "frames/s")
		})
	}
}
