package benchmarks_test

import (
	"context"
	"fmt"
	"log/slog"
	"testing"

	"github.com/gojargo/jargo/benchmarks/internal/run"
	"github.com/gojargo/jargo/frames"
)

// These are the jbench kinds as testing.B benchmarks, for following jargo's own
// numbers from one commit to the next with go test and benchstat. They share
// the sessions and pass-through chain jbench runs, and leave the comparison
// with the Python side to jbench, which holds both to the same scenario.

// depths are the chain depths the plumbing benchmarks run at.
//
//nolint:gochecknoglobals // a fixed table of sizes
var depths = []int{1, 4, 16, 64}

// quiet discards log records for the benchmark, as jbench does, so the logger
// stays off the measured path.
func quiet(b *testing.B) {
	b.Helper()
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.DiscardHandler))
	b.Cleanup(func() { slog.SetDefault(previous) })
}

// started builds a session of depth processors, starts it, and stops it when
// the benchmark ends.
func started(b *testing.B, depth int, arrived chan struct{}) *run.Session {
	b.Helper()
	s := run.NewSession(depth, arrived)
	s.Start(context.Background())
	if err := s.WaitStarted(b.Context()); err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if err := s.Stop(context.Background()); err != nil {
			b.Error(err)
		}
	})
	return s
}

// BenchmarkPlumbingLatency is the closed-loop latency of one TextFrame through
// a chain of pass-through processors: queued, then waited for at the end before
// the next is queued. ns/op is the whole chain and ns/hop one processor.
func BenchmarkPlumbingLatency(b *testing.B) {
	quiet(b)
	for _, depth := range depths {
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			arrived := make(chan struct{}, 1)
			s := started(b, depth, arrived)
			ctx := b.Context()
			b.ReportAllocs()
			for b.Loop() {
				s.Send(frames.NewTextFrame("x"))
				if err := run.Wait(ctx, arrived); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(depth), "ns/hop")
		})
	}
}

// BenchmarkPlumbingThroughput queues b.N TextFrames at once and times them
// until the last has reached the end of the chain.
func BenchmarkPlumbingThroughput(b *testing.B) {
	quiet(b)
	for _, depth := range depths {
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			arrived := make(chan struct{}, b.N)
			s := started(b, depth, arrived)
			ctx := b.Context()
			batch := make([]frames.Frame, b.N)
			for i := range batch {
				batch[i] = frames.NewTextFrame("x")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for _, f := range batch {
				s.Send(f)
			}
			for range batch {
				if err := run.Wait(ctx, arrived); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "frames/s")
		})
	}
}

// BenchmarkStartup is the time from running a session to its StartFrame having
// reached the end of the chain. Building and stopping it are not timed.
func BenchmarkStartup(b *testing.B) {
	quiet(b)
	for _, depth := range depths {
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			ctx := b.Context()
			b.ReportAllocs()
			for b.Loop() {
				b.StopTimer()
				s := run.NewSession(depth, nil)
				b.StartTimer()
				s.Start(context.Background())
				if err := s.WaitStarted(ctx); err != nil {
					b.Fatal(err)
				}
				b.StopTimer()
				if err := s.Stop(context.Background()); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
			}
		})
	}
}

// BenchmarkSessions runs many sessions with a chain of four at once. An
// iteration is one round: a TextFrame into every session, and all of them back
// out. The thousand-session size is left to jbench's full profile.
func BenchmarkSessions(b *testing.B) {
	quiet(b)
	const depth = 4
	for _, count := range []int{1, 10, 100} {
		b.Run(fmt.Sprintf("sessions=%d", count), func(b *testing.B) {
			arrived := make(chan struct{}, count)
			group := make([]*run.Session, count)
			for i := range group {
				group[i] = run.NewSession(depth, arrived)
				group[i].Start(context.Background())
			}
			b.Cleanup(func() {
				if err := run.StopAll(context.Background(), group); err != nil {
					b.Error(err)
				}
			})
			ctx := b.Context()
			for _, s := range group {
				if err := s.WaitStarted(ctx); err != nil {
					b.Fatal(err)
				}
			}
			for b.Loop() {
				for _, s := range group {
					s.Send(frames.NewTextFrame("x"))
				}
				for range group {
					if err := run.Wait(ctx, arrived); err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(b.N)*float64(count)/b.Elapsed().Seconds(), "frames/s")
		})
	}
}
