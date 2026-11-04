package run

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
)

// ErrTimeout is returned when a sample runs past its budget. The sessions it
// started have been canceled by the time it is returned.
var ErrTimeout = errors.New("budget exceeded")

// Metric is one value of a result, in a benchfmt unit such as ns/frame.
type Metric struct {
	Value float64
	Unit  string
}

// Result is one sample of one sub-benchmark: a benchfmt line once the name is
// prefixed with Benchmark.
type Result struct {
	// Iterations is the benchfmt iteration count: the frames, rounds or
	// startups the sample measured.
	Iterations int
	Metrics    []Metric
}

// unitFrame is the time one frame took, on average.
const unitFrame = "ns/frame"

// unitFramesPerSec is how many frames went through per second.
const unitFramesPerSec = "frames/s"

// Text is the frame payload. Every frame is built fresh, as a real pipeline's
// would be, and both sides send the same one.
const text = "x"

// Latency measures closed-loop latency through a chain of depth processors. The
// pipeline is started first and warmup frames sent through it, none of which is
// timed. Then each of frames frames is queued only once the one before it has
// reached the end, and timed from being queued to arriving.
func Latency(ctx context.Context, depth, warmup, frameCount int) (Result, error) {
	arrived := make(chan struct{}, 1)
	s := NewSession(depth, arrived)
	if err := startOne(ctx, s); err != nil {
		return Result{}, err
	}

	for range warmup {
		s.Send(frames.NewTextFrame(text))
		if err := Wait(ctx, arrived); err != nil {
			return Result{}, abandon(err, s)
		}
	}

	samples := make([]time.Duration, frameCount)
	for i := range samples {
		f := frames.NewTextFrame(text)
		start := time.Now()
		s.Send(f)
		if err := Wait(ctx, arrived); err != nil {
			return Result{}, abandon(err, s)
		}
		samples[i] = time.Since(start)
	}
	if err := s.Stop(ctx); err != nil {
		return Result{}, err
	}

	var total time.Duration
	for _, d := range samples {
		total += d
	}
	mean := float64(total.Nanoseconds()) / float64(frameCount)
	return Result{
		Iterations: frameCount,
		Metrics: []Metric{
			{mean, unitFrame},
			{mean / float64(depth), "ns/hop"},
			{float64(percentile(samples, 0.50).Nanoseconds()), "p50-ns/frame"},
			{float64(percentile(samples, 0.99).Nanoseconds()), "p99-ns/frame"},
		},
	}, nil
}

// Throughput measures open-loop throughput through a chain of depth processors.
// After starting the pipeline and sending warmup frames through it one at a
// time, it builds frames frames, queues them all without waiting, and times
// from the first being queued to the last arriving.
func Throughput(ctx context.Context, depth, warmup, frameCount int) (Result, error) {
	arrived := make(chan struct{}, frameCount)
	s := NewSession(depth, arrived)
	if err := startOne(ctx, s); err != nil {
		return Result{}, err
	}

	for range warmup {
		s.Send(frames.NewTextFrame(text))
		if err := Wait(ctx, arrived); err != nil {
			return Result{}, abandon(err, s)
		}
	}

	batch := make([]frames.Frame, frameCount)
	for i := range batch {
		batch[i] = frames.NewTextFrame(text)
	}
	start := time.Now()
	for _, f := range batch {
		s.Send(f)
	}
	for range frameCount {
		if err := Wait(ctx, arrived); err != nil {
			return Result{}, abandon(err, s)
		}
	}
	elapsed := time.Since(start)
	if err := s.Stop(ctx); err != nil {
		return Result{}, err
	}

	return Result{
		Iterations: frameCount,
		Metrics: []Metric{
			{float64(elapsed.Nanoseconds()) / float64(frameCount), unitFrame},
			{float64(frameCount) / elapsed.Seconds(), unitFramesPerSec},
		},
	}, nil
}

// Startup measures starting a pipeline of depth processors, repeats times over.
// Building the pipeline, the worker and the runner is reported apart, as
// build-ns/start, from running them until the StartFrame has reached the end of
// the pipeline, as ns/start. Stopping each one is not timed.
func Startup(ctx context.Context, depth, repeats int) (Result, error) {
	var build, start time.Duration
	for range repeats {
		t0 := time.Now()
		s := NewSession(depth, nil)
		t1 := time.Now()
		if err := startOne(ctx, s); err != nil {
			return Result{}, err
		}
		build += t1.Sub(t0)
		start += time.Since(t1)
		if err := s.Stop(ctx); err != nil {
			return Result{}, err
		}
	}
	return Result{
		Iterations: repeats,
		Metrics: []Metric{
			{float64(start.Nanoseconds()) / float64(repeats), "ns/start"},
			{float64(build.Nanoseconds()) / float64(repeats), "build-ns/start"},
		},
	}, nil
}

// Sessions measures what each of count pipelines of depth processors costs when
// they run at once. All of them are started together, and the time until the
// last is ready is reported per session as startup-ns/session. Then warmup
// rounds, and rounds timed rounds, each sending one frame into every pipeline
// and waiting for all of them to come out. The resident memory gained from
// before the first was built to after the last round is reported per session as
// B/session. Freed memory is handed back to the system before both readings, so
// what is counted is what the running sessions hold, not garbage not yet
// collected.
func Sessions(ctx context.Context, depth, count, warmup, rounds int) (Result, error) {
	releaseMemory()
	before, haveMemory := residentBytes()

	arrived := make(chan struct{}, count)
	group := make([]*Session, count)
	for i := range group {
		group[i] = NewSession(depth, arrived)
	}

	start := time.Now()
	for _, s := range group {
		s.Start(ctx)
	}
	for _, s := range group {
		if err := s.WaitStarted(ctx); err != nil {
			return Result{}, abandon(err, group...)
		}
	}
	startup := time.Since(start)

	round := func() error {
		for _, s := range group {
			s.Send(frames.NewTextFrame(text))
		}
		for range group {
			if err := Wait(ctx, arrived); err != nil {
				return err
			}
		}
		return nil
	}
	for range warmup {
		if err := round(); err != nil {
			return Result{}, abandon(err, group...)
		}
	}
	start = time.Now()
	for range rounds {
		if err := round(); err != nil {
			return Result{}, abandon(err, group...)
		}
	}
	elapsed := time.Since(start)

	releaseMemory()
	after, _ := residentBytes()
	if err := StopAll(ctx, group); err != nil {
		return Result{}, err
	}

	moved := float64(rounds) * float64(count)
	res := Result{
		Iterations: rounds,
		Metrics: []Metric{
			{float64(elapsed.Nanoseconds()) / float64(rounds), "ns/round"},
			{float64(elapsed.Nanoseconds()) / moved, unitFrame},
			{moved / elapsed.Seconds(), unitFramesPerSec},
			{float64(startup.Nanoseconds()) / float64(count), "startup-ns/session"},
		},
	}
	// Resident memory is read from /proc, so it is only reported where there
	// is one.
	if haveMemory {
		res.Metrics = append(res.Metrics, Metric{float64(after-before) / float64(count), "B/session"})
	}
	return res, nil
}

// StopAll stops every session at once and waits for them all.
func StopAll(ctx context.Context, group []*Session) error {
	errs := make([]error, len(group))
	var wg sync.WaitGroup
	for i, s := range group {
		wg.Go(func() { errs[i] = s.Stop(ctx) })
	}
	wg.Wait()
	return errors.Join(errs...)
}

// startOne starts a session and waits for it to be ready.
func startOne(ctx context.Context, s *Session) error {
	s.Start(ctx)
	if err := s.WaitStarted(ctx); err != nil {
		return abandon(err, s)
	}
	return nil
}

// abandon cancels the sessions of a sample that failed and returns why it did,
// as ErrTimeout when it ran out of time. The sessions are canceled on a context
// of their own, since the sample's may be the one that ended.
func abandon(cause error, group ...*Session) error {
	errs := make([]error, len(group))
	var wg sync.WaitGroup
	for i, s := range group {
		wg.Go(func() { errs[i] = s.Cancel(context.Background()) })
	}
	wg.Wait()
	if errors.Is(cause, context.DeadlineExceeded) {
		cause = ErrTimeout
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("%w (and canceling: %w)", cause, err)
	}
	return cause
}

// percentile returns the smallest sample at least the fraction p of the samples
// do not exceed, by the nearest-rank method. It sorts samples in place.
func percentile(samples []time.Duration, p float64) time.Duration {
	slices.Sort(samples)
	rank := int(math.Ceil(p * float64(len(samples))))
	return samples[max(0, min(rank-1, len(samples)-1))]
}
