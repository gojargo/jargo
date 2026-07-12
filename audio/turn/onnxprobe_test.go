package turn

import (
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/internal/onnxrt"
)

// probeFeatures returns a valid-shaped [1,80,800] input. Run cost is
// input-independent for this dense model, so the values only need to avoid
// denormals.
func probeFeatures() []float32 {
	f := make([]float32, nMels*nFrames)
	for i := range f {
		f[i] = 0.01
	}
	return f
}

// newProbeSession loads the model with intra-op threads pinned to 1. That keeps
// each Run on its calling thread, so the probe measures the Go scheduler's
// handling of concurrent native calls rather than ONNX Runtime's own thread
// pool oversubscribing the machine (streams × cores threads).
func newProbeSession(tb testing.TB) *onnxrt.Session {
	tb.Helper()
	s, err := onnxrt.NewWithOptions(smartTurnModel,
		[]string{"input_features"}, []string{"logits"},
		onnxrt.Options{IntraOpThreads: 1})
	if err != nil {
		tb.Fatalf("load model: %v", err)
	}
	return s
}

// BenchmarkSmartTurnInference times one Run on the active backend (cgo/yalue or
// purego). Compare across builds with:
//
//	CGO_ENABLED=1 go test ./audio/turn -run x -bench SmartTurnInference
//	CGO_ENABLED=0 go test ./audio/turn -run x -bench SmartTurnInference
func BenchmarkSmartTurnInference(b *testing.B) {
	if !onnxrt.Available() {
		b.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	sess := newProbeSession(b)
	defer func() { _ = sess.Close() }()

	in := []onnxrt.Tensor{onnxrt.Float32([]int64{1, nMels, nFrames}, probeFeatures())}
	b.ResetTimer()
	for range b.N {
		if _, err := sess.Run(in); err != nil {
			b.Fatal(err)
		}
	}
}

// TestSmartTurnSchedulerProbe measures how much a normal goroutine is delayed
// while N goroutines run inference concurrently, each on its own session (the
// one-session-per-stream production shape). It is the tool for deciding whether
// the purego backend's inability to hand off a P during a native call — unlike
// cgo — starves the rest of the process, and at what concurrency. Run it under
// both build configs and compare the hb_max column.
func TestSmartTurnSchedulerProbe(t *testing.T) {
	if !onnxrt.Available() {
		t.Skip("ONNX runtime not configured; set JARGO_ONNXRUNTIME_LIB")
	}
	if testing.Short() {
		t.Skip("scheduler probe is slow; skipped under -short")
	}

	g := runtime.GOMAXPROCS(0)
	counts := dedupSorted([]int{0, 1, g / 2, g, 2 * g})
	maxW := counts[len(counts)-1]

	sessions := make([]*onnxrt.Session, maxW)
	for i := range sessions {
		sessions[i] = newProbeSession(t)
		defer func(s *onnxrt.Session) { _ = s.Close() }(sessions[i])
	}
	in := []onnxrt.Tensor{onnxrt.Float32([]int64{1, nMels, nFrames}, probeFeatures())}

	// Warm every session so the first measured Run doesn't pay ONNX Runtime's
	// lazy allocation/optimization, then settle the heap so a GC pause doesn't
	// land on the baseline row.
	for _, s := range sessions {
		for range 2 {
			if _, err := s.Run(in); err != nil {
				t.Fatalf("warmup: %v", err)
			}
		}
	}
	runtime.GC()

	t.Logf("backend=%s GOMAXPROCS=%d intraOp=1", onnxrt.Backend(), g)
	t.Logf("%-8s %-11s %-11s %-11s %-10s", "workers", "hb_p50", "hb_p99", "hb_max", "runs/sec")
	for _, w := range counts {
		p50, p99, hbMax, rps := probeOnce(sessions[:w], in, 700*time.Millisecond)
		t.Logf("%-8d %-11s %-11s %-11s %-10.0f", w, ms(p50), ms(p99), ms(hbMax), rps)
	}
	t.Logf("hb_max is the worst delay a normal goroutine saw; it should stay near baseline")
	t.Logf("while workers < GOMAXPROCS and climb once every P is inside a native Run.")
}

// probeOnce runs one worker per session hammering Run while the calling
// goroutine keeps a 1ms heartbeat and records its scheduling delay.
func probeOnce(
	sessions []*onnxrt.Session, in []onnxrt.Tensor, dur time.Duration,
) (p50, p99, mx time.Duration, runsPerSec float64) {
	var runs int64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for _, s := range sessions {
		wg.Add(1)
		go func(s *onnxrt.Session) {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := s.Run(in); err != nil {
					return
				}
				atomic.AddInt64(&runs, 1)
			}
		}(s)
	}

	const tick = time.Millisecond
	var delays []time.Duration
	start := time.Now()
	for time.Since(start) < dur {
		t0 := time.Now()
		time.Sleep(tick)
		over := max(time.Since(t0)-tick, 0)
		delays = append(delays, over)
	}
	elapsed := time.Since(start)
	close(stop)
	wg.Wait()

	slices.Sort(delays)
	return pct(delays, 50), pct(delays, 99), delays[len(delays)-1],
		float64(atomic.LoadInt64(&runs)) / elapsed.Seconds()
}

func pct(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := len(sorted) * p / 100
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

func dedupSorted(xs []int) []int {
	slices.Sort(xs)
	out := xs[:0]
	for _, x := range xs {
		if x < 0 {
			continue
		}
		if len(out) == 0 || out[len(out)-1] != x {
			out = append(out, x)
		}
	}
	return out
}

func ms(d time.Duration) string {
	return d.Round(time.Microsecond).String()
}
