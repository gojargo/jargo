package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// scriptGen runs each generation through the next script in line, so a test
// can time the model's output by hand.
type scriptGen struct {
	*llm.Base
	mu      sync.Mutex
	scripts []func(ctx context.Context, g *scriptGen, emit llm.Emit) error
}

func (g *scriptGen) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.mu.Lock()
	script := g.scripts[0]
	g.scripts = g.scripts[1:]
	g.mu.Unlock()
	return script(ctx, g, emit)
}

// stamped is a frame reaching the end of the pipeline, with when it did.
type stamped struct {
	at time.Time
	f  frames.Frame
}

// runScripted runs one generation per script through an LLM service with
// metrics on, and returns every frame that reached the end, once the last
// generation has ended.
func runScripted(
	t *testing.T, onlyInitialTTFB bool, scripts ...func(ctx context.Context, g *scriptGen, emit llm.Emit) error,
) []stamped {
	t.Helper()
	gen := &scriptGen{scripts: scripts}
	svc := llm.New("FakeLLM", gen)
	gen.Base = svc

	noInitial := false
	var mu sync.Mutex
	var got []stamped
	ended := make(chan struct{}, len(scripts))
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			ReportOnlyInitialTTFB:   onlyInitialTTFB,
			SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, stamped{time.Now(), f})
		mu.Unlock()
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			ended <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-runDone
	}()

	for range scripts {
		convo := frames.NewLLMContext("sys")
		convo.AddUserMessage("hi")
		task.QueueFrame(frames.NewLLMContextFrame(convo))
		select {
		case <-ended:
		case <-time.After(3 * time.Second):
			t.Fatal("the response never ended")
		}
	}
	// The processing time follows the response's end.
	time.Sleep(20 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	return append([]stamped(nil), got...)
}

// ttfatsOf returns the TTFAT measurements among frames, with when each arrived,
// and fails if one shares its frame with the processing time.
func ttfatsOf(t *testing.T, got []stamped) ([]frames.TTFATMetricsData, []time.Time) {
	t.Helper()
	var out []frames.TTFATMetricsData
	var at []time.Time
	for _, s := range got {
		mf, ok := s.f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		hasProcessing := false
		for _, d := range mf.Data {
			switch d := d.(type) {
			case frames.TTFATMetricsData:
				out = append(out, d)
				at = append(at, s.at)
			case frames.ProcessingMetricsData:
				hasProcessing = true
			}
		}
		if hasProcessing && len(mf.Data) > 1 {
			t.Errorf("the processing time shares its frame with %v, want each measurement in its own", mf.Data)
		}
	}
	return out, at
}

const streamGap = 200 * time.Millisecond

// Ported from upstream (test_streamed_response_pushes_a_ttfat_frame). The
// TTFAT goes out in a frame of its own when the answer token is produced, not
// with the processing time at the end of the response.
func TestTTFATIsPushedWhenTheAnswerTokenArrives(t *testing.T) {
	got := runScripted(t, false, func(ctx context.Context, g *scriptGen, emit llm.Emit) error {
		g.StartTTFBMetrics()
		g.StopTTFBMetrics(ctx)
		if err := emit("Hello"); err != nil {
			return err
		}
		time.Sleep(streamGap)
		return emit(" there.")
	})
	ttfat, at := ttfatsOf(t, got)
	if len(ttfat) != 1 {
		t.Fatalf("got %d TTFAT, want one for the response", len(ttfat))
	}
	var text time.Time
	for _, s := range got {
		if _, ok := s.f.(*frames.LLMTextFrame); ok {
			text = s.at
			break
		}
	}
	if lag := at[0].Sub(text); lag > streamGap/2 {
		t.Errorf("the TTFAT arrived %s after the first answer token, want it with the token", lag)
	}
	if ttfat[0].TTFAT < ttfat[0].TTFB || ttfat[0].ThinkingTime != ttfat[0].TTFAT-ttfat[0].TTFB {
		t.Errorf("ttfat %+v: want ttfat >= ttfb and the thinking time between them", ttfat[0])
	}
}

// Ported from upstream. Later tokens of the same response do not report again.
func TestTTFATReportedOncePerResponse(t *testing.T) {
	got := runScripted(t, false, func(ctx context.Context, g *scriptGen, emit llm.Emit) error {
		g.StartTTFBMetrics()
		g.StopTTFBMetrics(ctx)
		if err := emit("One"); err != nil {
			return err
		}
		return emit(" two")
	})
	if ttfat, _ := ttfatsOf(t, got); len(ttfat) != 1 {
		t.Errorf("got %d TTFAT, want one", len(ttfat))
	}
}

// Ported from upstream. Nothing arms TTFAT if the response produced no output
// before its answer: no TTFB was measured.
func TestNoTTFATWithoutTTFB(t *testing.T) {
	got := runScripted(t, false, func(_ context.Context, g *scriptGen, emit llm.Emit) error {
		g.StartTTFBMetrics()
		return emit("Hello")
	})
	if ttfat, _ := ttfatsOf(t, got); len(ttfat) != 0 {
		t.Errorf("got %+v, want no TTFAT without a TTFB", ttfat)
	}
}

// Ported from upstream. An answer token arriving with the first output reports
// no thinking time.
func TestAnswerTokenWithFirstOutputReportsNoThinking(t *testing.T) {
	got := runScripted(t, false, func(ctx context.Context, g *scriptGen, emit llm.Emit) error {
		g.StartTTFBMetrics()
		time.Sleep(20 * time.Millisecond)
		g.StopTTFBMetrics(ctx)
		return emit("Hello")
	})
	ttfat, _ := ttfatsOf(t, got)
	if len(ttfat) != 1 {
		t.Fatalf("got %d TTFAT, want one", len(ttfat))
	}
	if ttfat[0].ThinkingTime > 5*time.Millisecond {
		t.Errorf("thinking time = %s, want about none", ttfat[0].ThinkingTime)
	}
}

// Ported from upstream. A response cut off before its answer token reports
// nothing, and the next is measured from its own request.
func TestInterruptedResponseDoesNotLeakTTFATIntoTheNext(t *testing.T) {
	const gap = 150 * time.Millisecond
	got := runScripted(t, false,
		func(ctx context.Context, g *scriptGen, _ llm.Emit) error {
			g.StartTTFBMetrics()
			g.StopTTFBMetrics(ctx)
			time.Sleep(gap) // No answer token arrives.
			return nil
		},
		func(ctx context.Context, g *scriptGen, emit llm.Emit) error {
			g.StartTTFBMetrics()
			g.StopTTFBMetrics(ctx)
			return emit("Hello")
		},
	)
	ttfat, _ := ttfatsOf(t, got)
	if len(ttfat) != 1 {
		t.Fatalf("got %d TTFAT, want one, for the second response", len(ttfat))
	}
	if ttfat[0].TTFAT >= gap {
		t.Errorf("ttfat = %s, want it measured from the second request, not the first", ttfat[0].TTFAT)
	}
}

// Ported from upstream. Reporting only the initial TTFB, a later request
// measures nothing, so it has no request start to measure TTFAT from either.
func TestInterruptedResponseDoesNotLeakTTFATWhenReportingOnlyInitialTTFB(t *testing.T) {
	got := runScripted(t, true,
		func(ctx context.Context, g *scriptGen, _ llm.Emit) error {
			g.StartTTFBMetrics()
			g.StopTTFBMetrics(ctx)
			return nil
		},
		func(ctx context.Context, g *scriptGen, emit llm.Emit) error {
			g.StartTTFBMetrics()
			g.StopTTFBMetrics(ctx)
			return emit("Hello")
		},
	)
	if ttfat, _ := ttfatsOf(t, got); len(ttfat) != 0 {
		t.Errorf("got %+v, want no TTFAT", ttfat)
	}
}
