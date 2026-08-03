package llm_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// timingGen mirrors how a real service reports timing: it records time to first
// byte itself, around the point its response stream opens.
type timingGen struct {
	*llm.Base
}

func (g *timingGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.StartTTFBMetrics()
	g.StopTTFBMetrics()
	return emit("hello")
}

func TestEmitsTimingMetricsWhenEnabled(t *testing.T) {
	gen := &timingGen{}
	svc := llm.New("FakeLLM", gen)
	gen.Base = svc
	svc.SetModel("m1")

	// The zeroed frames the task sends when the pipeline is ready would arrive
	// first and are not what this is measuring.
	noInitial := false
	mfCh := make(chan *frames.MetricsFrame, 4)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableMetrics:           true,
		SendInitialEmptyMetrics: &noInitial,
		OnReachedDownstream: func(f frames.Frame) {
			mf, ok := f.(*frames.MetricsFrame)
			if !ok {
				return
			}
			for _, d := range mf.Data {
				if _, ok := d.(frames.ProcessingMetricsData); ok {
					select {
					case mfCh <- mf:
					default:
					}
					return
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case mf := <-mfCh:
		var sawTTFB bool
		for _, d := range mf.Data {
			if ttfb, ok := d.(frames.TTFBMetricsData); ok {
				sawTTFB = true
				if ttfb.Model != "m1" {
					t.Fatalf("model = %q, want m1", ttfb.Model)
				}
			}
		}
		if !sawTTFB {
			t.Fatal("TTFB not reported on the timing MetricsFrame")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no timing MetricsFrame emitted")
	}

	task.StopWhenDone()
	<-runDone
}

func TestNoMetricsFrameWhenDisabled(t *testing.T) {
	gen := &fakeGen{deltas: []string{"hello"}}
	svc := llm.New("FakeLLM", gen)

	seen := make(chan struct{}, 1)
	end := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			switch f.(type) {
			case *frames.MetricsFrame:
				select {
				case seen <- struct{}{}:
				default:
				}
			case *frames.LLMFullResponseEndFrame:
				select {
				case end <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	<-end
	task.StopWhenDone()
	<-runDone

	select {
	case <-seen:
		t.Fatal("MetricsFrame emitted though metrics were disabled")
	default:
	}
}
