package llm_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

func TestEmitsTimingMetricsWhenEnabled(t *testing.T) {
	gen := &fakeGen{deltas: []string{"hello"}}
	svc := llm.New("FakeLLM", gen)
	svc.SetModel("m1")

	mfCh := make(chan *frames.MetricsFrame, 4)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableMetrics: true,
		OnReachedDownstream: func(f frames.Frame) {
			if mf, ok := f.(*frames.MetricsFrame); ok && mf.Processing != nil {
				select {
				case mfCh <- mf:
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

	select {
	case mf := <-mfCh:
		if mf.TTFB == nil {
			t.Fatal("TTFB not set on the timing MetricsFrame")
		}
		if mf.Model != "m1" {
			t.Fatalf("model = %q, want m1", mf.Model)
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
