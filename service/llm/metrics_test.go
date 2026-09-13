package llm_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
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
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
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
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
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

// thinkingGen streams something before it starts answering, the way a reasoning
// model does: the first byte arrives well before the first answer token.
type thinkingGen struct {
	*llm.Base
}

func (g *thinkingGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.StartTTFBMetrics()
	// The model's first output, which is not yet the answer.
	g.StopTTFBMetrics()
	time.Sleep(20 * time.Millisecond)
	return emit("hello")
}

// timingMetrics runs one generation and returns the measurements it reported.
func timingMetrics(t *testing.T, svc *llm.Base) []frames.MetricsData {
	t.Helper()

	noInitial := false
	mfCh := make(chan *frames.MetricsFrame, 4)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
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
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-runDone
	}()

	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case mf := <-mfCh:
		return mf.Data
	case <-time.After(3 * time.Second):
		t.Fatal("no timing MetricsFrame emitted")
		return nil
	}
}

// TestReportsTimeToFirstAnswerToken covers the measurement that separates the
// model thinking from the model answering: the first byte is not the first
// answer token, and the gap between them is what a reasoning model spends.
func TestReportsTimeToFirstAnswerToken(t *testing.T) {
	gen := &thinkingGen{}
	svc := llm.New("FakeLLM", gen)
	gen.Base = svc

	var got frames.TTFATMetricsData
	var found bool
	for _, d := range timingMetrics(t, svc) {
		if m, ok := d.(frames.TTFATMetricsData); ok {
			got, found = m, true
		}
	}
	if !found {
		t.Fatal("no time to first answer token was reported")
	}
	if got.TTFAT <= got.TTFB {
		t.Errorf("ttfat %s is not past the ttfb %s it builds on", got.TTFAT, got.TTFB)
	}
	if got.ThinkingTime != got.TTFAT-got.TTFB {
		t.Errorf("thinking time = %s, want the %s between the two", got.ThinkingTime, got.TTFAT-got.TTFB)
	}
	if got.ThinkingTime < 10*time.Millisecond {
		t.Errorf("thinking time = %s, want at least the 20ms the model spent", got.ThinkingTime)
	}
}

// realtimeGen answers in audio, as a speech-to-speech service does, and says so
// in its metadata.
type realtimeGen struct {
	*llm.Base
}

func (g *realtimeGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.StartTTFBMetrics()
	g.StopTTFBMetrics()
	time.Sleep(20 * time.Millisecond)
	return emit("hello")
}

func (g *realtimeGen) ServiceMetadataFrame() frames.ServiceMetadata {
	f := frames.NewLLMServiceMetadataFrame(g.Name())
	f.Realtime = true
	return f
}

// TestARealtimeServiceReportsNoAnswerToken covers the one kind of service the
// measurement does not apply to: it answers in audio, which has no answer token
// to measure to.
func TestARealtimeServiceReportsNoAnswerToken(t *testing.T) {
	gen := &realtimeGen{}
	svc := llm.New("FakeRealtime", gen)
	gen.Base = svc

	for _, d := range timingMetrics(t, svc) {
		if _, ok := d.(frames.TTFATMetricsData); ok {
			t.Error("a realtime service reported a time to first answer token, want none")
		}
	}
}
