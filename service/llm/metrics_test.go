package llm_test

import (
	"context"
	"slices"
	"sync"
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

func (g *timingGen) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.StartTTFBMetrics()
	g.StopTTFBMetrics(ctx)
	return emit("hello")
}

// TestEmitsTimingMetricsWhenEnabled checks the time to first byte is reported
// on its own as the first output arrives, ahead of the reply, and the processing
// time when the generation ends, without the first byte again.
func TestEmitsTimingMetricsWhenEnabled(t *testing.T) {
	gen := &timingGen{}
	svc := llm.New("FakeLLM", gen)
	gen.Base = svc
	svc.SetModel("m1")

	// The zeroed frames the task sends when the pipeline is ready would arrive
	// first and are not what this is measuring.
	noInitial := false
	var mu sync.Mutex
	var order []string
	done := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           true,
			SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.LLMTextFrame:
			order = append(order, "text")
		case *frames.MetricsFrame:
			for _, d := range fr.Data {
				switch m := d.(type) {
				case frames.TTFBMetricsData:
					if m.Model != "m1" {
						t.Errorf("model = %q, want m1", m.Model)
					}
					order = append(order, "ttfb")
				case frames.ProcessingMetricsData:
					order = append(order, "processing")
					select {
					case done <- struct{}{}:
					default:
					}
				}
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("no timing MetricsFrame emitted")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	want := []string{"ttfb", "text", "processing"}
	if !slices.Equal(order, want) {
		t.Errorf("frames in the order %v, want %v: the first byte reported as it arrives, once", order, want)
	}
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

func (g *thinkingGen) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.StartTTFBMetrics()
	// The model's first output, which is not yet the answer.
	g.StopTTFBMetrics(ctx)
	time.Sleep(20 * time.Millisecond)
	return emit("hello")
}

// timingMetrics runs one generation and returns the measurements it reported,
// from every metrics frame up to the one with the processing time, which closes
// the generation. Each measurement goes out in a frame of its own as it is
// taken, so they are gathered across frames.
func timingMetrics(t *testing.T, svc *llm.Base) []frames.MetricsData {
	t.Helper()

	noInitial := false
	var mu sync.Mutex
	var data []frames.MetricsData
	done := make(chan struct{}, 1)
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
		mu.Lock()
		data = append(data, mf.Data...)
		mu.Unlock()
		for _, d := range mf.Data {
			if _, ok := d.(frames.ProcessingMetricsData); ok {
				select {
				case done <- struct{}{}:
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
	case <-done:
		mu.Lock()
		defer mu.Unlock()
		return append([]frames.MetricsData(nil), data...)
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

func (g *realtimeGen) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.StartTTFBMetrics()
	g.StopTTFBMetrics(ctx)
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

// speculativeGen answers whatever it is asked, so a test can drive one
// inference through the service and watch what escapes it.
type speculativeGen struct {
	*llm.Base
}

func (g *speculativeGen) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	return emit("Booking your flight.")
}

// runSpeculation drives one context frame through a service and returns the
// frames that reached the pipeline, after sending whatever follow-up frames the
// test supplies.
func runSpeculation(t *testing.T, speculation bool, after ...frames.Frame) []frames.Frame {
	t.Helper()

	gen := &speculativeGen{}
	svc := llm.New("FakeLLM", gen)
	gen.Base = svc

	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("book me a flight")
	cf := frames.NewLLMContextFrame(convo)
	cf.Speculation = speculation
	task.QueueFrame(cf)
	for _, f := range after {
		task.QueueFrame(f)
	}

	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), got...)
}

// spokeText reports whether any model text escaped the service.
func spokeText(fs []frames.Frame) bool {
	for _, f := range fs {
		if _, ok := f.(*frames.LLMTextFrame); ok {
			return true
		}
	}
	return false
}

// TestASpeculativeReplyIsHeldByTheService covers the service gating what it
// generates: a reply to a turn that may not have ended reaches nobody until the
// turn is confirmed.
func TestASpeculativeReplyIsHeldByTheService(t *testing.T) {
	if got := runSpeculation(t, true); spokeText(got) {
		t.Error("the speculative reply escaped the service with the turn still open")
	}
}

// TestAnOrdinaryReplyIsNotHeld covers the other side: an inference answering a
// turn that has ended is not gated at all.
func TestAnOrdinaryReplyIsNotHeld(t *testing.T) {
	if got := runSpeculation(t, false); !spokeText(got) {
		t.Error("an ordinary reply was held, want it pushed straight through")
	}
}

// TestAConfirmedTurnReleasesTheSpeculativeReply covers the turn ending letting
// the held reply out.
func TestAConfirmedTurnReleasesTheSpeculativeReply(t *testing.T) {
	got := runSpeculation(t, true, frames.NewUserStoppedSpeakingFrame())
	if !spokeText(got) {
		t.Error("the confirmed reply never escaped the service")
	}
}

// TestAWithdrawnTurnDiscardsTheSpeculativeReply covers the withdrawal: the reply
// answered a turn the user had not finished, so nobody ever hears it.
func TestAWithdrawnTurnDiscardsTheSpeculativeReply(t *testing.T) {
	got := runSpeculation(t, true, frames.NewUserTurnResponseCancelFrame())
	if spokeText(got) {
		t.Error("the withdrawn reply escaped the service")
	}
}
