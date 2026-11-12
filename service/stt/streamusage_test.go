package stt_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// handedStream delivers the results a test hands it, when it hands them.
type handedStream struct {
	ctx     context.Context
	results chan []stt.Result
}

func (s *handedStream) Send([]byte) error { return nil }

func (s *handedStream) Recv() ([]stt.Result, error) {
	select {
	case r := <-s.results:
		return r, nil
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

func (s *handedStream) Close() error { return nil }

type handedConnector struct{ stream *handedStream }

func (c *handedConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// usageRun is a streaming STT service in a pipeline, recording what reaches the
// end of it: each usage report as its seconds, each transcript as its text.
type usageRun struct {
	task    *pipeline.Worker
	results chan []stt.Result
	runDone chan error

	mu     sync.Mutex
	events []any
	audio  int
}

func startUsageRun(t *testing.T, usage bool) *usageRun {
	t.Helper()
	return startUsageRunTraced(t, usage, false)
}

func startUsageRunTraced(t *testing.T, usage, tracing bool) *usageRun {
	t.Helper()
	r := &usageRun{results: make(chan []stt.Result, 4), runDone: make(chan error, 1)}
	svc := stt.NewStream("FakeSTT", &handedConnector{stream: &handedStream{results: r.results}}, 16000)
	r.task = pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		EnableTracing:           tracing,
		Params:                  pipeline.Params{EnableUsageMetrics: usage},
	})
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		defer r.mu.Unlock()
		switch f := f.(type) {
		case *frames.MetricsFrame:
			for _, d := range f.Data {
				if u, ok := d.(frames.STTUsageMetricsData); ok {
					r.events = append(r.events, u.Value.AudioSeconds)
				}
			}
		case *frames.TranscriptionFrame:
			r.events = append(r.events, f.Text)
		case *frames.InputAudioRawFrame:
			r.audio++
		}
	})
	go func() { r.runDone <- r.task.Run(context.Background()) }()
	return r
}

// sendAudio queues seconds of audio and waits for it to have gone through.
func (r *usageRun) sendAudio(t *testing.T, seconds float64) {
	t.Helper()
	r.mu.Lock()
	want := r.audio + 1
	r.mu.Unlock()
	r.task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, int(16000*2*seconds)), 16000, 1))
	r.waitFor(t, "the audio", func() bool { return r.audio >= want })
}

// transcribe has the provider deliver a final transcript and waits for it.
func (r *usageRun) transcribe(t *testing.T, text string) {
	t.Helper()
	r.results <- []stt.Result{{Text: text, Final: true, EndOfTurn: true}}
	r.waitFor(t, "the transcript "+text, func() bool {
		for _, e := range r.events {
			if e == text {
				return true
			}
		}
		return false
	})
}

func (r *usageRun) waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		ok := cond()
		r.mu.Unlock()
		if ok {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// stop ends the run and returns what reached the end, in order.
func (r *usageRun) stop() []any {
	r.task.StopWhenDone()
	<-r.runDone
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]any(nil), r.events...)
}

func sameEvents(got, want []any) bool {
	return slices.Equal(got, want)
}

// Ported from upstream (test_emit_reports_accumulated_audio_and_resets). Each
// final transcript is a billing event: the audio since the last one is reported
// ahead of it, so the span it closes carries it, and what was reported is not
// reported again.
func TestStreamServiceReportsUsageAheadOfEachFinalTranscript(t *testing.T) {
	r := startUsageRun(t, true)
	r.sendAudio(t, 1)
	r.sendAudio(t, 0.5)
	r.transcribe(t, "one")
	r.sendAudio(t, 0.5)
	r.transcribe(t, "two")
	if got, want := r.stop(), []any{1.5, "one", 0.5, "two"}; !sameEvents(got, want) {
		t.Errorf("reached the end %v, want %v", got, want)
	}
}

// Ported from upstream (test_stop_flushes_trailing_audio_usage). The audio no
// final transcript reported is reported when the pipeline stops.
func TestStreamServiceStopFlushesTrailingUsage(t *testing.T) {
	r := startUsageRun(t, true)
	r.sendAudio(t, 1)
	r.transcribe(t, "one")
	r.sendAudio(t, 0.5)
	if got, want := r.stop(), []any{1.0, "one", 0.5}; !sameEvents(got, want) {
		t.Errorf("reached the end %v, want %v", got, want)
	}
}

// Ported from upstream (test_usage_metrics_disabled_by_default and
// test_emit_gated_off_pushes_nothing_but_resets_pending). With usage metrics
// off, nothing is reported.
func TestStreamServiceUsageMetricsOffReportsNothing(t *testing.T) {
	r := startUsageRun(t, false)
	r.sendAudio(t, 1)
	r.transcribe(t, "one")
	r.sendAudio(t, 0.5)
	if got, want := r.stop(), []any{"one"}; !sameEvents(got, want) {
		t.Errorf("reached the end %v, want %v", got, want)
	}
}

// Ported from upstream (test_traced_stt_accumulates_pending_usage_for_span). The
// usage reported ahead of a final transcript lands on the transcription span
// that transcript closes.
func TestStreamServiceSpanCarriesItsUsage(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	r := startUsageRunTraced(t, false, true)
	r.sendAudio(t, 1)
	r.sendAudio(t, 0.5)
	r.transcribe(t, "one")
	r.stop()

	var span sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		if s.Name() == "stt" {
			span = s
		}
	}
	if span == nil {
		t.Fatal("no stt span recorded")
	}
	for _, kv := range span.Attributes() {
		if kv.Key == "metrics.audio_seconds" {
			if got := kv.Value.AsFloat64(); got != 1.5 {
				t.Errorf("metrics.audio_seconds = %v, want 1.5", got)
			}
			return
		}
	}
	t.Error("the span carries no metrics.audio_seconds")
}
