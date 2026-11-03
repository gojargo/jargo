package stt_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/telemetry/tracing"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// (Not a port of an upstream test: these cover the push-driven segment spans
// upstream installs on a service's push_frame with its traced_stt decorator.)

// pusher is a processor whose pushes are what TraceTranscripts follows.
type pusher struct{ *processor.Base }

func (p *pusher) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	return p.Base.ProcessFrame(ctx, f, dir)
}

// tracedPusher is a traced processor following its pushes, and the recorder its
// spans end in.
func tracedPusher(t *testing.T) (*pusher, *tracetest.SpanRecorder) {
	t.Helper()
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	p := &pusher{}
	p.Base = processor.New("Realtime", p, processor.WithDirectMode())
	ctx := context.Background()
	if err := p.Setup(ctx, processor.Setup{
		Clock: clock.NewSystem(), TracingEnabled: true, Tracing: tracing.NewTracingContext(),
	}); err != nil {
		t.Fatalf("setup: %v", err)
	}
	stt.TraceTranscripts(p.Base, func() tracing.STTAttributes {
		return tracing.STTAttributes{Service: "OpenAIRealtimeLLMService", Model: "gpt-realtime"}
	})
	return p, rec
}

func (p *pusher) push(t *testing.T, f frames.Frame) {
	t.Helper()
	if err := p.PushFrame(context.Background(), f, processor.Upstream); err != nil {
		t.Fatalf("push: %v", err)
	}
}

// sttSpans are the stt spans that ended, with their attributes.
func sttSpans(rec *tracetest.SpanRecorder) []map[string]string {
	var out []map[string]string
	for _, s := range rec.Ended() {
		if s.Name() != "stt" {
			continue
		}
		attrs := map[string]string{}
		for _, kv := range s.Attributes() {
			attrs[string(kv.Key)] = kv.Value.String()
		}
		out = append(out, attrs)
	}
	return out
}

// A transcript the service pushes opens a span, and the end of the user's turn
// closes it as incomplete when no transcript finalized it.
func TestTraceTranscriptsClosesOnUserStopped(t *testing.T) {
	p, rec := tracedPusher(t)
	p.push(t, frames.NewTranscriptionFrame("hello", "", time.Now().Format(time.RFC3339)))
	p.push(t, frames.NewTranscriptionFrame("hello there", "", time.Now().Format(time.RFC3339)))
	if got := sttSpans(rec); len(got) != 0 {
		t.Fatalf("spans = %v, want the segment still open", got)
	}
	p.push(t, frames.NewUserStoppedSpeakingFrame())
	spans := sttSpans(rec)
	if len(spans) != 1 {
		t.Fatalf("spans = %v, want one", spans)
	}
	want := map[string]string{
		"gen_ai.request.model": "gpt-realtime", "transcript": "hello there",
		"is_final": "false", "stt.incomplete": "true", "vad_enabled": "false",
	}
	for k, v := range want {
		if spans[0][k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, spans[0][k], v, spans[0])
		}
	}
}

// A finalized transcript closes its span, and a time to first byte reported
// while it is open is recorded on it.
func TestTraceTranscriptsFinalizedClosesAndRecordsTTFB(t *testing.T) {
	p, rec := tracedPusher(t)
	tf := frames.NewTranscriptionFrame("hi", "", time.Now().Format(time.RFC3339))
	p.push(t, frames.NewTranscriptionFrame("h", "", time.Now().Format(time.RFC3339)))
	var ttfb frames.TTFBMetricsData
	ttfb.Value = 250 * time.Millisecond
	p.push(t, frames.NewMetricsFrame(ttfb))
	tf.Finalized = true
	p.push(t, tf)
	spans := sttSpans(rec)
	if len(spans) != 1 || spans[0]["metrics.ttfb"] != "0.25" || spans[0]["is_final"] != "true" {
		t.Errorf("spans = %v, want one finalized span with the time to first byte", spans)
	}
}

// Speech the VAD reports anchors the span where the speech began.
func TestTraceTranscriptsAnchorsAtSpeechStart(t *testing.T) {
	p, rec := tracedPusher(t)
	speechStart := time.Now().Add(-time.Second)
	p.push(t, frames.NewVADUserStartedSpeakingFrame(0.2, speechStart.Add(200*time.Millisecond)))
	tf := frames.NewTranscriptionFrame("hi", "", time.Now().Format(time.RFC3339))
	tf.Finalized = true
	p.push(t, tf)
	ended := rec.Ended()
	if len(ended) != 1 {
		t.Fatalf("spans = %d, want one", len(ended))
	}
	if got := ended[0].StartTime(); got.Sub(speechStart).Abs() > 10*time.Millisecond {
		t.Errorf("span starts at %v, want the speech start %v", got, speechStart)
	}
}
