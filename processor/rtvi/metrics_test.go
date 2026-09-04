package rtvi_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

func TestMetricsFrameBecomesMetricsMessage(t *testing.T) {
	out := make(chan rtvi.Message, 8)
	rtviProc := rtvi.NewProcessor()
	task := pipeline.NewWorker(pipeline.New(rtviProc), pipeline.WorkerConfig{
		// Events are reported by the observer; the processor only carries them.
		Observers:               []pipeline.Observer{rtvi.NewObserver(rtviProc)},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				select {
				case out <- msg:
				default:
				}
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	llm := frames.BaseMetricsData{Processor: "AnthropicLLM#1", Model: "claude-haiku"}
	tts := frames.BaseMetricsData{Processor: "CartesiaTTS#2", Model: "sonic"}
	task.QueueFrame(frames.NewMetricsFrame(
		frames.TTFBMetricsData{BaseMetricsData: llm, Value: 300 * time.Millisecond},
		frames.ProcessingMetricsData{BaseMetricsData: llm, Value: 1200 * time.Millisecond},
		frames.LLMUsageMetricsData{
			BaseMetricsData: llm,
			Value:           frames.LLMTokenUsage{PromptTokens: 100, CompletionTokens: 40, TotalTokens: 140},
		},
		// A second processor in the same frame, which is what the list shape is
		// for: every kind reaches the client as its own list.
		frames.TTFAMetricsData{
			BaseMetricsData: tts,
			TTFA:            500 * time.Millisecond,
			TTFB:            400 * time.Millisecond,
			LeadingSilence:  100 * time.Millisecond,
		},
		frames.STTUsageMetricsData{
			Processor: "DeepgramSTT#3",
			Value:     frames.STTUsage{AudioSeconds: 2.5},
		},
	))

	got := waitMessage(t, out)
	if got.Type != rtvi.TypeMetrics {
		t.Fatalf("type = %q, want metrics", got.Type)
	}
	d, ok := got.Data.(rtvi.MetricsData)
	if !ok {
		t.Fatalf("data type = %T, want rtvi.MetricsData", got.Data)
	}
	if len(d.TTFB) != 1 || d.TTFB[0].Processor != "AnthropicLLM#1" || !approx(d.TTFB[0].Value, 0.3) {
		t.Fatalf("ttfb = %+v, want one ~0.3s entry", d.TTFB)
	}
	if len(d.Processing) != 1 || !approx(d.Processing[0].Value, 1.2) {
		t.Fatalf("processing = %+v, want one ~1.2s entry", d.Processing)
	}
	if len(d.Tokens) != 1 || d.Tokens[0].TotalTokens != 140 || d.Tokens[0].Model != "claude-haiku" {
		t.Fatalf("tokens = %+v, want one entry totalling 140", d.Tokens)
	}
	if len(d.TTFA) != 1 || d.TTFA[0].Processor != "CartesiaTTS#2" || !approx(d.TTFA[0].LeadingSilence, 0.1) {
		t.Fatalf("ttfa = %+v, want one entry with ~0.1s of leading silence", d.TTFA)
	}
	if len(d.STTUsage) != 1 || !approx(d.STTUsage[0].Value, 2.5) {
		t.Fatalf("stt_usage = %+v, want one ~2.5s entry", d.STTUsage)
	}

	task.StopWhenDone()
	<-runDone
}

func approx(got, want float64) bool {
	d := got - want
	return d < 0.001 && d > -0.001
}
