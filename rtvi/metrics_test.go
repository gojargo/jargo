package rtvi_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/rtvi"
)

func TestMetricsFrameBecomesMetricsMessage(t *testing.T) {
	out := make(chan rtvi.Message, 8)
	task := pipeline.NewTask(pipeline.New(rtvi.NewProcessor()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if m, ok := f.(*frames.OutputTransportMessageFrame); ok {
				if msg, ok := m.Message.(rtvi.Message); ok {
					select {
					case out <- msg:
					default:
					}
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	mf := frames.NewMetricsFrame("AnthropicLLM#1")
	mf.Model = "claude-haiku"
	ttfb := 300 * time.Millisecond
	proc := 1200 * time.Millisecond
	mf.TTFB = &ttfb
	mf.Processing = &proc
	mf.Tokens = &frames.LLMTokenUsage{PromptTokens: 100, CompletionTokens: 40, TotalTokens: 140}
	task.QueueFrame(mf)

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

	task.StopWhenDone()
	<-runDone
}

func approx(got, want float64) bool {
	d := got - want
	return d < 0.001 && d > -0.001
}
