package stt_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/stt"
)

// collectSTTUsage runs svc under a task, feeds it a turn of audio, and returns
// the audio usage it reported in band.
func collectSTTUsage(t *testing.T, svc processor.Processor, usageMetrics bool) []frames.STTUsageMetricsData {
	t.Helper()

	var mu sync.Mutex
	var got []frames.STTUsageMetricsData
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableUsageMetrics: usageMetrics,
		OnReachedDownstream: func(f frames.Frame) {
			switch fr := f.(type) {
			case *frames.MetricsFrame:
				mu.Lock()
				for _, d := range fr.Data {
					if u, ok := d.(frames.STTUsageMetricsData); ok {
						got = append(got, u)
					}
				}
				mu.Unlock()
			case *frames.TranscriptionFrame:
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 32000), 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("service never transcribed")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return append([]frames.STTUsageMetricsData(nil), got...)
}

func TestSegmentServiceReportsUsageInBand(t *testing.T) {
	tr := &fakeTranscriber{text: "buffered words", got: make(chan []byte, 1)}
	svc := stt.NewSegment("FakeSTT", tr, 16000)
	go func() {
		for range tr.got {
		}
	}()

	got := collectSTTUsage(t, svc, true)
	if len(got) != 1 {
		t.Fatalf("got %d STT usage reports, want 1", len(got))
	}
	// 32000 bytes of 16-bit mono at 16 kHz is one second of audio.
	if got[0].Value.AudioSeconds != 1 {
		t.Errorf("audio seconds = %v, want 1", got[0].Value.AudioSeconds)
	}
	if got[0].Processor != svc.Name() {
		t.Errorf("processor = %q, want %q", got[0].Processor, svc.Name())
	}
}

func TestSegmentServiceReportsNoUsageWhenDisabled(t *testing.T) {
	tr := &fakeTranscriber{text: "buffered words", got: make(chan []byte, 1)}
	svc := stt.NewSegment("FakeSTT", tr, 16000)
	go func() {
		for range tr.got {
		}
	}()

	if got := collectSTTUsage(t, svc, false); len(got) != 0 {
		t.Errorf("got %d STT usage reports with usage metrics off, want none", len(got))
	}
}
