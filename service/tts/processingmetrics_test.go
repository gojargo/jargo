package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// handOffSynth stands for a provider holding a connection of its own: RunTTS
// sends the text and returns, and the audio arrives on its receive loop.
type handOffSynth struct {
	host tts.AudioContextHost
}

func (s *handOffSynth) SampleRate() int                            { return skipSampleRate }
func (s *handOffSynth) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

func (s *handOffSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	go func() {
		s.host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(make([]byte, 640), skipSampleRate, 1))
		s.host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
	}()
	return nil
}

// overriddenSynth hands off like handOffSynth but says its RunTTS waits for
// the synthesis to finish.
type overriddenSynth struct{ handOffSynth }

func (*overriddenSynth) SupportsProcessingMetrics() bool { return true }

// processingMetrics speaks text through a TTS service over syn, with metrics
// on, and counts the processing metrics it pushes.
func processingMetrics(t *testing.T, syn tts.Synthesizer, text string) int {
	t.Helper()
	var mu sync.Mutex
	n := 0
	noInitial := false
	task := pipeline.NewWorker(pipeline.New(tts.New("MetricsTTS", syn)), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{EnableMetrics: true, SendInitialEmptyMetrics: &noInitial},
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if mf, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range mf.Data {
				if _, ok := d.(frames.ProcessingMetricsData); ok {
					mu.Lock()
					n++
					mu.Unlock()
				}
			}
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrames([]frames.Frame{
		frames.NewLLMFullResponseStartFrame(), frames.NewLLMTextFrame(text), frames.NewLLMFullResponseEndFrame(),
	})
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return n
}

// TestSynchronousServicesReportProcessingPerSentence checks a provider whose
// RunTTS synthesizes before it returns reports a processing time for each
// sentence it speaks.
func TestSynchronousServicesReportProcessingPerSentence(t *testing.T) {
	if n := processingMetrics(t, &recordingSynth{}, "One. Two."); n != 2 {
		t.Errorf("processing metrics = %d, want one per sentence", n)
	}
}

// TestHandOffServicesDoNotReportProcessing checks a provider that hands the
// text off reports no processing time, since the measurement would cover the
// handoff and nothing else, unless it says its RunTTS waits for the synthesis.
func TestHandOffServicesDoNotReportProcessing(t *testing.T) {
	if n := processingMetrics(t, &handOffSynth{}, "One. Two."); n != 0 {
		t.Errorf("processing metrics = %d, want none for a provider that hands off", n)
	}
	if n := processingMetrics(t, &overriddenSynth{}, "One. Two."); n != 2 {
		t.Errorf("processing metrics = %d, want one per sentence once the provider says so", n)
	}
}
