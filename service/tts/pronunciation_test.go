package tts_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

// pronouncingSynth records the text it is asked to speak; pronunciations are
// wrapped in <>.
type pronouncingSynth struct {
	spacedSynth
	supported atomic.Bool
}

func (s *pronouncingSynth) FormatPronunciation(_, ipa string) (string, bool) {
	return "<" + ipa + ">", true
}

func (s *pronouncingSynth) SupportsPronunciations() bool { return s.supported.Load() }

// speak runs each text through a service with an upper-casing transform ahead
// of the pronunciation one, and returns what the provider was given.
func speak(t *testing.T, supported bool, texts ...string) []string {
	t.Helper()
	syn := &pronouncingSynth{}
	syn.supported.Store(supported)
	base := tts.New("PronounceTTS", syn)
	upper := tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			return strings.ToUpper(text), nil
		},
	}
	base.SetTextTransformers(upper, base.PronunciationTransformIPA(map[string]string{"crete": "kriːt"}))

	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, text := range texts {
		task.QueueFrame(frames.NewTTSSpeakFrame(text))
		time.Sleep(200 * time.Millisecond)
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
	out := syn.texts()
	for i := range out {
		out[i] = strings.TrimSpace(out[i])
	}
	return out
}

func TestPronunciationsAreAppliedWhenTheProviderReadsThem(t *testing.T) {
	if got := speak(t, true, "Visit Crete"); len(got) != 1 || got[0] != "VISIT <kriːt>" {
		t.Fatalf("spoken = %q, want VISIT <kriːt>", got)
	}
}

// Other transforms still run, and the warning is logged once.
func TestPronunciationsAreSkippedWhenTheProviderCannotReadThem(t *testing.T) {
	var logs bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	got := speak(t, false, "Visit Crete", "Crete again")
	if len(got) != 2 || got[0] != "VISIT CRETE" || got[1] != "CRETE AGAIN" {
		t.Fatalf("spoken = %q, want the words as written", got)
	}
	if n := strings.Count(logs.String(), "pronunciation markup"); n != 1 {
		t.Fatalf("warned %d times, want once:\n%s", n, logs.String())
	}
}

// A service whose provider takes no pronunciations speaks every word as written.
func TestAServiceWithoutAFormatterPronouncesNothing(t *testing.T) {
	base := tts.New("PlainTTS", &spacedSynth{})
	tr := base.PronunciationTransformIPA(map[string]string{"Metformin": "mɛtˈfɔɹmɪn"})
	got, err := tr.Transform(context.Background(), "Metformin", frames.AnyAggregation)
	if err != nil || got != "Metformin" {
		t.Fatalf("got %q, %v; want the word as written", got, err)
	}
}
