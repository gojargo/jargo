package stt

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
)

// languageTranscriber knows the language of what it transcribes.
type languageTranscriber struct{ lang string }

func (t *languageTranscriber) Transcribe(context.Context, []byte, int) (string, error) {
	return "from Transcribe", nil
}

func (t *languageTranscriber) TranscribeLanguage(context.Context, []byte, int) (string, string, error) {
	return "hola", t.lang, nil
}

// A transcriber that knows the language of its transcript is transcribed
// through, and the transcript is labeled with that language.
func TestSegmentTranscriptCarriesTheLanguageTheTranscriberReports(t *testing.T) {
	t.Parallel()

	svc := NewSegment("TestSTT", &languageTranscriber{lang: "es"}, 16000)
	got := make(chan *frames.TranscriptionFrame, 1)
	task, stop := runWatched(t, pipeline.New(svc), func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			select {
			case got <- tf:
			default:
			}
		}
	})
	defer stop()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 320), 16000, 1))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}))

	select {
	case tf := <-got:
		if tf.Text != "hola" || tf.Language != "es" {
			t.Errorf("transcript = %q in %q, want %q in es", tf.Text, tf.Language, "hola")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no transcript")
	}
}

// settlingTranscriber settles on another model than the one an update asked
// for, as a provider whose language does not publish it does.
type settlingTranscriber struct {
	settingsTranscriber
	settled string
}

func (t *settlingTranscriber) Settings() any { return &t.store }

func (t *settlingTranscriber) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	t.store.Model = settings.Set(t.settled)
	return false, nil
}

// The model the provider settled on, not the one the update asked for, labels
// what the service reports next.
func TestSegmentServiceRelabelsWithTheModelTheProviderSettledOn(t *testing.T) {
	tr := &settlingTranscriber{settled: "base"}
	tr.store.Model = settings.Set("small")
	tr.meta = Metadata{Model: "small"}
	s := NewSegment("TestSTT", tr, 0)

	s.updateSettings(context.Background(), updateFrame(t, "language", "uk"))

	if got := s.modelName(); got != "base" {
		t.Errorf("model = %q, want the base model the provider settled on", got)
	}
}
