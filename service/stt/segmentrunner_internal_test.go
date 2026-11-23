package stt

import (
	"context"
	"iter"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// diarizingRunner transcribes every segment into two speaker runs, attributed
// to the user the service names, the way a provider owning its run does. With
// reject set it fails the segment instead, with the status it names.
type diarizingRunner struct {
	svc    *SegmentService
	reject int
}

func (r *diarizingRunner) RunSTT(_ context.Context, _ []byte, _ int) iter.Seq[frames.Frame] {
	return func(yield func(frames.Frame) bool) {
		r.svc.StartProcessingMetrics()
		if r.reject != 0 {
			ef := frames.NewErrorFrame("rejected")
			ef.Err = errs.NewHTTPStatusError(r.reject, nil)
			yield(ef)
			return
		}
		r.svc.StopProcessingMetrics(context.Background())
		ts := frames.NowTimestamp()
		for _, run := range r.svc.DiarizedRuns("Hi. Hello.", []SpeakerWord{{"Hi.", 0}, {"Hello.", 1}}) {
			tf := frames.NewTranscriptionFrame(run.Text, r.svc.UserID(), ts)
			tf.Speaker = run.Speaker
			if !yield(tf) {
				return
			}
		}
	}
}

// A service built with a SegmentRunner pushes what its run yields: the
// transcripts carry the user the audio was attributed to, and only the last
// speaker run of the segment is finalized.
func TestSegmentRunnerOwnsTheRun(t *testing.T) {
	t.Parallel()

	runner := &diarizingRunner{}
	svc := NewSegmentRunner("RunnerSTT", runner, 16000)
	runner.svc = svc

	var (
		mu  sync.Mutex
		got []*frames.TranscriptionFrame
	)
	done := make(chan struct{})
	task, stop := runWatched(t, pipeline.New(svc), func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			mu.Lock()
			got = append(got, tf)
			if len(got) == 2 {
				close(done)
			}
			mu.Unlock()
		}
	})
	defer stop()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0, time.Time{}))
	task.QueueFrame(frames.NewUserAudioRawFrame("user-1", make([]byte, 320), 16000, 1))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("no transcripts")
	}
	mu.Lock()
	defer mu.Unlock()
	for i, want := range []struct {
		text      string
		number    int
		finalized bool
	}{{"Hi.", 1, false}, {"Hello.", 2, true}} {
		tf := got[i]
		if tf.Text != want.text || tf.Speaker == nil || tf.Speaker.Number != want.number || tf.Finalized != want.finalized {
			t.Errorf("transcript %d = %q by %v finalized %v, want %q by speaker %d finalized %v",
				i, tf.Text, tf.Speaker, tf.Finalized, want.text, want.number, want.finalized)
		}
		if tf.UserID != "user-1" {
			t.Errorf("transcript %d user = %q, want user-1", i, tf.UserID)
		}
	}
}

// An error a run yields is reported as the service's own, classified, so a
// rejected request leaves the service unusable.
func TestSegmentRunnerErrorIsReportedAndClassified(t *testing.T) {
	t.Parallel()

	runner := &diarizingRunner{reject: http.StatusUnauthorized}
	svc := NewSegmentRunner("RunnerSTT", runner, 16000)
	runner.svc = svc

	var reported []*frames.ErrorFrame
	events.On(svc.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		reported = append(reported, ef)
	})
	svc.pushSegmentFrames(t.Context(), svc.RunSTT()(t.Context(), make([]byte, 320), 16000))

	if len(reported) != 1 || reported[0].Category != errs.Authentication {
		t.Fatalf("reported = %v, want one authentication error", reported)
	}
	if svc.Usable() {
		t.Error("the service is still usable after its key was rejected")
	}
}
