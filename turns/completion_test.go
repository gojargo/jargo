package turns_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/turns"
)

type cfRecorder struct {
	down chan string
	up   chan string
}

func newCFRecorder() *cfRecorder {
	return &cfRecorder{down: make(chan string, 32), up: make(chan string, 32)}
}

func (r *cfRecorder) onDown(f frames.Frame) {
	switch v := f.(type) {
	case *frames.UserTurnInferenceCompletedFrame:
		r.down <- "complete"
	case *frames.LLMMarkerFrame:
		r.down <- "marker:" + v.Marker
	case *frames.LLMTextFrame:
		r.down <- "text:" + v.Text
	}
}

func (r *cfRecorder) onUp(f frames.Frame) {
	switch f.(type) {
	case *frames.LLMMessagesAppendFrame:
		r.up <- "append"
	case *frames.LLMRunFrame:
		r.up <- "run"
	}
}

func expectStr(t *testing.T, ch chan string, want string) {
	t.Helper()
	select {
	case got := <-ch:
		if got != want {
			t.Fatalf("event = %q, want %q", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %q", want)
	}
}

func expectNoStr(t *testing.T, ch chan string, d time.Duration) {
	t.Helper()
	select {
	case got := <-ch:
		t.Fatalf("unexpected event %q", got)
	case <-time.After(d):
	}
}

func runCompletion(t *testing.T, f *turns.CompletionFilter) (*cfRecorder, *pipeline.Task, chan error) {
	t.Helper()
	rec := newCFRecorder()
	task := pipeline.NewTask(pipeline.New(f), pipeline.TaskParams{
		OnReachedDownstream: rec.onDown,
		OnReachedUpstream:   rec.onUp,
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return rec, task, done
}

// TestCompletionFilterComplete covers a "✓" reply: the turn completes and the
// text after the marker is forwarded.
func TestCompletionFilterComplete(t *testing.T) {
	f := turns.NewCompletionFilter(turns.UserTurnCompletionConfig{})
	rec, task, done := runCompletion(t, f)

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("✓ Hello there"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	expectStr(t, rec.down, "complete")
	expectStr(t, rec.down, "marker:✓")
	expectStr(t, rec.down, "text:Hello there")

	finish(t, task, done)
}

// TestCompletionFilterIncomplete covers a "○" reply: the text is suppressed and,
// after the timeout, the LLM is re-prompted.
func TestCompletionFilterIncomplete(t *testing.T) {
	f := turns.NewCompletionFilter(turns.UserTurnCompletionConfig{IncompleteShortTimeout: 30 * time.Millisecond})
	rec, task, done := runCompletion(t, f)

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("○"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	expectStr(t, rec.down, "marker:○")
	// The re-prompt fires upstream after the timeout.
	expectStr(t, rec.up, "append")
	expectStr(t, rec.up, "run")
	// No reply text or completion was forwarded downstream.
	expectNoStr(t, rec.down, 50*time.Millisecond)

	finish(t, task, done)
}
