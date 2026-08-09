package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// stampGen answers every turn with one fixed reply.
type stampGen struct{}

func (stampGen) Generate(_ context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	return emit("the reply")
}

// runStamped runs one turn through an LLM service, optionally configuring its
// output first, and returns what the frames describing the response were
// stamped with. A nil entry is an unstamped frame.
func runStamped(t *testing.T, configure *bool) []*bool {
	t.Helper()

	var mu sync.Mutex
	var stamps []*bool
	record := func(f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.LLMTextFrame:
			stamps = append(stamps, fr.SkipTTS)
		case *frames.LLMFullResponseStartFrame:
			stamps = append(stamps, fr.SkipTTS)
		case *frames.LLMFullResponseEndFrame:
			stamps = append(stamps, fr.SkipTTS)
		}
	}

	task := pipeline.NewTask(
		pipeline.New(llm.New("StampLLM", stampGen{})),
		pipeline.TaskParams{OnReachedDownstream: record},
	)
	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()

	if configure != nil {
		task.QueueFrame(frames.NewLLMConfigureOutputFrame(*configure))
	}
	task.QueueFrame(frames.NewLLMContextFrame(frames.NewLLMContext("system")))
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return append([]*bool(nil), stamps...)
}

// TestOutputUnstampedByDefault checks nothing is stamped until something
// configures the output, leaving the decision to whatever else may set it.
func TestOutputUnstampedByDefault(t *testing.T) {
	stamps := runStamped(t, nil)
	if len(stamps) == 0 {
		t.Fatal("the turn produced no response frames")
	}
	for i, s := range stamps {
		if s != nil {
			t.Errorf("response frame %d was stamped %v, want unstamped", i, *s)
		}
	}
}

// TestOutputStampedToSkipTTS checks the frames of a response carry the
// configuration in effect, which is what a TTS service downstream reads.
func TestOutputStampedToSkipTTS(t *testing.T) {
	skip := true
	stamps := runStamped(t, &skip)
	if len(stamps) == 0 {
		t.Fatal("the turn produced no response frames")
	}
	for i, s := range stamps {
		if s == nil || !*s {
			t.Errorf("response frame %d = %v, want stamped to skip TTS", i, s)
		}
	}
}

// TestOutputStampedToSpeak checks the configuration is carried whichever way it
// is set, so putting it back reaches the frames too.
func TestOutputStampedToSpeak(t *testing.T) {
	skip := false
	stamps := runStamped(t, &skip)
	if len(stamps) == 0 {
		t.Fatal("the turn produced no response frames")
	}
	for i, s := range stamps {
		if s == nil || *s {
			t.Errorf("response frame %d = %v, want stamped to be spoken", i, s)
		}
	}
}
