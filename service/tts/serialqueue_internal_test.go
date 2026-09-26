package tts

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestSerialQueueResetKeepsUninterruptibleFrames checks an interruption drops
// the audio contexts and interruptible frames queued for serialization, and
// keeps the uninterruptible ones, which must not be lost mid-flight.
func TestSerialQueueResetKeepsUninterruptibleFrames(t *testing.T) {
	q := newSerialQueue()
	result := frames.NewFunctionCallResultFrame("1", "f", nil, "done")
	q.push(serialItem{contextID: "ctx"})
	q.push(serialItem{frame: frames.NewTextFrame("dropped")})
	q.push(serialItem{frame: result})
	q.push(serialItem{end: true})

	q.reset()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	got, ok := q.get(t.Context())
	if !ok || got.frame != result {
		t.Fatalf("reset kept %+v, want the function call result", got)
	}
	if it, ok := q.get(ctx); ok {
		t.Fatalf("reset kept %+v as well", it)
	}
}
