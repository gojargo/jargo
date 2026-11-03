package live

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// These tests are not ports of upstream tests: they cover how frames travel
// through the service, which upstream sets in process_frame, and the report of
// an async tool on a model that takes no non-blocking calls.

// The session opens when the service is set up, before the pipeline starts.
func TestTheSessionOpensAtSetup(t *testing.T) {
	srv := newRecordingSession(t)
	startedOn(t, wsURL(srv))
	waitFor(t, "the setup", func() bool { return len(srv.withKey("setup")) == 1 })
}

// The conversation stops at the service, and so does a tool set it cannot
// apply to a running session.
func TestTheConversationAndToolSetsStopHere(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()

	if err := s.ProcessFrame(ctx, frames.NewLLMContextFrame(frames.NewLLMContext("")), processor.Downstream); err != nil {
		t.Fatalf("context: %v", err)
	}
	if err := s.ProcessFrame(ctx, frames.NewLLMSetToolsFrame(nil), processor.Downstream); err != nil {
		t.Fatalf("set tools: %v", err)
	}
	for _, n := range sink.names() {
		if n == "LLMContextFrame" || n == "LLMSetToolsFrame" {
			t.Errorf("%s traveled on, want it stopped here", n)
		}
	}
}

// An async tool's messages on a model that takes no non-blocking calls are
// reported once, as an error.
func TestAnAsyncToolOnAModelWithoutNonBlockingCallsIsReportedOnce(t *testing.T) {
	s, _, source := newTestServiceBothWays(t)
	// A 3.x model before 3.8 takes no non-blocking calls.
	s.live.Model = settings.Set("gemini-3.1-flash-live-preview")
	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.NewAsyncToolStartedMessage("call_1"))

	s.processCompletedCalls(context.Background(), convo, true)
	s.processCompletedCalls(context.Background(), convo, true)
	time.Sleep(50 * time.Millisecond)

	n := 0
	for _, f := range source.frames() {
		if _, ok := f.(*frames.ErrorFrame); ok {
			n++
		}
	}
	if n != 1 {
		t.Errorf("reported %d errors, want one", n)
	}
}
