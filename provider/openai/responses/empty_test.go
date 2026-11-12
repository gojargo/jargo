package responses

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

// turnErrors runs one turn through a WebSocket service answered with events,
// and returns the errors it reported.
func turnErrors(t *testing.T, events_ ...string) []*frames.ErrorFrame {
	t.Helper()
	srv := newFakeResponsesWS(t, [][]string{events_})
	svc := NewLLM(Config{APIKey: "k", WSURL: srv.wsURL()})

	var mu sync.Mutex
	var errs []*frames.ErrorFrame
	ended := make(chan struct{}, 1)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			mu.Lock()
			errs = append(errs, ef)
			mu.Unlock()
		}
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
			select {
			case ended <- struct{}{}:
			default:
			}
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	task.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("the response never ended")
	}
	time.Sleep(50 * time.Millisecond)
	task.StopWhenDone()
	<-done
	mu.Lock()
	defer mu.Unlock()
	return append([]*frames.ErrorFrame(nil), errs...)
}

const created1 = `{"type":"response.created","response":{"id":"resp_1"}}`

// wantEmptyResponseError checks one non-fatal error naming the response was
// reported.
func wantEmptyResponseError(t *testing.T, errs []*frames.ErrorFrame) {
	t.Helper()
	if len(errs) != 1 {
		t.Fatalf("errors = %d, want one", len(errs))
	}
	if errs[0].Fatal || !strings.Contains(errs[0].Error, "resp_1 completed with no text or function call") {
		t.Errorf("error = %q (fatal %v), want a non-fatal report naming resp_1", errs[0].Error, errs[0].Fatal)
	}
}

// Ported from upstream (test_reasoning_only_response_pushes_error).
func TestReasoningOnlyResponsePushesError(t *testing.T) {
	wantEmptyResponseError(t, turnErrors(t, created1,
		`{"type":"response.completed","response":{"id":"resp_1","output":[{"type":"reasoning","id":"rs_1"}]}}`))
}

// Ported from upstream (test_refusal_is_named_in_the_warning).
func TestRefusalPushesError(t *testing.T) {
	wantEmptyResponseError(t, turnErrors(t, created1,
		`{"type":"response.completed","response":{"id":"resp_1","output":[{"type":"message",`+
			`"role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}]}}`))
}

// Ported from upstream (test_whitespace_only_text_pushes_error).
func TestWhitespaceOnlyTextPushesError(t *testing.T) {
	wantEmptyResponseError(t, turnErrors(t, created1,
		`{"type":"response.output_text.delta","delta":"  \n"}`,
		`{"type":"response.completed","response":{"id":"resp_1"}}`))
}

// TestTextResponsePushesNoError checks a response that said something reports
// nothing.
func TestTextResponsePushesNoError(t *testing.T) {
	if errs := turnErrors(t, created1,
		`{"type":"response.output_text.delta","delta":"Hello."}`,
		`{"type":"response.completed","response":{"id":"resp_1"}}`); len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
}

// TestRefusalIsNamedInTheOutputItems checks the warning lists a message by its
// content types, so a refusal shows as message[refusal].
func TestRefusalIsNamedInTheOutputItems(t *testing.T) {
	s := &streamState{output: []responseOutput{{Type: "message"}}}
	s.output[0].Content = append(s.output[0].Content, struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{Type: "refusal"})
	if got := outputItemKinds(s); len(got) != 1 || got[0] != "message[refusal]" {
		t.Errorf("items = %v, want [message[refusal]]", got)
	}
}
