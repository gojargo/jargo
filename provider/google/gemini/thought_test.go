package gemini

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// streamedFrames runs one generation of convo through a service streaming reply
// and returns the thought, text and context-append frames it pushed downstream,
// in order.
func streamedFrames(t *testing.T, reply string, convo *frames.LLMContext) []frames.Frame {
	t.Helper()
	srv := newGenServer(t, reply)
	svc := NewShapedLLM("GoogleLLM", &testShaper{base: srv.URL}, Config{APIKey: "k"})

	var mu sync.Mutex
	var got []frames.Frame
	done := make(chan struct{})
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f.(type) {
		case *frames.LLMThoughtStartFrame, *frames.LLMThoughtTextFrame,
			*frames.LLMThoughtEndFrame, *frames.LLMTextFrame, *frames.LLMMessagesAppendFrame:
			got = append(got, f)
		case *frames.LLMFullResponseEndFrame:
			close(done)
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- worker.Run(context.Background()) }()
	t.Cleanup(func() {
		worker.StopWhenDone()
		<-runDone
	})

	worker.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response never ended")
	}
	mu.Lock()
	defer mu.Unlock()
	return got
}

// keptSignature reads the thought signature a context-append frame carries.
func keptSignature(t *testing.T, f frames.Frame) (signature any, bookmark map[string]any) {
	t.Helper()
	app, ok := f.(*frames.LLMMessagesAppendFrame)
	if !ok || len(app.Messages) != 1 {
		t.Fatalf("frame = %v, want one message appended to the context", f)
	}
	m := app.Messages[0]
	native, ok := m.Native.(map[string]any)
	if m.LLM != "google" || !ok || native["type"] != "thought_signature" {
		t.Fatalf("message = %+v, want a thought signature written for google", m)
	}
	bookmark, _ = native["bookmark"].(map[string]any)
	return native["signature"], bookmark
}

// TestThoughtPartIsPushedAsAThought checks a thought summary is pushed as a
// whole thought rather than spoken as part of the answer.
func TestThoughtPartIsPushedAsAThought(t *testing.T) {
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("Is there tea?")
	got := streamedFrames(t, sse(
		`{"candidates":[{"content":{"parts":[{"text":"Checking the menu.","thought":true}]}}]}`,
		textChunk("We have tea."),
	), convo)

	if len(got) != 4 {
		t.Fatalf("frames = %v, want a whole thought then the answer", got)
	}
	if start, ok := got[0].(*frames.LLMThoughtStartFrame); !ok || start.AppendToContext {
		t.Errorf("frame 0 = %v, want a thought start not asked into the context", got[0])
	}
	if txt, ok := got[1].(*frames.LLMThoughtTextFrame); !ok || txt.Text != "Checking the menu." {
		t.Errorf("frame 1 = %v, want the thought text", got[1])
	}
	if _, ok := got[2].(*frames.LLMThoughtEndFrame); !ok {
		t.Errorf("frame 2 = %v, want a thought end", got[2])
	}
	if txt, ok := got[3].(*frames.LLMTextFrame); !ok || txt.Text != "We have tea." {
		t.Errorf("frame 3 = %v, want only the answer spoken", got[3])
	}
}

// TestTextSignatureIsKeptWithTheTextSoFar checks a signature arriving on a
// trailing empty text chunk is kept bookmarked by all the answer text streamed
// before it.
func TestTextSignatureIsKeptWithTheTextSoFar(t *testing.T) {
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("Is there tea?")
	got := streamedFrames(t, sse(
		textChunk("We have "),
		textChunk("tea."),
		`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"c2lnLXRleHQ="}]}}]}`,
	), convo)

	if len(got) != 3 {
		t.Fatalf("frames = %v, want the two chunks then the kept signature", got)
	}
	sig, bookmark := keptSignature(t, got[2])
	if sig != "c2lnLXRleHQ=" {
		t.Errorf("signature = %v, want it kept as sent", sig)
	}
	if bookmark[BookmarkText] != "We have tea." {
		t.Errorf("bookmark = %v, want the answer text so far", bookmark)
	}
}

// TestFunctionCallSignatureIsKeptByTheCallID checks a signed function call is
// reported under the id Gemini gave it, and its signature kept bookmarked by
// that same id, which is what puts it back on the call next turn.
func TestFunctionCallSignatureIsKeptByTheCallID(t *testing.T) {
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("Book a table")
	convo.SetTools([]frames.Tool{{Name: "book_table", Description: "Book a table."}})
	got := streamedFrames(t, sse(
		`{"candidates":[{"content":{"parts":[`+
			`{"functionCall":{"id":"fc-1","name":"book_table","args":{}},"thoughtSignature":"c2lnLWNhbGw="},`+
			`{"functionCall":{"name":"book_table","args":{}},"thoughtSignature":"c2lnLWZyZXNo"}]}}]}`,
	), convo)

	if len(got) != 2 {
		t.Fatalf("frames = %v, want a kept signature per call", got)
	}
	sig, bookmark := keptSignature(t, got[0])
	if sig != "c2lnLWNhbGw=" || bookmark[BookmarkFunctionCall] != "fc-1" {
		t.Errorf("first = %v %v, want the signature bookmarked by the call's own id", sig, bookmark)
	}
	_, bookmark = keptSignature(t, got[1])
	if id, _ := bookmark[BookmarkFunctionCall].(string); id == "" || id == "fc-1" {
		t.Errorf("second bookmark = %v, want the fresh id the call was given", bookmark)
	}
}
