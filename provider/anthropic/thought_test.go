package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
)

// thoughtStream is a thinking block streamed in two chunks, then signed and
// closed, then the answer in a text block of its own.
const thoughtStream = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\"," +
	"\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"stop_reason\":null," +
	"\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":0," +
	"\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0," +
	"\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"The menu \"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0," +
	"\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"has tea.\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0," +
	"\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-1\"}}\n\n" +
	"event: content_block_stop\n" +
	"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":1," +
	"\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":1," +
	"\"delta\":{\"type\":\"text_delta\",\"text\":\"We have tea.\"}}\n\n" +
	"event: content_block_stop\n" +
	"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
	"event: message_delta\n" +
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}," +
	"\"usage\":{\"output_tokens\":12}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// streamedFrames runs one generation of convo through a service streaming body
// and returns the thought and text frames it pushed downstream, in order.
func streamedFrames(t *testing.T, body string, convo *frames.LLMContext) []frames.Frame {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL})

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
			*frames.LLMThoughtEndFrame, *frames.LLMTextFrame:
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

// TestThinkingBlockStreamsAsThoughtFrames checks a thinking block is pushed as
// a thought: opened when the block starts, asked back into the conversation as
// this provider's own message, a chunk per thinking delta, and ended by the
// signature delta carrying the signature. The answer text follows as usual.
func TestThinkingBlockStreamsAsThoughtFrames(t *testing.T) {
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("Is there tea?")

	got := streamedFrames(t, thoughtStream, convo)
	if len(got) != 5 {
		t.Fatalf("frames = %v, want start, two chunks, end, then the text", got)
	}
	start, ok := got[0].(*frames.LLMThoughtStartFrame)
	if !ok || !start.AppendToContext || start.LLM != "anthropic" {
		t.Errorf("frame 0 = %v, want a thought start asked back into the context for anthropic", got[0])
	}
	for i, want := range []string{"The menu ", "has tea."} {
		if txt, ok := got[1+i].(*frames.LLMThoughtTextFrame); !ok || txt.Text != want {
			t.Errorf("frame %d = %v, want thought text %q", 1+i, got[1+i], want)
		}
	}
	if end, ok := got[3].(*frames.LLMThoughtEndFrame); !ok || end.Signature != "sig-1" {
		t.Errorf("frame 3 = %v, want a thought end carrying the signature", got[3])
	}
	if txt, ok := got[4].(*frames.LLMTextFrame); !ok || txt.Text != "We have tea." {
		t.Errorf("frame 4 = %v, want the answer text", got[4])
	}
}

// TestThinkingBlockStreamsAsThoughtFramesWithTools checks the tool-capable
// generation pushes the same thought frames.
func TestThinkingBlockStreamsAsThoughtFramesWithTools(t *testing.T) {
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("Is there tea?")
	convo.SetTools([]frames.Tool{{Name: "check_menu", Description: "Check the menu."}})

	got := streamedFrames(t, thoughtStream, convo)
	if len(got) != 5 {
		t.Fatalf("frames = %v, want start, two chunks, end, then the text", got)
	}
	if _, ok := got[0].(*frames.LLMThoughtStartFrame); !ok {
		t.Errorf("frame 0 = %v, want a thought start", got[0])
	}
	if end, ok := got[3].(*frames.LLMThoughtEndFrame); !ok || end.Signature != "sig-1" {
		t.Errorf("frame 3 = %v, want a thought end carrying the signature", got[3])
	}
}
