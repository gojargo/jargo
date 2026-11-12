package nvidia_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/nvidia"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/utils/events"
)

// ttfbMarker stands in for the metrics frame carrying the TTFB in what
// nvidiaFrames returns.
type ttfbMarker struct{ frames.Frame }

// sse renders deltas as a chat-completions stream.
func sse(deltas ...map[string]any) string {
	var b strings.Builder
	for _, d := range deltas {
		data, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": d}}})
		b.WriteString("data: " + string(data) + "\n\n")
	}
	return b.String() + "data: [DONE]\n\n"
}

// nvidiaFrames runs one generation through a service streaming body, and
// returns the TTFB, thought and text frames it pushed downstream, in order, and
// the Authorization header the request carried.
func nvidiaFrames(t *testing.T, cfg chat.LLMConfig, body string) ([]frames.Frame, string) {
	t.Helper()
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	cfg.BaseURL = srv.URL
	svc := nvidia.NewLLM(cfg)

	noInitial := false
	var mu sync.Mutex
	var got []frames.Frame
	done := make(chan struct{})
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params:                  pipeline.Params{EnableMetrics: true, SendInitialEmptyMetrics: &noInitial},
	})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f := f.(type) {
		case *frames.MetricsFrame:
			for _, d := range f.Data {
				if _, ok := d.(frames.TTFBMetricsData); ok {
					got = append(got, ttfbMarker{f})
				}
			}
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

	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hi")
	worker.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the response never ended")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), got...), auth
}

// describe names each frame for comparison: a thought's start and end, its
// text, the answer's text, or the TTFB.
func describe(fs []frames.Frame) []string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		switch f := f.(type) {
		case ttfbMarker:
			out = append(out, "ttfb")
		case *frames.LLMThoughtStartFrame:
			out = append(out, "start")
		case *frames.LLMThoughtTextFrame:
			out = append(out, "thought:"+f.Text)
		case *frames.LLMThoughtEndFrame:
			out = append(out, "end")
		case *frames.LLMTextFrame:
			out = append(out, "text:"+f.Text)
		}
	}
	return out
}

func wantFrames(t *testing.T, got []frames.Frame, want ...string) {
	t.Helper()
	if g := describe(got); strings.Join(g, "|") != strings.Join(want, "|") {
		t.Errorf("frames = %q, want %q", g, want)
	}
}

// Ported from upstream. Reasoning is observable only after its TTFB is
// reported, whether it streams in its own field or in <think> tags.
func TestReasoningOutputReportsTTFBBeforeThoughtFrames(t *testing.T) {
	for name, delta := range map[string]map[string]any{
		"reasoning-content": {"reasoning_content": "Let me think."},
		"think-tags":        {"content": "<think>Let me think.</think>Answer"},
	} {
		t.Run(name, func(t *testing.T) {
			got, _ := nvidiaFrames(t, chat.LLMConfig{APIKey: "test-key"}, sse(delta))
			ttfb := 0
			for _, f := range got {
				if _, ok := f.(ttfbMarker); ok {
					ttfb++
				}
			}
			if ttfb != 1 || len(got) < 2 {
				t.Fatalf("frames = %q, want one TTFB", describe(got))
			}
			if _, ok := got[0].(ttfbMarker); !ok {
				t.Errorf("frames = %q, want the TTFB first", describe(got))
			}
			if _, ok := got[1].(*frames.LLMThoughtStartFrame); !ok {
				t.Errorf("frames = %q, want the thought start after the TTFB", describe(got))
			}
		})
	}
}

// Ported from upstream. A normal response reports its TTFB without pushing
// thought frames.
func TestNonReasoningOutputReportsTTFBWithoutThoughtFrames(t *testing.T) {
	got, _ := nvidiaFrames(t, chat.LLMConfig{APIKey: "test-key"}, sse(map[string]any{"content": "Answer"}))
	wantFrames(t, got, "ttfb", "text:Answer")
}

// TestThinkTagsSplitAcrossDeltas checks a leading <think> block split across
// deltas is pushed as a thought, and only what follows it is the answer. Inside
// the thought, what could open the closing tag is held back, so the thought text
// comes out as far as is safe.
func TestThinkTagsSplitAcrossDeltas(t *testing.T) {
	got, _ := nvidiaFrames(t, chat.LLMConfig{APIKey: "k"}, sse(
		map[string]any{"content": "<thi"},
		map[string]any{"content": "nk>Hmm.</th"},
		map[string]any{"content": "ink>\n"},
		map[string]any{"content": "Hello!"},
	))
	wantFrames(t, got, "ttfb", "start", "thought:H", "thought:mm.", "end", "text:Hello!")
}

// TestReasoningFieldEndsWhenTheAnswerBegins checks a thought read from the
// reasoning field ends at the first answer content, and that the field's older
// name, reasoning, is read too.
func TestReasoningFieldEndsWhenTheAnswerBegins(t *testing.T) {
	got, _ := nvidiaFrames(t, chat.LLMConfig{APIKey: "k"}, sse(
		map[string]any{"reasoning_content": "First, "},
		map[string]any{"reasoning": "the menu."},
		map[string]any{"content": "We have tea."},
	))
	wantFrames(t, got, "ttfb", "start", "thought:First, ", "thought:the menu.", "end", "text:We have tea.")
}

// TestReasoningFieldLeftOpenIsEndedWithTheStream checks a stream that ends
// while reasoning still ends the thought.
func TestReasoningFieldLeftOpenIsEndedWithTheStream(t *testing.T) {
	got, _ := nvidiaFrames(t, chat.LLMConfig{APIKey: "k"}, sse(map[string]any{"reasoning_content": "Hmm."}))
	wantFrames(t, got, "ttfb", "start", "thought:Hmm.", "end")
}

// TestHeldBackTextIsReleasedAtTheEnd checks text held back as a possible
// <think> opening reaches the answer when the stream ends.
func TestHeldBackTextIsReleasedAtTheEnd(t *testing.T) {
	got, _ := nvidiaFrames(t, chat.LLMConfig{APIKey: "k"}, sse(map[string]any{"content": "<"}))
	wantFrames(t, got, "ttfb", "text:<")
}

// TestLocalNIMNeedsNoKey checks a service given no API key still sends one,
// as a local NIM takes none, rather than an empty bearer token.
func TestLocalNIMNeedsNoKey(t *testing.T) {
	_, auth := nvidiaFrames(t, chat.LLMConfig{}, sse(map[string]any{"content": "Hi"}))
	if auth != "Bearer not-needed" {
		t.Errorf("Authorization = %q, want the placeholder key", auth)
	}
}
