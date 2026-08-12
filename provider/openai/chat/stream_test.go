package chat

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// usageChunk is a streamed chunk carrying a snapshot of the token counts. The
// provider sends it with no choices of its own.
func usageChunk(prompt, completion, cached, reasoning int) string {
	return fmt.Sprintf(
		`{"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d,`+
			`"prompt_tokens_details":{"cached_tokens":%d},`+
			`"completion_tokens_details":{"reasoning_tokens":%d}}}`,
		prompt, completion, prompt+completion, cached, reasoning,
	)
}

// reportedUsage runs one generation against a server replying with replies[i]
// on request i, with usage metrics collected, and returns every token-usage
// report that reached the pipeline.
func reportedUsage(t *testing.T, replies ...string) []frames.LLMTokenUsage {
	t.Helper()

	var n int
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		reply := replies[min(n, len(replies)-1)]
		n++
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		// A reply marked with a trailing NUL is cut off where the marker is, as a
		// completion interrupted part way through is.
		cut := strings.HasSuffix(reply, "\x00")
		_, _ = w.Write([]byte(strings.TrimSuffix(reply, "\x00")))
		if !cut {
			return
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Drop the connection where the marker is, leaving the stream unfinished.
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, err := hj.Hijack()
			if err != nil {
				t.Errorf("hijacking the connection: %v", err)
				return
			}
			_ = conn.Close()
		}
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})

	var reported []frames.LLMTokenUsage
	ends := make(chan struct{}, len(replies))
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableUsageMetrics:      true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.MetricsFrame:
				for _, d := range fr.Data {
					if u, ok := d.(frames.LLMUsageMetricsData); ok {
						reported = append(reported, u.Value)
					}
				}
			case *frames.LLMFullResponseEndFrame:
				select {
				case ends <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(t.Context()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	for range replies {
		task.QueueFrame(frames.NewLLMContextFrame(convo))
		select {
		case <-ends:
		case <-time.After(3 * time.Second):
			t.Fatal("the response did not complete")
		}
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return reported
}

// TestSnapshotsAreReportedOnceAsAFinalTotal checks three snapshots for one
// completion produce one report, of the last. Providers differ over how often
// they send the counts, and a running total repeated on every chunk must not
// be reported as many times as it arrives.
func TestSnapshotsAreReportedOnceAsAFinalTotal(t *testing.T) {
	reported := reportedUsage(t, sse(
		usageChunk(20, 5, 0, 0),
		usageChunk(20, 12, 0, 0),
		usageChunk(20, 30, 0, 0),
	))

	if len(reported) != 1 {
		t.Fatalf("reports = %+v, want exactly one", reported)
	}
	got := reported[0]
	if got.PromptTokens != 20 || got.CompletionTokens != 30 || got.TotalTokens != 50 {
		t.Errorf("usage = %+v, want 20 in / 30 out / 50 total", got)
	}
}

// TestUsageIsReportedWhenTheResponseIsInterrupted checks a completion cut off
// part way still reports the last snapshot it saw: the tokens were spent.
func TestUsageIsReportedWhenTheResponseIsInterrupted(t *testing.T) {
	// No terminating [DONE], and the connection drops after the chunks.
	reported := reportedUsage(t,
		"data: "+usageChunk(20, 5, 0, 0)+"\n\ndata: "+usageChunk(20, 12, 0, 0)+"\n\n\x00",
	)

	if len(reported) != 1 {
		t.Fatalf("reports = %+v, want exactly one", reported)
	}
	if reported[0].CompletionTokens != 12 {
		t.Errorf("completion tokens = %d, want the last snapshot's 12", reported[0].CompletionTokens)
	}
}

// TestCachedAndReasoningCountsReachTheReport checks the snapshot is reported
// whole, so every count the provider sent survives.
func TestCachedAndReasoningCountsReachTheReport(t *testing.T) {
	reported := reportedUsage(t, sse(usageChunk(20, 30, 15, 8)))

	if len(reported) != 1 {
		t.Fatalf("reports = %+v, want exactly one", reported)
	}
	if n, ok := frames.TokenCount(reported[0].CacheReadTokens); !ok || n != 15 {
		t.Errorf("cache-read tokens = %d (reported: %v), want 15", n, ok)
	}
	if n, ok := frames.TokenCount(reported[0].ReasoningTokens); !ok || n != 8 {
		t.Errorf("reasoning tokens = %d (reported: %v), want 8", n, ok)
	}
}

// TestACompletionWithoutUsageReportsNothing checks a stream carrying no counts
// produces no metrics rather than a report of zeros.
func TestACompletionWithoutUsageReportsNothing(t *testing.T) {
	if reported := reportedUsage(t, sse(contentChunk("hi"))); len(reported) != 0 {
		t.Errorf("reports = %+v, want none", reported)
	}
}

// TestALaterCompletionDoesNotInheritEarlierUsage checks each completion starts
// from a clean slate, so the counts of one turn are not billed to the next.
func TestALaterCompletionDoesNotInheritEarlierUsage(t *testing.T) {
	reported := reportedUsage(t, sse(usageChunk(20, 30, 0, 0)), sse(contentChunk("hi")))

	if len(reported) != 1 {
		t.Errorf("reports = %+v, want only the first completion's", reported)
	}
}

// TestToolChoiceIsSentWithTheTools checks the conversation's tool choice reaches
// the endpoint, and that a request advertising no tools carries none.
func TestToolChoiceIsSentWithTheTools(t *testing.T) {
	tools := []frames.Tool{{Name: "get_weather", Description: "weather"}}

	t.Run("defaults to letting the model decide", func(t *testing.T) {
		srv := newLLMServer(t, sse(contentChunk("ok")))
		svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
		convo := frames.NewLLMContext("")
		convo.SetTools(tools)
		if err := svc.GenerateWithTools(t.Context(), convo, &fakeSink{}); err != nil {
			t.Fatalf("GenerateWithTools: %v", err)
		}
		if srv.body["tool_choice"] != string(frames.ToolChoiceAuto) {
			t.Errorf("tool_choice = %v, want %q", srv.body["tool_choice"], frames.ToolChoiceAuto)
		}
	})

	t.Run("carries what the conversation asked for", func(t *testing.T) {
		srv := newLLMServer(t, sse(contentChunk("ok")))
		svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
		convo := frames.NewLLMContext("")
		convo.SetTools(tools)
		convo.SetToolChoice(frames.ToolChoiceRequired)
		if err := svc.GenerateWithTools(t.Context(), convo, &fakeSink{}); err != nil {
			t.Fatalf("GenerateWithTools: %v", err)
		}
		if srv.body["tool_choice"] != string(frames.ToolChoiceRequired) {
			t.Errorf("tool_choice = %v, want %q", srv.body["tool_choice"], frames.ToolChoiceRequired)
		}
	})

	t.Run("absent when no tool is advertised", func(t *testing.T) {
		srv := newLLMServer(t, sse(contentChunk("ok")))
		svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
		if err := svc.GenerateWithTools(t.Context(), frames.NewLLMContext(""), &fakeSink{}); err != nil {
			t.Fatalf("GenerateWithTools: %v", err)
		}
		if _, ok := srv.body["tool_choice"]; ok {
			t.Errorf("body carries a tool choice with no tools to choose from: %v", srv.body)
		}
	})
}

// TestServiceTierIsSentWhenSet checks the tier reaches an endpoint that offers
// them and is left off otherwise.
func TestServiceTierIsSentWhenSet(t *testing.T) {
	srv := newLLMServer(t, sse(contentChunk("ok")))
	generate(t, srv, LLMConfig{APIKey: "k", ServiceTier: "flex"})
	if srv.body["service_tier"] != "flex" {
		t.Errorf("service_tier = %v, want %q", srv.body["service_tier"], "flex")
	}

	plain := newLLMServer(t, sse(contentChunk("ok")))
	generate(t, plain, LLMConfig{APIKey: "k"})
	if _, ok := plain.body["service_tier"]; ok {
		t.Errorf("body carries a tier nobody asked for: %v", plain.body)
	}
}

// TestUnparsableToolArgumentsAreDropped checks a call whose arguments did not
// arrive as valid JSON is not reported: nothing downstream could act on them,
// and the handler's own failure would read to the model as the tool failing.
func TestUnparsableToolArgumentsAreDropped(t *testing.T) {
	c := &toolCoalescer{calls: map[int]*toolAccumulator{}}
	sink := &fakeSink{}

	mustAdd(t, c, sink, chatDelta{ToolCalls: []toolCallDelta{toolDelta(0, "call_a", "broken", `{"loc`)}})
	mustAdd(t, c, sink, chatDelta{ToolCalls: []toolCallDelta{toolDelta(1, "call_b", "whole", `{"ok":true}`)}})
	if err := c.emit(sink); err != nil {
		t.Fatalf("emit: %v", err)
	}

	if len(sink.calls) != 1 {
		t.Fatalf("calls = %+v, want only the one whose arguments can be read", sink.calls)
	}
	if sink.calls[0].Name != "whole" {
		t.Errorf("call = %+v, want the whole one", sink.calls[0])
	}
}

// TestRetryOnTimeoutAsksAgainUnbounded checks a first attempt that does not
// start in time is abandoned and asked again, and that the second attempt is not
// held to the same bound.
func TestRetryOnTimeoutAsksAgainUnbounded(t *testing.T) {
	var mu sync.Mutex
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		first := attempts == 1
		mu.Unlock()
		if first {
			// Never answers: the attempt is abandoned when the bound expires. The
			// body is drained first, because a server with an unread body cannot
			// tell a client hanging up from more of the request arriving.
			_, _ = io.Copy(io.Discard, r.Body)
			<-r.Context().Done()
			return
		}
		// Slower than the bound, and answered in full all the same.
		time.Sleep(80 * time.Millisecond)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse(contentChunk("second time lucky"))))
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(LLMConfig{
		APIKey: "k", BaseURL: srv.URL,
		RetryOnTimeout: true, RetryTimeout: 40 * time.Millisecond,
	})

	var out strings.Builder
	err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(text string) error {
		out.WriteString(text)
		return nil
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if out.String() != "second time lucky" {
		t.Errorf("streamed text = %q, want the retry's answer", out.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if attempts != 2 {
		t.Errorf("attempts = %d, want the first abandoned and one retry", attempts)
	}
}

// TestATimeoutIsReportedAsOne checks a stalled request without the retry is
// reported as a timeout, which is the failure a switcher fails over on rather
// than an error like any other.
func TestATimeoutIsReportedAsOne(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	svc.http = &http.Client{Timeout: 40 * time.Millisecond}

	err := svc.Generate(t.Context(), frames.NewLLMContext(""), func(string) error { return nil })
	if !errors.Is(err, llm.ErrCompletionTimeout) {
		t.Errorf("Generate error = %v, want it to report a completion timeout", err)
	}
}

// TestRunInferenceAnswersOnce checks the one-shot path: no streaming, the
// instruction it was given ahead of the conversation's own, its own bound on the
// answer, and the text handed straight back.
func TestRunInferenceAnswersOnce(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"a short summary"}}]}`))
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("what did we agree?")

	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{
		MaxTokens:         64,
		SystemInstruction: "Summarize the conversation.",
	})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	if got != "a short summary" {
		t.Errorf("answer = %q, want the completion's content", got)
	}
	if body["stream"] != false {
		t.Errorf("stream = %v, want the one-shot path not to stream", body["stream"])
	}
	if _, ok := body["stream_options"]; ok {
		t.Errorf("body asks for streamed usage on a request that does not stream: %v", body)
	}
	if body["max_tokens"] != float64(64) {
		t.Errorf("max_tokens = %v, want the bound this inference was given", body["max_tokens"])
	}
	msgs := messagesOf(t, body)
	if len(msgs) != 3 {
		t.Fatalf("messages = %v, want the instruction ahead of the conversation", msgs)
	}
	if msgs[0]["role"] != RoleSystem || msgs[0]["content"] != "Summarize the conversation." {
		t.Errorf("message 0 = %v, want the instruction this inference was given", msgs[0])
	}
	if msgs[1]["role"] != RoleSystem || msgs[1]["content"] != "be brief" {
		t.Errorf("message 1 = %v, want the conversation's own system prompt", msgs[1])
	}
}

// TestRunInferenceReportsAFailedRequest checks an endpoint that refuses the
// request is reported rather than read as an empty answer.
func TestRunInferenceReportsAFailedRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"slow down"}`))
	}))
	t.Cleanup(srv.Close)

	svc := NewLLM(LLMConfig{APIKey: "k", BaseURL: srv.URL})
	if _, err := svc.RunInference(t.Context(), frames.NewLLMContext(""), llm.InferenceOptions{}); err == nil {
		t.Error("RunInference returned no error for a refused request")
	}
}

// TestServiceRunsOneShotInferences checks the service satisfies the interface a
// summarizer or a judge asks for.
func TestServiceRunsOneShotInferences(t *testing.T) {
	var _ llm.Inferencer = (*LLMService)(nil)
}
