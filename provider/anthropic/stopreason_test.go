package anthropic

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// warningRecorder is a log handler keeping the message of every record at
// WARN and above.
type warningRecorder struct {
	mu       sync.Mutex
	messages []string
}

func (w *warningRecorder) Enabled(_ context.Context, l slog.Level) bool { return l >= slog.LevelWarn }

func (w *warningRecorder) Handle(_ context.Context, r slog.Record) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.messages = append(w.messages, r.Message)
	return nil
}

func (w *warningRecorder) WithAttrs([]slog.Attr) slog.Handler { return w }
func (w *warningRecorder) WithGroup(string) slog.Handler      { return w }

func (w *warningRecorder) all() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return slices.Clone(w.messages)
}

// captureWarnings routes the default logger to a recorder of the warnings
// logged until the test ends.
func captureWarnings(t *testing.T) *warningRecorder {
	t.Helper()
	rec := &warningRecorder{}
	prev := slog.Default()
	slog.SetDefault(slog.New(rec))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return rec
}

// streamText is a streamed text delta.
func streamText(text string) string {
	return sseEvent("content_block_delta",
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"`+text+`"}}`)
}

// streamStop is the message_delta ending a stream with reason, refused under
// category when one is given. An empty reason sends a null one.
func streamStop(reason, category string) string {
	stopReason, details := "null", "null"
	if reason != "" {
		stopReason = `"` + reason + `"`
	}
	if category != "" {
		details = `{"type":"refusal","category":"` + category + `","explanation":null}`
	}
	return sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":`+stopReason+
		`,"stop_details":`+details+`},"usage":{"output_tokens":1}}`)
}

// streamedTurn is what one streamed turn pushed and logged.
type streamedTurn struct {
	// frames are the response's start, text and end frames, in order.
	frames []frames.Frame
	// texts is the text of each text frame.
	texts    []string
	warnings []string
}

// streamTurn runs a conversation through the service in a pipeline, against a
// stub streaming the given events between message_start and message_stop.
func streamTurn(t *testing.T, body ...string) streamedTurn {
	t.Helper()
	_, turn := runTurn(t, convoOf("hi"), func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(messageStart + textBlockStart + strings.Join(body, "") + messageStop))
	})
	return turn
}

// convoOf is a conversation of one user message.
func convoOf(text string) *frames.LLMContext {
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage(text)
	return convo
}

// runTurn runs convo through a service in a pipeline, against a stub answering
// with handle, and returns the service and what the turn pushed and logged.
func runTurn(t *testing.T, convo *frames.LLMContext, handle http.HandlerFunc) (*Service, streamedTurn) {
	t.Helper()
	srv := httptest.NewServer(handle)
	t.Cleanup(srv.Close)
	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL, Model: "claude-sonnet-5-5"})
	rec := captureWarnings(t)

	var mu sync.Mutex
	var turn streamedTurn
	done := make(chan struct{})
	worker := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f := f.(type) {
		case *frames.LLMFullResponseStartFrame:
			turn.frames = append(turn.frames, f)
		case *frames.LLMTextFrame:
			turn.frames = append(turn.frames, f)
			turn.texts = append(turn.texts, f.Text)
		case *frames.LLMFullResponseEndFrame:
			turn.frames = append(turn.frames, f)
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
	turn.warnings = rec.all()
	return svc, turn
}

// TestANormalStopLogsNoWarning checks a response that ended the way the model
// meant it to is passed through untouched.
func TestANormalStopLogsNoWarning(t *testing.T) {
	for _, reason := range []string{"end_turn", "tool_use", "stop_sequence"} {
		t.Run(reason, func(t *testing.T) {
			turn := streamTurn(t, streamText("Hello there."), streamStop(reason, ""))
			if !slices.Equal(turn.texts, []string{"Hello there."}) {
				t.Errorf("texts = %q, want the response passed through", turn.texts)
			}
			if len(turn.warnings) != 0 {
				t.Errorf("warnings = %q, want none", turn.warnings)
			}
		})
	}
}

// TestARefusalLogsTheReasonAndCategory checks a refused turn names why, rather
// than ending as a silent empty turn.
func TestARefusalLogsTheReasonAndCategory(t *testing.T) {
	turn := streamTurn(t, streamStop("refusal", "reasoning_extraction"))
	if len(turn.warnings) != 1 || !strings.Contains(turn.warnings[0], "refusal (reasoning_extraction)") {
		t.Fatalf("warnings = %q, want one naming the refusal and its category", turn.warnings)
	}
	if _, ok := turn.frames[0].(*frames.LLMFullResponseStartFrame); !ok {
		t.Errorf("first frame = %T, want the response's start", turn.frames[0])
	}
	if _, ok := turn.frames[len(turn.frames)-1].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("last frame = %T, want the response's end", turn.frames[len(turn.frames)-1])
	}
}

// TestARefusalWithoutACategoryLogsTheReason checks the warning when Anthropic
// leaves the category out of a refusal that maps to no named one.
func TestARefusalWithoutACategoryLogsTheReason(t *testing.T) {
	turn := streamTurn(t, streamStop("refusal", ""))
	if len(turn.warnings) != 1 || !strings.HasSuffix(turn.warnings[0], "refusal") {
		t.Fatalf("warnings = %q, want one ending with the reason", turn.warnings)
	}
}

// TestATruncatedResponseIsPassedThroughWithAWarning checks a curtailed response
// is a warning: the partial text is still usable.
func TestATruncatedResponseIsPassedThroughWithAWarning(t *testing.T) {
	for _, reason := range []string{"max_tokens", "model_context_window_exceeded", "pause_turn", "a_new_reason"} {
		t.Run(reason, func(t *testing.T) {
			turn := streamTurn(t, streamText("Sure, here's the "), streamStop(reason, ""))
			if !slices.Equal(turn.texts, []string{"Sure, here's the "}) {
				t.Errorf("texts = %q, want the partial response passed through", turn.texts)
			}
			if len(turn.warnings) != 1 || !strings.Contains(turn.warnings[0], reason) {
				t.Errorf("warnings = %q, want one naming %s", turn.warnings, reason)
			}
		})
	}
}

// TestAMessageDeltaWithoutAStopReasonLogsNoWarning checks a message_delta that
// only carries usage is not reported.
func TestAMessageDeltaWithoutAStopReasonLogsNoWarning(t *testing.T) {
	turn := streamTurn(t, streamText("Partial"), streamStop("", ""))
	if len(turn.warnings) != 0 {
		t.Errorf("warnings = %q, want none", turn.warnings)
	}
}

// runInferenceReply runs a one-shot inference against a stub answering with a
// complete message holding text, or no content when text is empty, that
// stopped for reason under category. It returns the answer and the warnings
// logged during the call.
func runInferenceReply(t *testing.T, text, reason, category string) (string, []string) {
	t.Helper()
	content, details := "[]", "null"
	if text != "" {
		content = `[{"type":"text","text":"` + text + `"}]`
	}
	if category != "" {
		details = `{"type":"refusal","category":"` + category + `","explanation":null}`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"claude",` +
			`"content":` + content + `,"stop_reason":"` + reason + `","stop_details":` + details +
			`,"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	svc := NewLLM(Config{APIKey: "k", BaseURL: srv.URL, Model: "claude-sonnet-5-5"})
	rec := captureWarnings(t)
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	got, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	return got, rec.all()
}

// TestRunInferenceNormalStopLogsNoWarning checks a one-shot reply that ended
// normally is returned without a warning.
func TestRunInferenceNormalStopLogsNoWarning(t *testing.T) {
	for _, reason := range []string{"end_turn", "tool_use", "stop_sequence"} {
		t.Run(reason, func(t *testing.T) {
			got, warnings := runInferenceReply(t, "Hello there.", reason, "")
			if got != "Hello there." {
				t.Errorf("answer = %q, want the reply's text", got)
			}
			if len(warnings) != 0 {
				t.Errorf("warnings = %q, want none", warnings)
			}
		})
	}
}

// TestRunInferenceRefusalLogsTheReasonAndCategory checks a refused one-shot
// inference returns no text and logs why.
func TestRunInferenceRefusalLogsTheReasonAndCategory(t *testing.T) {
	got, warnings := runInferenceReply(t, "", "refusal", "reasoning_extraction")
	if got != "" {
		t.Errorf("answer = %q, want none", got)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "refusal (reasoning_extraction)") {
		t.Errorf("warnings = %q, want one naming the refusal and its category", warnings)
	}
}

// TestRunInferenceTruncatedReplyIsReturnedWithAWarning checks a reply cut off
// at max_tokens is still returned, with a warning naming the cause.
func TestRunInferenceTruncatedReplyIsReturnedWithAWarning(t *testing.T) {
	got, warnings := runInferenceReply(t, "The conversation so far cove", "max_tokens", "")
	if got != "The conversation so far cove" {
		t.Errorf("answer = %q, want the partial reply", got)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "max_tokens") {
		t.Errorf("warnings = %q, want one naming max_tokens", warnings)
	}
}
