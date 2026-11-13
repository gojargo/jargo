package gemini

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// TestResponseIsBracketedWhenNoTextIsGenerated checks an empty response still
// opens and closes, so the aggregators are not left waiting for its end.
func TestResponseIsBracketedWhenNoTextIsGenerated(t *testing.T) {
	logged := capturedWarnings(t)

	var response []frames.Frame
	for _, f := range responseFrames(t, sse(doneChunk())) {
		switch f.(type) {
		case *frames.LLMFullResponseStartFrame, *frames.LLMTextFrame, *frames.LLMFullResponseEndFrame:
			response = append(response, f)
		}
	}

	if len(response) != 2 {
		t.Fatalf("response frames = %v, want a start and an end", response)
	}
	if _, ok := response[0].(*frames.LLMFullResponseStartFrame); !ok {
		t.Errorf("first = %v, want the response opened", response[0])
	}
	if _, ok := response[1].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("last = %v, want the response closed", response[1])
	}
	if logged() != "" {
		t.Errorf("logs = %q, want nothing said about a normal stop", logged())
	}
}

// inferOnce runs a one-shot inference against a reply and returns its answer
// and what it logged at WARN and above.
func inferOnce(t *testing.T, reply string) (answer, logs string) {
	t.Helper()
	logged := capturedWarnings(t)
	srv := newGenServer(t, reply)
	svc := serviceFor(srv, &testShaper{base: srv.URL}, Config{APIKey: "k"})

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hello")
	answer, err := svc.RunInference(t.Context(), convo, llm.InferenceOptions{})
	if err != nil {
		t.Fatalf("RunInference: %v", err)
	}
	return answer, logged()
}

// TestRunInferenceNormalStopLogsNoWarning checks a one-shot answer that ended
// normally is returned with nothing said about it.
func TestRunInferenceNormalStopLogsNoWarning(t *testing.T) {
	answer, logs := inferOnce(t,
		`{"candidates":[{"content":{"parts":[{"text":"Hello there."}]},"finishReason":"STOP"}]}`)

	if answer != "Hello there." {
		t.Errorf("answer = %q, want the candidate's text", answer)
	}
	if logs != "" {
		t.Errorf("logs = %q, want nothing said about a normal stop", logs)
	}
}

// TestRunInferenceWithheldReplyLogsTheReason checks a one-shot answer the model
// withheld comes back empty, with the reason it was withheld logged: without
// it, an empty summary says nothing about why.
func TestRunInferenceWithheldReplyLogsTheReason(t *testing.T) {
	for _, reason := range []string{"SAFETY", "PROHIBITED_CONTENT"} {
		t.Run(reason, func(t *testing.T) {
			answer, logs := inferOnce(t, `{"candidates":[{"finishReason":"`+reason+`"}]}`)

			if answer != "" {
				t.Errorf("answer = %q, want none", answer)
			}
			if strings.Count(logs, "\n") != 1 || !strings.Contains(logs, reason) {
				t.Errorf("logs = %q, want one warning naming the reason", logs)
			}
		})
	}
}

// TestRunInferenceTruncatedReplyIsReturnedWithAWarning checks a one-shot answer
// cut off at the output token limit is still returned, and the cut is logged.
func TestRunInferenceTruncatedReplyIsReturnedWithAWarning(t *testing.T) {
	answer, logs := inferOnce(t,
		`{"candidates":[{"content":{"parts":[{"text":"The conversation so far cove"}]},`+
			`"finishReason":"MAX_TOKENS"}]}`)

	if answer != "The conversation so far cove" {
		t.Errorf("answer = %q, want what did arrive", answer)
	}
	if strings.Count(logs, "\n") != 1 || !strings.Contains(logs, "MAX_TOKENS") {
		t.Errorf("logs = %q, want one warning naming the reason", logs)
	}
}
