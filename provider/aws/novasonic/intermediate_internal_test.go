package novasonic

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Nova Sonic takes a tool's intermediate results as text beside the call, and
// appended messages as text, interactive as asked.

const weather = `{"temperature":75}`

// sentEvents records what a service sends, with the session's prompt set.
type sentEvents struct {
	mu     sync.Mutex
	events []map[string]any
}

func recordSent(s *Service) *sentEvents {
	r := &sentEvents{}
	s.mu.Lock()
	s.promptName = "prompt"
	s.mu.Unlock()
	s.write = func(ev map[string]any) error {
		r.mu.Lock()
		r.events = append(r.events, ev)
		r.mu.Unlock()
		return nil
	}
	return r
}

// sentText is one text block the service sent: its text and whether it asked
// the model to answer.
type sentText struct {
	text        string
	interactive bool
}

// texts are the text blocks sent, in order.
func (r *sentEvents) texts(t *testing.T) []sentText {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []sentText
	var interactive bool
	for _, ev := range r.events {
		envelope, _ := ev["event"].(map[string]any)
		if start, ok := envelope["contentStart"].(map[string]any); ok && start["type"] == "TEXT" {
			interactive, _ = start["interactive"].(bool)
		}
		if in, ok := envelope["textInput"].(map[string]any); ok {
			text, _ := in[keyContent].(string)
			out = append(out, sentText{text: text, interactive: interactive})
		}
	}
	return out
}

// toolResults counts the tool results sent.
func (r *sentEvents) toolResults() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, ev := range r.events {
		envelope, _ := ev["event"].(map[string]any)
		if _, ok := envelope["toolResult"]; ok {
			n++
		}
	}
	return n
}

func novaResult(final, run bool) *frames.FunctionCallResultFrame {
	f := frames.NewFunctionCallResultFrame("call_1", "delegate", json.RawMessage(`{}`), weather)
	f.Properties = &frames.FunctionCallResultProperties{IsFinal: &final, RunLLM: &run}
	return f
}

func TestAnIntermediateResultGoesInAsInteractiveText(t *testing.T) {
	s, _ := newTestService(t)
	sent := recordSent(s)

	s.handleFunctionCallResult(context.Background(), novaResult(false, true))

	texts := sent.texts(t)
	if len(texts) != 1 || !texts[0].interactive {
		t.Fatalf("texts = %+v, want one interactive", texts)
	}
	p, ok := frames.ParseAsyncToolMessage(frames.Message{Role: frames.RoleDeveloper, Text: texts[0].text})
	if !ok || p.Kind != frames.AsyncToolIntermediate {
		t.Errorf("text = %q, want the intermediate result envelope", texts[0].text)
	}
	if n := sent.toolResults(); n != 0 {
		t.Errorf("tool results sent = %d, want none", n)
	}
}

func TestASilentResultGoesInWithoutAskingForAReply(t *testing.T) {
	s, _ := newTestService(t)
	sent := recordSent(s)

	s.handleFunctionCallResult(context.Background(), novaResult(false, false))

	if texts := sent.texts(t); len(texts) != 1 || texts[0].interactive {
		t.Fatalf("texts = %+v, want one not interactive", texts)
	}
}

func TestAFinalResultSettlesTheCall(t *testing.T) {
	s, _ := newTestService(t)
	sent := recordSent(s)

	s.handleFunctionCallResult(context.Background(), novaResult(true, true))

	if texts := sent.texts(t); len(texts) != 0 {
		t.Errorf("texts = %+v, want none", texts)
	}
	if n := sent.toolResults(); n != 1 {
		t.Errorf("tool results sent = %d, want one", n)
	}
	if !s.resultSeen("call_1") {
		t.Error("the call is not settled")
	}
}

func TestAnAppendedMessageGoesInAsTextInteractiveAsAsked(t *testing.T) {
	s, _ := newTestService(t)
	sent := recordSent(s)

	run := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleDeveloper, Text: "Backend: On it."}})
	run.RunLLM = true
	quiet := frames.NewLLMMessagesAppendFrame(
		[]frames.Message{{Role: frames.RoleDeveloper, Text: "Backend (working): read_file()"}})
	s.handleMessagesAppend(context.Background(), run)
	s.handleMessagesAppend(context.Background(), quiet)

	want := []sentText{{"Backend: On it.", true}, {"Backend (working): read_file()", false}}
	got := sent.texts(t)
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("texts = %+v, want %+v", got, want)
	}
}
