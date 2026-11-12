package rtvi

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Tests for how the observer reports function calls, this pipeline's own and
// ones made outside it.

func callObserver(levels map[string]FunctionCallReportLevel) *Observer {
	params := DefaultObserverParams()
	params.FunctionCallReportLevel = levels
	return NewObserverWithParams(nil, params)
}

// reportCalls is what the observer sends for each frame, in order.
func reportCalls(o *Observer, fs ...frames.Frame) []Message {
	var sent []Message
	for _, f := range fs {
		msgs, _ := o.functionCallMessagesFor(f)
		sent = append(sent, msgs...)
	}
	return sent
}

func messageTypes(msgs []Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Type)
	}
	return out
}

// dataOf is the payload of m, read as a T.
func dataOf[T any](t *testing.T, m Message) T {
	t.Helper()
	d, ok := m.Data.(T)
	if !ok {
		t.Fatalf("%s carries %T", m.Type, m.Data)
	}
	return d
}

func externalCall(phase string) frames.Frame {
	args := json.RawMessage(`{"location":"Seattle"}`)
	switch phase {
	case "started":
		return frames.NewExternalFunctionCallsStartedFrame(
			[]frames.ExternalFunctionCall{{ToolName: "get_weather", ToolCallID: "toolu_1"}})
	case "in_progress":
		return frames.NewExternalFunctionCallInProgressFrame("toolu_1", "get_weather", args)
	default:
		return frames.NewExternalFunctionCallResultFrame("toolu_1", "get_weather", args, `{"temp":62}`)
	}
}

func TestAnExternalCallIsReportedLikeThePipelinesOwn(t *testing.T) {
	o := callObserver(map[string]FunctionCallReportLevel{"*": ReportFull})

	sent := reportCalls(o, externalCall("started"), externalCall("in_progress"), externalCall("stopped"))

	want := []string{TypeLLMFunctionCallStart, TypeLLMFunctionCall, TypeLLMFunctionCallStop}
	if got := messageTypes(sent); !slices.Equal(got, want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	if d := dataOf[LLMFunctionCallStartData](t, sent[0]); d.FunctionName != "get_weather" {
		t.Errorf("started names %q, want get_weather", d.FunctionName)
	}
	if d := dataOf[LLMFunctionCallData](t, sent[1]); string(d.Arguments) != `{"location":"Seattle"}` {
		t.Errorf("in progress carries %s, want the arguments", d.Arguments)
	}
	// The result goes as the value it holds, an object here, not as a string.
	raw, err := json.Marshal(dataOf[LLMFunctionCallStoppedData](t, sent[2]))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"result":{"temp":62}`) {
		t.Errorf("stopped carries %s, want the result as an object", raw)
	}
}

// TestExternalCallsAnnouncedTogetherAreEachReportedStarted checks the started
// frame carries every call announced at once, as the pipeline's own does.
func TestExternalCallsAnnouncedTogetherAreEachReportedStarted(t *testing.T) {
	o := callObserver(map[string]FunctionCallReportLevel{"*": ReportFull})

	sent := reportCalls(o, frames.NewExternalFunctionCallsStartedFrame([]frames.ExternalFunctionCall{
		{ToolName: "get_weather", ToolCallID: "toolu_1"},
		{ToolName: "get_time", ToolCallID: "toolu_2"},
	}))

	want := []string{TypeLLMFunctionCallStart, TypeLLMFunctionCallStart}
	if got := messageTypes(sent); !slices.Equal(got, want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	var names []string
	for _, m := range sent {
		names = append(names, dataOf[LLMFunctionCallStartData](t, m).FunctionName)
	}
	if want := []string{"get_weather", "get_time"}; !slices.Equal(names, want) {
		t.Errorf("named %v, want %v", names, want)
	}
}

// TestAnIntermediateResultLeavesTheCallRunning checks only a final result stops
// a call, the pipeline's own or an external one.
func TestAnIntermediateResultLeavesTheCallRunning(t *testing.T) {
	o := callObserver(map[string]FunctionCallReportLevel{"*": ReportFull})

	notFinal := false
	own := frames.NewFunctionCallResultFrame("call_1", "get_weather", json.RawMessage(`{}`), "62")
	own.Properties = &frames.FunctionCallResultProperties{IsFinal: &notFinal}
	external := frames.NewExternalFunctionCallResultFrame("toolu_1", "get_weather", json.RawMessage(`{}`), "62")
	external.IsFinal = false

	sent := reportCalls(o, own, external, frames.NewExternalFunctionCallCancelFrame("toolu_1", "get_weather"))

	if got, want := messageTypes(sent), []string{TypeLLMFunctionCallStop}; !slices.Equal(got, want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	if !dataOf[LLMFunctionCallStoppedData](t, sent[0]).Canceled {
		t.Error("the stop was not reported as canceled")
	}
}

func TestTheReportLevelAppliesByTheExternalCallsOwnName(t *testing.T) {
	o := callObserver(map[string]FunctionCallReportLevel{"*": ReportFull, "get_weather": ReportName})

	sent := reportCalls(o, externalCall("in_progress"))

	d := dataOf[LLMFunctionCallData](t, sent[0])
	if d.FunctionName != "get_weather" {
		t.Errorf("named %q, want get_weather", d.FunctionName)
	}
	if d.Arguments != nil {
		t.Errorf("carries arguments %s, want none", d.Arguments)
	}
}
