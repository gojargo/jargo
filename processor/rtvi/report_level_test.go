package rtvi_test

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
)

// observerHarness runs frames through an observer built with params and returns
// the RTVI messages it sends.
func observerHarness(t *testing.T, params rtvi.ObserverParams, queue ...frames.Frame) []rtvi.Message {
	t.Helper()
	out := make(chan rtvi.Message, 16)
	proc := rtvi.NewProcessor()
	task := pipeline.NewTask(pipeline.New(proc), pipeline.TaskParams{
		Observers: []pipeline.Observer{rtvi.NewObserverWithParams(proc, params)},
		OnReachedDownstream: func(f frames.Frame) {
			if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
				if msg, ok := m.Message.(rtvi.Message); ok {
					out <- msg
				}
			}
		},
	})

	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()
	for _, f := range queue {
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	close(out)

	var msgs []rtvi.Message
	for msg := range out {
		msgs = append(msgs, msg)
	}
	return msgs
}

// levels builds the params for one report level applied to every function.
func levels(l rtvi.FunctionCallReportLevel) rtvi.ObserverParams {
	return rtvi.ObserverParams{
		FunctionCallReportLevel: map[string]rtvi.FunctionCallReportLevel{"*": l},
	}
}

// weatherCall is a tool call with arguments, to check what each level exposes.
func weatherCall() frames.Frame {
	return frames.NewFunctionCallInProgressFrame(
		"call-1", "get_weather", json.RawMessage(`{"city":"Paris"}`), true, "g1")
}

// callData is the function-call payload of the nth message.
func callData(t *testing.T, msgs []rtvi.Message, n int) rtvi.LLMFunctionCallData {
	t.Helper()
	d, ok := msgs[n].Data.(rtvi.LLMFunctionCallData)
	if !ok {
		t.Fatalf("message %d: unexpected payload type %T", n, msgs[n].Data)
	}
	return d
}

// TestObserverFunctionCallReportLevel checks each level exposes exactly what it
// promises: the id alone by default, then the name, then the arguments too.
func TestObserverFunctionCallReportLevel(t *testing.T) {
	cases := []struct {
		name     string
		params   rtvi.ObserverParams
		wantMsg  bool
		wantName string
		wantArgs string
	}{
		{name: "default withholds everything but the id", params: rtvi.DefaultObserverParams(), wantMsg: true},
		{name: "none withholds everything but the id", params: levels(rtvi.ReportNone), wantMsg: true},
		{name: "name adds the function name", params: levels(rtvi.ReportName), wantMsg: true, wantName: "get_weather"},
		{
			name: "full adds the arguments", params: levels(rtvi.ReportFull),
			wantMsg: true, wantName: "get_weather", wantArgs: `{"city":"Paris"}`,
		},
		{name: "disabled emits no message at all", params: levels(rtvi.ReportDisabled)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msgs := observerHarness(t, tc.params, weatherCall())
			if !tc.wantMsg {
				if len(msgs) != 0 {
					t.Fatalf("expected no message, got %+v", msgs)
				}
				return
			}
			if len(msgs) != 1 || msgs[0].Type != rtvi.TypeLLMFunctionCall {
				t.Fatalf("expected one llm-function-call-in-progress, got %+v", msgs)
			}
			d := callData(t, msgs, 0)
			if d.ToolCallID != "call-1" {
				t.Fatalf("the tool call id is always reported, got %q", d.ToolCallID)
			}
			if d.FunctionName != tc.wantName {
				t.Fatalf("function name: got %q, want %q", d.FunctionName, tc.wantName)
			}
			if string(d.Arguments) != tc.wantArgs {
				t.Fatalf("arguments: got %q, want %q", d.Arguments, tc.wantArgs)
			}
		})
	}
}

// TestObserverReportLevelPerFunction checks a function's own entry wins over the
// "*" default, so one tool can be exposed without exposing the rest.
func TestObserverReportLevelPerFunction(t *testing.T) {
	params := rtvi.ObserverParams{
		FunctionCallReportLevel: map[string]rtvi.FunctionCallReportLevel{
			"*":           rtvi.ReportNone,
			"get_weather": rtvi.ReportFull,
		},
	}
	secret := frames.NewFunctionCallInProgressFrame("call-2", "charge_card", nil, true, "g1")
	msgs := observerHarness(t, params, weatherCall(), secret)

	if len(msgs) != 2 {
		t.Fatalf("expected two messages, got %+v", msgs)
	}
	if d := callData(t, msgs, 0); d.FunctionName != "get_weather" {
		t.Fatalf("the listed function should be reported in full, got %+v", d)
	}
	if d := callData(t, msgs, 1); d.FunctionName != "" {
		t.Fatalf("an unlisted function should fall back to the default, got %+v", d)
	}
}

// TestObserverConfigureRaisesLevelAtRuntime checks a ConfigureObserverFrame
// elevates a running observer: bots default to the secure level, and only a
// trusted, server-side source raises it.
func TestObserverConfigureRaisesLevelAtRuntime(t *testing.T) {
	configure := rtvi.NewConfigureObserverFrame(
		map[string]rtvi.FunctionCallReportLevel{"*": rtvi.ReportFull})
	msgs := observerHarness(t, rtvi.DefaultObserverParams(), weatherCall(), configure, weatherCall())

	if len(msgs) != 2 {
		t.Fatalf("expected two messages, got %+v", msgs)
	}
	if d := callData(t, msgs, 0); d.FunctionName != "" {
		t.Fatalf("the call before the config frame should be withheld, got %+v", d)
	}
	d := callData(t, msgs, 1)
	if d.FunctionName != "get_weather" || string(d.Arguments) != `{"city":"Paris"}` {
		t.Fatalf("the call after the config frame should be reported in full, got %+v", d)
	}
}

// TestObserverConfigureNilLeavesLevelUnchanged checks an unset field leaves the
// observer's current configuration alone.
func TestObserverConfigureNilLeavesLevelUnchanged(t *testing.T) {
	msgs := observerHarness(t, levels(rtvi.ReportName), rtvi.NewConfigureObserverFrame(nil), weatherCall())

	if len(msgs) != 1 {
		t.Fatalf("expected one message, got %+v", msgs)
	}
	if d := callData(t, msgs, 0); d.FunctionName != "get_weather" {
		t.Fatalf("the level should be unchanged, got %+v", d)
	}
}
