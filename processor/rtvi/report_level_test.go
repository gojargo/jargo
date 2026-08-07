package rtvi_test

import (
	"encoding/json"
	"slices"
	"testing"
	"time"

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
		map[string]rtvi.FunctionCallReportLevel{"*": rtvi.ReportFull}, nil)
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
	msgs := observerHarness(t, levels(rtvi.ReportName), rtvi.NewConfigureObserverFrame(nil, nil), weatherCall())

	if len(msgs) != 1 {
		t.Fatalf("expected one message, got %+v", msgs)
	}
	if d := callData(t, msgs, 0); d.FunctionName != "get_weather" {
		t.Fatalf("the level should be unchanged, got %+v", d)
	}
}

// TestObserverVADUserSpeaking checks the raw VAD speaking signal is withheld by
// default and reported once asked for, at runtime.
func TestObserverVADUserSpeaking(t *testing.T) {
	speaking := []frames.Frame{
		frames.NewVADUserStartedSpeakingFrame(0),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Time{}),
	}

	if msgs := observerHarness(t, rtvi.DefaultObserverParams(), speaking...); len(msgs) != 0 {
		t.Fatalf("the raw VAD signal is off by default, got %+v", msgs)
	}

	on := true
	queue := append([]frames.Frame{rtvi.NewConfigureObserverFrame(nil, &on)}, speaking...)
	msgs := observerHarness(t, rtvi.DefaultObserverParams(), queue...)
	if len(msgs) != 2 ||
		msgs[0].Type != rtvi.TypeVADUserStarted || msgs[1].Type != rtvi.TypeVADUserStopped {
		t.Fatalf("expected both raw VAD messages, got %+v", msgs)
	}
}

// TestObserverReportsTheWholeCallLifecycle checks each stage of a tool call is
// reported: the model asking for it, it beginning, and it finishing, whether it
// finished with a result or was canceled.
func TestObserverReportsTheWholeCallLifecycle(t *testing.T) {
	started := frames.NewFunctionCallsStartedFrame([]frames.ToolCall{
		{ID: "call-1", Name: "get_weather"},
		{ID: "call-2", Name: "get_restaurants"},
	})
	result := frames.NewFunctionCallResultFrame("call-1", "get_weather", nil, "sunny")
	cancel := frames.NewFunctionCallCancelFrame("call-2", "get_restaurants")

	msgs := observerHarness(t, levels(rtvi.ReportFull), started, weatherCall(), result, cancel)

	var types []string
	for _, m := range msgs {
		types = append(types, m.Type)
	}
	want := []string{
		// One started message per call in the batch.
		rtvi.TypeLLMFunctionCallStart, rtvi.TypeLLMFunctionCallStart,
		rtvi.TypeLLMFunctionCall,
		rtvi.TypeLLMFunctionCallStop, rtvi.TypeLLMFunctionCallStop,
	}
	if !slices.Equal(types, want) {
		t.Fatalf("got %v, want %v", types, want)
	}

	if d, ok := msgs[0].Data.(rtvi.LLMFunctionCallStartData); !ok || d.FunctionName != "get_weather" {
		t.Fatalf("unexpected started data: %+v", msgs[0].Data)
	}
	done, ok := msgs[3].Data.(rtvi.LLMFunctionCallStoppedData)
	if !ok || done.Canceled || done.Result != "sunny" {
		t.Fatalf("a completed call should report its result: %+v", msgs[3].Data)
	}
	canceled, ok := msgs[4].Data.(rtvi.LLMFunctionCallStoppedData)
	if !ok || !canceled.Canceled || canceled.Result != "" {
		t.Fatalf("a canceled call should report no result: %+v", msgs[4].Data)
	}
}

// TestObserverDisabledFunctionIsSilentThroughout checks a disabled function
// reports nothing at any stage, not even that something happened.
func TestObserverDisabledFunctionIsSilentThroughout(t *testing.T) {
	started := frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: "call-1", Name: "get_weather"}})
	result := frames.NewFunctionCallResultFrame("call-1", "get_weather", nil, "sunny")
	cancel := frames.NewFunctionCallCancelFrame("call-1", "get_weather")

	msgs := observerHarness(t, levels(rtvi.ReportDisabled), started, weatherCall(), result, cancel)
	if len(msgs) != 0 {
		t.Fatalf("expected no messages, got %+v", msgs)
	}
}
