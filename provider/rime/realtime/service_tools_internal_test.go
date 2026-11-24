package realtime

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// gatedLookup is a tool whose result waits until the returned function is
// called.
func gatedLookup(t *testing.T) (llm.FunctionCallHandler, func()) {
	t.Helper()
	finished := make(chan struct{})
	var once bool
	finish := func() {
		if !once {
			once = true
			close(finished)
		}
	}
	t.Cleanup(finish)
	return func(ctx context.Context, p llm.FunctionCallParams) error {
		select {
		case <-finished:
		case <-ctx.Done():
			return ctx.Err()
		}
		return p.Result(ctx, `{"status": "paid"}`, nil)
	}, finish
}

// withTools configures the tools a conversation advertises.
func withTools(tools ...frames.Tool) func(*Config) {
	return func(c *Config) { c.Settings.Tools = toolset(tools...) }
}

func (h *harness) waitCall(id string) {
	h.t.Helper()
	until(h.t, "the call "+id, func() bool { _, ok := h.callOf(id); return ok })
}

func (h *harness) waitResponse(ref string) {
	h.t.Helper()
	until(h.t, "the reply "+ref, func() bool { _, ok := h.responseOf(ref); return ok })
}

func TestAsyncFinalResultEnvelopeDoesNotDuplicateProviderHistory(t *testing.T) {
	handler, finish := gatedLookup(t)
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
	ref := h.peer.start(nil, nil)
	h.peer.emitCall(ref, "call-1", "lookup")
	h.waitCall("call-1")
	h.peer.complete(ref)
	finish()
	h.answer(h.peer.nextReply(), "Async answer", 480)
	until(t, "the answer", func() bool { return h.hasMessage(frames.RoleAssistant, "Async answer") })
	if n := len(h.peer.sent(kindItemCreate)); n != 1 || !h.usable() {
		t.Errorf("%d items, usable = %v", n, h.usable())
	}
}

func TestLaterContextFramesDoNotReimportHistoryOrReply(t *testing.T) {
	h := converse(t, harnessOptions{})
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{itemID: "message-1", text: "First sentence."})
	h.peer.message(ref, messageOptions{itemID: "message-2", text: "Second sentence."})
	h.peer.complete(ref)
	until(t, "the reply", func() bool {
		return slices.ContainsFunc(h.convo.Messages(), func(m frames.Message) bool {
			return strings.Contains(m.StandardText(), "Second sentence.")
		})
	})
	// A run asked for mid-conversation sends the conversation downstream again.
	h.queue(frames.NewLLMRunFrame())
	time.Sleep(50 * time.Millisecond)
	if len(h.peer.sent(kindItemCreate)) > 0 || len(h.peer.sent(kindItemDelete)) > 0 {
		t.Error("the history was edited")
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestLateTranscriptIsFlushedBeforeTheAssistantRecord(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.userSpeaks()
	ref := h.peer.start(nil, []string{"speech-1"})
	h.peer.say(ref, "Answer")
	h.peer.emit("conversation.item.input_audio_transcription.completed",
		object{"item_id": "speech-1", "transcript": "Late question"})
	h.peer.complete(ref)
	until(t, "both messages", func() bool { return len(h.convo.Messages()) == 2 })
	messages := h.convo.Messages()
	if messages[0].Role != frames.RoleUser || messages[0].StandardText() != "Late question" ||
		messages[1].Role != frames.RoleAssistant || messages[1].StandardText() != "Answer" {
		t.Errorf("messages = %+v", messages)
	}
	if len(h.peer.sent(kindItemCreate)) > 0 {
		t.Error("the history was edited")
	}
}

// asyncPayload parses an item's text as the async tool envelope.
func asyncPayload(t *testing.T, item object) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal([]byte(item["text"].(string)), &payload); err != nil {
		t.Fatalf("item %v carries no envelope: %v", item, err)
	}
	return payload
}

func TestSilentProgressKeepsTheCallOpenUntilTheFinalResult(t *testing.T) {
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		if err := p.Result(ctx, `"Working"`, &frames.FunctionCallResultProperties{
			IsFinal: ptr(false), RunLLM: ptr(false),
		}); err != nil {
			return err
		}
		return p.Result(ctx, `"Done"`, nil)
	}
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
	ref := h.peer.start(nil, nil)
	h.peer.emitCall(ref, "call-1", "lookup")
	h.peer.complete(ref)
	continuation := h.peer.nextReply()
	if replyCause(continuation) != "tool_continuation" {
		t.Errorf("cause = %v", replyCause(continuation))
	}
	h.answer(continuation, "Hello", 480)
	items := h.peer.createdItems()
	var kinds []any
	for _, item := range items {
		kinds = append(kinds, item["type"])
	}
	if !slices.Equal(kinds, []any{"message", "function_call_output", "message"}) {
		t.Fatalf("items = %v", kinds)
	}
	progress := asyncPayload(t, items[0])
	if items[0]["role"] != "system" || progress["tool_call_id"] != "call-1" || progress["status"] != "running" ||
		progress["result"] != `"Working"` {
		t.Errorf("progress = %v", items[0])
	}
	if items[1]["call_id"] != "call-1" || items[1]["output"] != `"Done"` {
		t.Errorf("output = %v", items[1])
	}
	// Prism files the output beside the call, so the result follows the
	// progress too.
	settled := asyncPayload(t, items[2])
	if items[2]["role"] != "system" || settled["tool_call_id"] != "call-1" || settled["status"] != "finished" ||
		settled["result"] != `"Done"` {
		t.Errorf("settled = %v", items[2])
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 2 || h.failed() || !h.usable() {
		t.Errorf("%d replies requested, errors = %v, usable = %v", n, h.errors(), h.usable())
	}
}

func TestToolResultWaitsForTheBotToStopSpeaking(t *testing.T) {
	handler, finish := gatedLookup(t)
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	release := h.output.holdAudio(t)
	parent := h.peer.start(nil, nil)
	h.peer.message(parent, messageOptions{text: "Checking", samples: 2400})
	h.peer.emitCall(parent, "call-1", "lookup")
	h.peer.complete(parent)
	until(t, "the bot to speak", func() bool { return has[*frames.BotStartedSpeakingFrame](h.down) })
	finish()
	until(t, "the result", func() bool { return len(h.peer.sent(kindItemCreate)) == 1 })
	time.Sleep(100 * time.Millisecond)
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested while the bot speaks, want 1", n)
	}
	release()
	continuation := h.peer.nextReply()
	if replyCause(continuation) != "tool_continuation" {
		t.Errorf("cause = %v", replyCause(continuation))
	}
	h.answer(continuation, "Hello", 480)
	if !h.usable() {
		t.Error("unusable")
	}
}

func TestToolResultFramesKeepTheirReplyPolicy(t *testing.T) {
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, `{"status": "paid"}`, &frames.FunctionCallResultProperties{RunLLM: ptr(false)})
	}
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	parent := h.peer.start(nil, nil)
	h.peer.emitCall(parent, "call-1", "lookup")
	h.peer.complete(parent)
	until(t, "the result", func() bool { return has[*frames.FunctionCallResultFrame](h.down) })
	result := captured[*frames.FunctionCallResultFrame](h.down)[0]
	if result.Properties == nil || result.Properties.RunLLM == nil || *result.Properties.RunLLM {
		t.Errorf("properties = %+v, want no reply", result.Properties)
	}
	time.Sleep(50 * time.Millisecond)
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
	held := func() bool {
		h.svc.mu.Lock()
		defer h.svc.mu.Unlock()
		value := h.svc.conversation.responses[parent]
		return value != nil && value.hold != nil
	}
	if !held() {
		t.Error("the reply is not held for the result")
	}
	// A later reply, such as the user's next turn, takes the result in.
	h.peer.complete(h.peer.start(nil, nil))
	until(t, "the reply let go", func() bool { return !held() })
	if !h.usable() {
		t.Error("unusable")
	}
}

func TestToolResultDeadlineIsQuietWhileTheToolRuns(t *testing.T) {
	for _, running := range []bool{true, false} {
		t.Run(map[bool]string{true: "running", false: "finished"}[running], func(t *testing.T) {
			handler, finish := gatedLookup(t)
			h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
			h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
			parent := h.peer.start(nil, nil)
			h.peer.emitCall(parent, "call-1", "lookup")
			h.peer.complete(parent)
			h.waitCall("call-1")
			if !running {
				finish()
				until(t, "the result submitted", func() bool {
					call, _ := h.callOf("call-1")
					h.svc.mu.Lock()
					defer h.svc.mu.Unlock()
					return call.submitted
				})
			}
			h.peer.emit("error", object{"error": object{
				"code":    "tool_result_timeout",
				"message": "tool results did not arrive in time",
				"scope":   "tool_roundtrip",
				"owner":   object{"kind": "response", "response_id": parent, "call_ids": []string{"call-1"}},
			}})
			time.Sleep(50 * time.Millisecond)
			if h.failed() == running {
				t.Errorf("errors = %v, want them only once the tool finished", h.errors())
			}
			if running {
				finish()
				h.answer(h.peer.nextReply(), "Hello", 480)
			}
			if !h.usable() {
				t.Error("unusable")
			}
		})
	}
}

func TestAsyncResultAfterItsRoundClosedIsReported(t *testing.T) {
	for _, movedOn := range []bool{false, true} {
		t.Run(map[bool]string{true: "moved on", false: "same round"}[movedOn], func(t *testing.T) {
			handler, finish := gatedLookup(t)
			h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
			h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
			parent := h.peer.start(nil, nil)
			h.peer.emitCall(parent, "call-1", "lookup")
			h.peer.complete(parent)
			h.waitCall("call-1")
			if movedOn {
				turn := h.peer.start(nil, nil)
				h.peer.complete(turn)
				h.waitResponse(turn)
			}
			finish()
			request := h.peer.nextReply()
			var outputs []any
			for _, item := range h.peer.createdItems() {
				outputs = append(outputs, item["output"])
			}
			if !slices.Equal(outputs, []any{`{"status": "paid"}`}) {
				t.Errorf("outputs = %v", outputs)
			}
			if movedOn {
				if replyCause(request) != "proactive" || request["response"].(object)["prism_call_id"] != "call-1" {
					t.Errorf("request = %v, want a report of the call", request["response"])
				}
			} else if replyCause(request) != "tool_continuation" {
				t.Errorf("cause = %v", replyCause(request))
			}
			h.answer(request, "Hello", 480)
			time.Sleep(100 * time.Millisecond)
			if n := len(h.peer.sent(kindResponseNew)); n != 2 || !h.usable() {
				t.Errorf("%d replies requested, usable = %v", n, h.usable())
			}
		})
	}
}

// lateResult closes a running asynchronous call's round with a later turn,
// then finishes the call.
func (h *harness) lateResult(finish func()) {
	parent := h.peer.start(nil, nil)
	h.peer.emitCall(parent, "call-1", "lookup")
	h.peer.complete(parent)
	h.waitCall("call-1")
	turn := h.peer.start(nil, nil)
	h.peer.complete(turn)
	h.waitResponse(turn)
	finish()
}

func TestLateResultReportSurvivesAnInterruption(t *testing.T) {
	handler, finish := gatedLookup(t)
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
	open := h.peer.gate(kindResponseNew)
	h.lateResult(finish)
	until(t, "the report requested", func() bool { return h.requestsPending() > 0 })
	h.queue(frames.NewInterruptionFrame())
	time.Sleep(50 * time.Millisecond)
	open()
	request := h.peer.nextReply()
	if request["response"].(object)["prism_call_id"] != "call-1" {
		t.Errorf("request = %v", request["response"])
	}
	report := h.peer.start(request, nil)
	h.peer.say(report, "It is paid")
	h.peer.complete(report)
	// An interrupted reply would be stopped, and its words left out of the
	// conversation.
	until(t, "the report", func() bool { return h.hasMessage(frames.RoleAssistant, "It is paid") })
	if value, _ := h.responseOf(report); value == nil || value.stopped || !h.usable() {
		t.Errorf("the report was stopped, or the service is unusable")
	}
}

func TestRefusedLateResultReportIsRequestedOnceMore(t *testing.T) {
	handler, finish := gatedLookup(t)
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
	h.peer.setRefusal(kindResponseNew, "nudge_superseded")
	h.lateResult(finish)
	until(t, "the report asked again", func() bool { return len(h.peer.sent(kindResponseNew)) == 3 })
	sent := h.peer.sent(kindResponseNew)
	retry := sent[len(sent)-1]
	if retry["response"].(object)["prism_call_id"] != "call-1" {
		t.Errorf("retry = %v", retry["response"])
	}
	h.answer(h.peer.nextReply(), "Hello", 480)
	if h.failed() || !h.usable() {
		t.Errorf("errors = %v, usable = %v", h.errors(), h.usable())
	}
}

func TestServerPlaybackAdmissionWaitDoesNotBlockMicrophoneFrames(t *testing.T) {
	h := converse(t, harnessOptions{})
	ref := h.peer.start(nil, nil)
	h.peer.say(ref, "Waiting for server playback estimate")
	h.peer.end(ref, "completed", false, nil)
	until(t, "the reply in the conversation", func() bool { return len(h.convo.Messages()) > 0 })
	h.queue(appendUser("Waiting", true), frames.NewInputAudioRawFrame(make([]byte, 960), 24000, 1))
	h.peer.next("input_audio_buffer.append")
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested before Prism was ready, want 1", n)
	}
	h.peer.emit("prism.typed_input.ready", nil)
	h.answer(nil, "Hello", 480)
	if !h.usable() {
		t.Error("unusable")
	}
}

// runService runs the service alone in a pipeline with frames, as a test of a
// processor does, and returns the frames that traveled upstream.
func runService(t *testing.T, svc *Service, wait time.Duration, fs ...frames.Frame) []frames.Frame {
	t.Helper()
	up := newCapture(processor.Upstream)
	w := pipeline.NewWorker(pipeline.New(up, svc), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	w.QueueFrames(fs)
	time.Sleep(wait)
	w.QueueFrame(frames.NewEndFrame())
	select {
	case <-done:
	case <-time.After(waitTimeout):
		t.Error("the pipeline did not end")
		w.Cancel(context.Background(), "")
		<-done
	}
	return up.frames()
}

func errorFrames(fs []frames.Frame) []*frames.ErrorFrame {
	var out []*frames.ErrorFrame
	for _, f := range fs {
		if e, ok := f.(*frames.ErrorFrame); ok {
			out = append(out, e)
		}
	}
	return out
}

func TestFailedStartupClosesSessionButLeavesTheEndpointUsable(t *testing.T) {
	p := newPeer(t, false)
	p.setRefusal(kindSessionUpdate, "invalid_request")
	svc := newService(p, nil)
	up := runService(t, svc, 50*time.Millisecond, frames.NewLLMContextFrame(frames.NewLLMContext("")))
	if svc.Usable() || len(errorFrames(up)) == 0 {
		t.Errorf("usable = %v, errors = %v, want a failed service", svc.Usable(), errorFrames(up))
	}
	options := connectOptions{apiKey: "local-peer", endpoint: p.endpoint(), interruptOnSpeech: true,
		timeouts: Timeouts{}.withDefaults()}
	sess, err := connect(t.Context(), options)
	if err != nil {
		t.Fatalf("connect() = %v", err)
	}
	defer sess.close()
	if info, _ := sess.sessionInfo(); info.RequestID != "actual-peer-request" {
		t.Errorf("request id = %q", info.RequestID)
	}
}

func TestCompletedToolHistoryImportDoesNotExecuteTools(t *testing.T) {
	calls := &counter{}
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		calls.add()
		return p.Result(ctx, "Unexpected", nil)
	}
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{
		{Role: frames.RoleAssistant, ToolCalls: []frames.ToolCall{{ID: "imported-call", Name: "lookup", Args: json.RawMessage("{}")}}},
		{Role: frames.RoleUser, ToolResults: []frames.ToolResult{{ID: "imported-call", Name: "lookup", Content: "Completed result"}}},
	})
	convo.SetTools([]frames.Tool{lookup("lookup", handler)})
	h := converse(t, harnessOptions{convo: convo})
	if calls.count() != 0 || len(h.peer.sent(kindItemCreate)) != 2 {
		t.Errorf("%d calls, %d items", calls.count(), len(h.peer.sent(kindItemCreate)))
	}
	time.Sleep(50 * time.Millisecond)
	if calls.count() != 0 {
		t.Error("an imported call ran")
	}
}

func TestUnexpectedConnectionCloseRequiresExplicitRecovery(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.peer.closeSocket()
	until(t, "an error", h.failed)
	if h.usable() {
		t.Error("the service is still usable")
	}
	if n := len(h.peer.sent(kindSessionUpdate)); n != 1 {
		t.Errorf("%d session updates, want 1", n)
	}
}
