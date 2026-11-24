package realtime

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

func TestRefusedReplyAllowsNextTurn(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.peer.setRefusal(kindResponseNew, "invalid_request")
	h.queue(appendUser("Refused", true))
	until(t, "an error", h.failed)
	if !h.usable() {
		t.Fatal("unusable")
	}
	h.queue(appendUser("Accepted", true))
	h.answer(nil, "Answered", 480)
	until(t, "the answer", func() bool { return h.hasMessage(frames.RoleAssistant, "Answered") })
	if texts := h.peer.historyTexts(); !slices.Equal(texts, []string{"Refused", "Accepted"}) {
		t.Errorf("history = %v", texts)
	}
}

// TestPromptClearAndFillerSurvivesToolHandoff waits for the last update to
// reach the peer before it asserts on it.
func TestPromptClearAndFillerSurvivesToolHandoff(t *testing.T) {
	tool := lookup("lookup", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, `"done"`, nil)
	})
	h := converse(t, harnessOptions{config: func(c *Config) {
		c.Settings.SystemInstruction = settings.Set("Original")
		c.Settings.Tools = toolset(tool)
		c.Settings.ToolFillers = settings.Set(map[string]Filler{"lookup": {Type: "text", Text: new("Checking")}})
	}})
	h.queue(
		update(instruction("")), frames.NewLLMSetToolsFrame([]frames.Tool{}), frames.NewLLMSetToolsFrame([]frames.Tool{tool}),
	)
	until(t, "the updates", func() bool { return len(h.peer.sent(kindSessionUpdate)) == 4 })
	if got := h.peer.setting("instructions"); got != "" {
		t.Errorf("instructions = %q", got)
	}
	tools, _ := h.peer.setting(keyTools).([]any)
	if len(tools) == 0 || at(tools[0], "prism_filler", "text") != "Checking" {
		t.Errorf("tools = %v, want the filler", tools)
	}
}

func TestToolSuppressionAcknowledgesThenRestores(t *testing.T) {
	calls := &counter{}
	tool := lookup("lookup", func(ctx context.Context, p llm.FunctionCallParams) error {
		calls.add()
		return p.Result(ctx, `"done"`, nil)
	})
	h := converse(t, harnessOptions{config: withTools(tool)})
	open := h.peer.gate(kindSessionUpdate)
	h.queue(frames.NewLLMSetToolChoiceFrame(frames.ToolChoiceNone), appendUser("Hello", true))
	until(t, "the suppression sent", func() bool { return len(h.peer.sent(kindSessionUpdate)) == 2 })
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested before the acknowledgment, want 1", n)
	}
	open()
	request := h.peer.nextReply()
	if tools, _ := h.peer.setting(keyTools).([]any); len(tools) != 0 {
		t.Errorf("tools = %v, want none while the reply is asked for", tools)
	}
	ref := h.peer.start(request, nil)
	until(t, "the tools restored", func() bool { return len(h.peer.toolNames()) > 0 })
	h.peer.emitCall(ref, "lookup-1", "lookup")
	until(t, "the service to fail", func() bool { return !h.usable() })
	if calls.count() != 0 {
		t.Error("the call of a reply without tools ran")
	}
}

func TestDiscardReceiptOrdersNextAdmissionAndSuppressesSpeechAudio(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) { c.Playback = PlaybackDiscard }})
	until(t, "the opening receipt", func() bool { return len(h.peer.sent("prism.playback.drained")) == 1 })
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{text: "Discarded speech", samples: 2400})
	h.peer.end(ref, "completed", false, nil)
	until(t, "the receipt", func() bool { return len(h.peer.sent("prism.playback.drained")) == 2 })
	if h.output.written() > 0 {
		t.Error("audio was played")
	}
	until(t, "the text", func() bool {
		return slices.ContainsFunc(captured[*frames.TTSTextFrame](h.down), func(f *frames.TTSTextFrame) bool {
			return f.Text == "Discarded speech"
		})
	})
	h.queue(appendUser("Next", true))
	h.answer(nil, "Hello", 480)
	until(t, "the third receipt", func() bool { return len(h.peer.sent("prism.playback.drained")) == 3 })
	if h.output.written() > 0 || !h.usable() {
		t.Errorf("audio played, or unusable")
	}
}

func TestPendingToolSurvivesMoreThan128CompletedResponses(t *testing.T) {
	calls := &counter{}
	release := make(chan struct{})
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		calls.add()
		select {
		case <-release:
		case <-ctx.Done():
			return ctx.Err()
		}
		return p.Result(ctx, "false", nil)
	}
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
	parent := h.peer.start(nil, nil)
	h.peer.emitCall(parent, "lookup-1", "lookup")
	h.peer.complete(parent)
	until(t, "the call", func() bool { return calls.count() > 0 })
	for range 130 {
		ends := len(captured[*frames.LLMFullResponseEndFrame](h.down))
		h.queue(appendUser("Later turn", true))
		h.answer(nil, "Later turn", 0)
		// The service takes the reply's events in before the next append, as
		// they arrived first.
		until(t, "the reply's end", func() bool { return len(captured[*frames.LLMFullResponseEndFrame](h.down)) > ends })
	}
	close(release)
	request := h.peer.nextReply()
	if replyCause(request) != "proactive" {
		t.Errorf("cause = %v", replyCause(request))
	}
	h.answer(request, "Result false", 0)
	var outputs []any
	for _, item := range h.peer.createdItems() {
		if item["type"] == "function_call_output" {
			outputs = append(outputs, item["output"])
		}
	}
	if calls.count() != 1 || !slices.Equal(outputs, []any{"false"}) || !h.usable() {
		t.Errorf("%d calls, outputs = %v, usable = %v", calls.count(), outputs, h.usable())
	}
}

func TestStartupHTTPRefusalPolicy(t *testing.T) {
	for _, status := range []int{408, 429, 500, 502, 503, 504, 401} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			p := newPeer(t, false)
			zero := "0"
			p.refuseUpgrade = func(attempt int) (int, *string) {
				if attempt == 1 || status == 401 {
					return status, &zero
				}
				return 0, nil
			}
			svc := newService(p, func(c *Config) {
				c.Startup = StartupParams{RetryInterval: new(time.Duration(0)), TotalTimeout: 2 * time.Second}
			})
			svc.shutdownTimeout = 100 * time.Millisecond
			up := runService(t, svc, 150*time.Millisecond, frames.NewLLMContextFrame(frames.NewLLMContext("")))
			want := 2
			if status == 401 {
				want = 1
			}
			if n := len(p.attemptTimes()); n != want {
				t.Errorf("%d attempts, want %d", n, want)
			}
			if failed := len(errorFrames(up)) > 0; failed != (status == 401) {
				t.Errorf("errors = %v", errorFrames(up))
			}
		})
	}
}

func TestUsageOnceForTerminalResponseWithRawDiagnostics(t *testing.T) {
	for _, status := range []string{"completed", "cancelled"} { //nolint:misspell // the wire's spelling
		t.Run(status, func(t *testing.T) {
			var raw func() []ResponseEnded
			h := converse(t, harnessOptions{
				params: pipeline.Params{EnableMetrics: true, EnableUsageMetrics: true},
				setup:  func(s *Service) { raw = observe[ResponseEnded](s) },
			})
			ended := len(raw())
			ref := h.peer.start(nil, nil)
			usage := object{"input_tokens": 12, "output_tokens": 4, "total_tokens": 16, "context_window": 32768}
			h.peer.end(ref, status, true, usage)
			usages := func() []frames.LLMUsageMetricsData {
				var out []frames.LLMUsageMetricsData
				for _, f := range captured[*frames.MetricsFrame](h.down) {
					for _, d := range f.Data {
						if u, ok := d.(frames.LLMUsageMetricsData); ok {
							out = append(out, u)
						}
					}
				}
				return out
			}
			until(t, "the usage", func() bool { return len(usages()) > 0 })
			time.Sleep(20 * time.Millisecond)
			if got := usages(); len(got) != 1 || got[0].Value.TotalTokens != 16 {
				t.Errorf("usage = %+v, want one report of 16 tokens", got)
			}
			until(t, "the raw event", func() bool { return len(raw()) > ended })
			if events := raw(); events[ended].Usage == nil || events[ended].Usage.ContextWindow != 32768 {
				t.Errorf("raw usage = %+v", events[ended].Usage)
			}
			h.peer.complete(h.peer.start(nil, nil))
			until(t, "the second raw event", func() bool { return len(raw()) == ended+2 })
			if events := raw(); events[ended+1].Usage != nil {
				t.Errorf("usage = %+v, want none", events[ended+1].Usage)
			}
		})
	}
}

func TestProgressThenFinalRespectsReplyPolicy(t *testing.T) {
	for _, finalRun := range []bool{true, false} {
		for _, progressRun := range []bool{true, false} {
			t.Run(fmt.Sprintf("final=%v,progress=%v", finalRun, progressRun), func(t *testing.T) {
				testProgressThenFinal(t, finalRun, progressRun)
			})
		}
	}
}

func testProgressThenFinal(t *testing.T, finalRun, progressRun bool) {
	t.Helper()
	calls := &counter{}
	release := make(chan struct{})
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		calls.add()
		if err := p.Result(ctx, `"Working"`, &frames.FunctionCallResultProperties{
			IsFinal: new(false), RunLLM: new(progressRun),
		}); err != nil {
			return err
		}
		select {
		case <-release:
		case <-ctx.Done():
			return ctx.Err()
		}
		return p.Result(ctx, `"Finished"`, &frames.FunctionCallResultProperties{RunLLM: new(finalRun)})
	}
	h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
	parent := h.peer.start(nil, nil)
	h.peer.emitCall(parent, "lookup-1", "lookup")
	h.peer.complete(parent)
	until(t, "the progress", func() bool { return len(h.peer.sent(kindItemCreate)) == 1 })
	if progress := h.peer.createdItems()[0]; progress["type"] != "message" || progress["role"] != "system" {
		t.Errorf("progress = %v", progress)
	}
	if !progressRun {
		h.queue(appendUser("Answer while the lookup runs", true))
	}
	h.answer(nil, "Progress recorded", 480)
	close(release)
	writes := 3
	if !progressRun {
		writes = 4
	}
	until(t, "the final result", func() bool { return len(h.peer.sent(kindItemCreate)) == writes })
	items := h.peer.createdItems()
	final, settled := items[len(items)-2], items[len(items)-1]
	if final["type"] != "function_call_output" || final["call_id"] != "lookup-1" {
		t.Errorf("final = %v", final)
	}
	if settled["role"] != "system" || asyncPayload(t, settled)["status"] != "finished" {
		t.Errorf("settled = %v", settled)
	}
	if finalRun {
		// The progress reply superseded the tool round, so the result is
		// reported.
		request := h.peer.nextReply()
		if at(request, "response", "prism_call_id") != "lookup-1" {
			t.Errorf("request = %v", request["response"])
		}
		h.answer(request, "Final recorded", 480)
		until(t, "the report", func() bool { return h.hasMessage(frames.RoleAssistant, "Final recorded") })
		// The report answers the result; the aggregator's request adds no
		// reply.
		time.Sleep(100 * time.Millisecond)
		if n := len(h.peer.sent(kindResponseNew)); n != 3 {
			t.Errorf("%d replies requested, want 3", n)
		}
	} else {
		until(t, "the result in the conversation", func() bool {
			return slices.ContainsFunc(h.convo.Messages(), func(m frames.Message) bool {
				return strings.Contains(fmt.Sprint(m), "Finished")
			})
		})
		time.Sleep(50 * time.Millisecond)
		if n := len(h.peer.sent(kindResponseNew)); n != 2 {
			t.Errorf("%d replies requested, want 2", n)
		}
	}
	if calls.count() != 1 {
		t.Errorf("%d calls, want 1", calls.count())
	}
	// A later conversation frame does not write the asynchronous final a
	// second time.
	h.queue(frames.NewLLMContextFrame(h.convo))
	time.Sleep(50 * time.Millisecond)
	if n := len(h.peer.sent(kindItemCreate)); n != writes || h.failed() {
		t.Errorf("%d items, errors = %v", n, h.errors())
	}
}

// errSynthetic is the failure of a tool that fails.
var errSynthetic = errors.New("synthetic failure")

func TestToolFailureSettlesOnceWithoutRepeatExecution(t *testing.T) {
	t.Run("exception", func(t *testing.T) {
		calls := &counter{}
		handler := func(context.Context, llm.FunctionCallParams) error {
			calls.add()
			return errSynthetic
		}
		h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
		h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false))
		parent := h.peer.start(nil, nil)
		h.peer.emitCall(parent, "lookup-1", "lookup")
		h.peer.complete(parent)
		h.answer(nil, "Tool failed", 480)
		if calls.count() != 1 || !h.usable() {
			t.Errorf("%d calls, usable = %v", calls.count(), h.usable())
		}
	})
	t.Run("timeout", func(t *testing.T) {
		calls := &counter{}
		handler := func(ctx context.Context, p llm.FunctionCallParams) error {
			calls.add()
			if err := p.Result(ctx, `"Working"`, &frames.FunctionCallResultProperties{
				IsFinal: new(false), RunLLM: new(false),
			}); err != nil {
				return err
			}
			<-ctx.Done()
			return ctx.Err()
		}
		h := converse(t, harnessOptions{config: withTools(lookup("lookup", handler))})
		h.svc.RegisterFunction("lookup", handler, llm.WithCancelOnInterruption(false),
			llm.WithTimeout(50*time.Millisecond))
		parent := h.peer.start(nil, nil)
		h.peer.emitCall(parent, "lookup-1", "lookup")
		h.peer.complete(parent)
		until(t, "the call settled", func() bool {
			return slices.ContainsFunc(h.peer.createdItems(), func(item object) bool {
				text, _ := item["text"].(string)
				return strings.Contains(text, `"status":"finished"`) || strings.Contains(text, `"status": "finished"`)
			})
		})
		items := h.peer.createdItems()
		var outputs []any
		for _, item := range items {
			if item["type"] != "message" {
				outputs = append(outputs, item["output"])
			}
		}
		if !slices.Equal(outputs, []any{frames.ToolResultCanceled}) {
			t.Errorf("outputs = %v", outputs)
		}
		settled := asyncPayload(t, items[len(items)-1])
		if result, _ := settled["result"].(string); !strings.HasPrefix(result, frames.ToolResultCanceled+": this tool call was cancelled") { //nolint:misspell,lll // the message's spelling
			t.Errorf("settled = %v", settled)
		}
		if calls.count() != 1 || !h.usable() {
			t.Errorf("%d calls, usable = %v", calls.count(), h.usable())
		}
	})
}

func TestHistoryAcknowledgmentTimeoutIsTerminal(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) { c.Timeouts.Request = 50 * time.Millisecond }})
	h.peer.gate(kindItemCreate)
	h.queue(appendUser("Uncertain", true))
	until(t, "the service to fail", func() bool { return !h.usable() })
	until(t, "an error", h.failed)
	errs := h.errors()
	update, ok := errors.AsType[*HistoryUpdateError](errs[len(errs)-1].Err)
	if !ok || update.Status != "uncertain" {
		t.Errorf("error = %v, want an uncertain history update", errs[len(errs)-1].Err)
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestSettingsConfirmationSurvivesRejectedLabelUpdate(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.peer.setRefusal("session.labels", "invalid_request")
	delta := instruction("Confirmed")
	delta.ResponseMetadata = settings.Set(map[string]string{"stage": "next"})
	h.queue(update(delta))
	until(t, "the instruction", func() bool { return h.peer.setting("instructions") == "Confirmed" })
	until(t, "an error", h.failed)
	if info, _ := h.svc.SessionInfo(); info.ResponseMetadata["stage"] == "next" {
		t.Error("the refused labels are confirmed")
	}
	if !h.usable() {
		t.Error("unusable")
	}
}

func TestDiscardOrdersOutOfOrderCompletions(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) { c.Playback = PlaybackDiscard }})
	until(t, "the opening receipt", func() bool { return len(h.peer.sent("prism.playback.drained")) == 1 })
	first := h.peer.start(nil, nil)
	second := h.peer.start(nil, nil)
	h.peer.end(second, "completed", false, nil)
	time.Sleep(20 * time.Millisecond)
	if n := len(h.peer.sent("prism.playback.drained")); n != 1 {
		t.Errorf("%d receipts before the first reply ended, want 1", n)
	}
	h.peer.end(first, "completed", false, nil)
	until(t, "both receipts", func() bool { return len(h.peer.sent("prism.playback.drained")) == 3 })
	receipts := h.peer.sent("prism.playback.drained")
	if got := []any{receipts[1]["response_id"], receipts[2]["response_id"]}; !slices.Equal(got, []any{first, second}) {
		t.Errorf("receipts = %v, want %v then %v", got, first, second)
	}
}

func TestProtectedResponseLimitFailsWithoutEviction(t *testing.T) {
	h := converse(t, harnessOptions{})
	for range 129 {
		h.peer.start(nil, nil)
	}
	until(t, "the service to fail", func() bool { return !h.usable() })
	until(t, "an error", h.failed)
	if !slices.ContainsFunc(h.errors(), func(e *frames.ErrorFrame) bool {
		return strings.Contains(strings.ToLower(e.Error), "protected")
	}) {
		t.Errorf("errors = %v, want the protected limit", h.errors())
	}
}

func TestStartupRetryExhaustionAndDiagnostics(t *testing.T) {
	for _, retryAfter := range []*string{new("0"), new("0.02"), new("bad"), nil} {
		name := "none"
		if retryAfter != nil {
			name = *retryAfter
		}
		t.Run(name, func(t *testing.T) {
			p := newPeer(t, false)
			p.refuseUpgrade = func(int) (int, *string) { return 503, retryAfter }
			svc := newService(p, func(c *Config) {
				c.Startup = StartupParams{
					MaxRetries: new(1), RetryInterval: new(10 * time.Millisecond), TotalTimeout: 400 * time.Millisecond,
				}
			})
			svc.shutdownTimeout = 100 * time.Millisecond
			up := runService(t, svc, 120*time.Millisecond, frames.NewLLMContextFrame(frames.NewLLMContext("")))
			attempts := p.attemptTimes()
			if len(attempts) != 2 {
				t.Fatalf("%d attempts, want 2", len(attempts))
			}
			if gap := attempts[1].Sub(attempts[0]); gap < 10*time.Millisecond {
				t.Errorf("attempts %v apart, want at least 10ms", gap)
			}
			if len(errorFrames(up)) == 0 {
				t.Error("no error")
			}
		})
	}
}

func TestInterruptionBeforeCreationCancelsLateReply(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.queue(frames.NewLLMSetToolChoiceFrame(frames.ToolChoiceNone))
	h.queue(appendUser("Accepted late", true))
	request := h.peer.nextReply()
	h.queue(frames.NewInterruptionFrame())
	until(t, "the interruption", func() bool {
		h.svc.mu.Lock()
		defer h.svc.mu.Unlock()
		return h.svc.interruptEpoch == 1
	})
	ref := h.peer.start(request, nil)
	h.peer.message(ref, messageOptions{text: "Should not play", samples: 2400})
	until(t, "the cancellation", func() bool {
		return slices.ContainsFunc(h.peer.sent(kindResponseStop), func(e object) bool { return e["response_id"] == ref })
	})
	until(t, "the tools restored", func() bool { return len(h.peer.sent(kindSessionUpdate)) == 3 })
	if h.output.written() > 0 || !h.usable() {
		t.Errorf("audio played, or unusable")
	}
	if n := h.countMessage(frames.RoleUser, "Accepted late"); n != 1 {
		t.Errorf("the message is in the conversation %d times, want once", n)
	}
}

func TestRefusedFinalToolOutputEndsUnsettledConversation(t *testing.T) {
	tool := lookup("lookup", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, `"Final"`, nil)
	})
	h := converse(t, harnessOptions{config: withTools(tool)})
	h.peer.setRefusal(kindItemCreate, "invalid_request")
	parent := h.peer.start(nil, nil)
	h.peer.emitCall(parent, "lookup-1", "lookup")
	h.peer.complete(parent)
	until(t, "the service to fail", func() bool { return !h.usable() })
	until(t, "an error", h.failed)
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestSuppressionRestoreFailureIsTerminal(t *testing.T) {
	tool := lookup("lookup", func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, `"unused"`, nil)
	})
	h := converse(t, harnessOptions{config: withTools(tool)})
	h.queue(frames.NewLLMSetToolChoiceFrame(frames.ToolChoiceNone), appendUser("Hello", true))
	request := h.peer.nextReply()
	h.peer.setRefusal(kindSessionUpdate, "invalid_request")
	h.answer(request, "Hello", 480)
	until(t, "the service to fail", func() bool { return !h.usable() })
	until(t, "an error", h.failed)
	if tools, ok := h.peer.setting(keyTools).([]any); !ok || len(tools) != 0 {
		t.Errorf("tools = %v, want none", h.peer.setting(keyTools))
	}
}
