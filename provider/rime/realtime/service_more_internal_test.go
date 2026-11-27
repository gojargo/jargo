package realtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
)

func TestInterruptedLocalTextDoesNotReplaceProviderHistory(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.down.mu.Lock()
	h.down.transform = func(f frames.Frame) {
		if text, ok := f.(*frames.TTSTextFrame); ok {
			text.Text = "Heard portion"
		}
	}
	h.down.mu.Unlock()
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{text: "Complete generated answer", samples: 24000, open: true})
	until(t, "audio written", func() bool { return h.output.written() > 0 })
	h.queue(frames.NewInterruptionFrame())
	until(t, "a truncation", func() bool { return len(h.peer.sent(kindItemTruncate)) > 0 })
	until(t, "the heard text in the conversation", func() bool {
		return h.hasMessage(frames.RoleAssistant, "Heard portion")
	})
	if len(h.peer.sent(kindItemDelete)) > 0 || len(h.peer.sent(kindItemCreate)) > 0 {
		t.Error("the provider history was edited")
	}
	if !h.usable() || h.failed() {
		t.Errorf("usable = %v, errors = %v", h.usable(), h.errors())
	}
}

func TestAppendIsWrittenAtOnceAndRepliesOnlyWhenAsked(t *testing.T) {
	for _, runLLM := range []bool{false, true} {
		t.Run(fmt.Sprint(runLLM), func(t *testing.T) {
			h := converse(t, harnessOptions{})
			ref := h.peer.start(nil, nil)
			h.peer.message(ref, messageOptions{text: "Provider answer"})
			h.peer.complete(ref)
			until(t, "the reply in the conversation", func() bool { return len(h.convo.Messages()) > 0 })
			h.queue(appendUser("Next question", runLLM))
			until(t, "the append written", func() bool {
				return slices.Equal(h.peer.createdTexts(), []string{"Next question"})
			})
			until(t, "the append in the conversation", func() bool { return len(h.convo.Messages()) == 2 })
			want := 1
			if runLLM {
				h.answer(nil, "Hello", 480)
				want = 2
			} else {
				time.Sleep(50 * time.Millisecond)
			}
			if n := len(h.peer.sent(kindResponseNew)); n != want {
				t.Errorf("%d replies requested, want %d", n, want)
			}
			if len(h.peer.sent(kindItemDelete)) > 0 || !h.usable() {
				t.Error("history deleted or service unusable")
			}
		})
	}
}

func TestContextReplacementLeavesProviderHistory(t *testing.T) {
	h := converse(t, harnessOptions{})
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{text: "Old provider answer"})
	h.peer.complete(ref)
	until(t, "the reply in the conversation", func() bool { return len(h.convo.Messages()) > 0 })
	h.queue(frames.NewLLMMessagesUpdateFrame([]frames.Message{{Role: frames.RoleUser, Text: "Replacement"}}))
	until(t, "the replacement", func() bool {
		got := h.convo.Messages()
		return len(got) == 1 && got[0].Text == "Replacement"
	})
	time.Sleep(50 * time.Millisecond)
	if len(h.peer.sent(kindItemCreate)) > 0 || len(h.peer.sent(kindItemDelete)) > 0 {
		t.Error("the provider history was edited")
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 || !h.usable() {
		t.Errorf("%d replies requested, usable = %v", n, h.usable())
	}
}

func TestAppendDuringReplyIsWrittenAtOnce(t *testing.T) {
	h := converse(t, harnessOptions{})
	ref := h.peer.start(nil, nil)
	h.queue(appendUser("Side note", false))
	until(t, "the append written", func() bool { return slices.Equal(h.peer.createdTexts(), []string{"Side note"}) })
	h.peer.message(ref, messageOptions{text: "Current answer"})
	h.peer.complete(ref)
	until(t, "both messages", func() bool { return len(h.convo.Messages()) == 2 })
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestRejectedHistoryEditLeavesLocalContextChanged(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.peer.setRefusal(kindItemCreate, "invalid_request")
	h.queue(appendUser("Rejected input", true))
	until(t, "an error", h.failed)
	update, ok := errors.AsType[*HistoryUpdateError](h.errors()[0].Err)
	if !ok || update.Status != "rejected" {
		t.Fatalf("error = %v, want a rejected history update", h.errors()[0].Err)
	}
	if !h.hasMessage(frames.RoleUser, "Rejected input") {
		t.Error("the local conversation lost the message")
	}
	if len(h.peer.historyTexts()) != 0 || !h.usable() {
		t.Errorf("history = %v, usable = %v", h.peer.historyTexts(), h.usable())
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestPartialAppendFailureIsRecoverable(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.peer.failHistoryAt(2)
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleUser, Text: "First"}, {Role: frames.RoleUser, Text: "Second"},
	})
	f.RunLLM = true
	h.queue(f)
	until(t, "an error", h.failed)
	update, ok := errors.AsType[*HistoryUpdateError](h.errors()[0].Err)
	if !ok || update.Status != "partial" || !update.Recoverable() {
		t.Fatalf("error = %v, want a recoverable partial history update", h.errors()[0].Err)
	}
	if !h.usable() || !slices.Equal(h.peer.historyTexts(), []string{"First"}) {
		t.Errorf("usable = %v, history = %v", h.usable(), h.peer.historyTexts())
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestAppendQueuedWithTheOpeningReachesPrism(t *testing.T) {
	for _, runLLM := range []bool{false, true} {
		t.Run(fmt.Sprint(runLLM), func(t *testing.T) {
			convo := frames.NewLLMContext("")
			convo.SetMessages([]frames.Message{{Role: frames.RoleSystem, Text: "Be brief"}})
			h := converse(t, harnessOptions{convo: convo, opening: func(h *harness) {
				h.queue(appendUser("Early", runLLM), frames.NewLLMRunFrame())
			}})
			until(t, "the append in the conversation", func() bool { return h.hasMessage(frames.RoleUser, "Early") })
			time.Sleep(50 * time.Millisecond)
			if texts := h.peer.createdTexts(); !slices.Equal(texts, []string{"Be brief", "Early"}) {
				t.Errorf("created = %v", texts)
			}
			if n := len(h.peer.sent(kindResponseNew)); n != 1 || !h.usable() {
				t.Errorf("%d replies requested, usable = %v", n, h.usable())
			}
		})
	}
}

func TestAppendRecordedBeforeTheOpeningIsWrittenOnce(t *testing.T) {
	h := converse(t, harnessOptions{opening: func(h *harness) {
		h.queue(appendUser("Early", false))
		until(h.t, "the append in the conversation", func() bool { return len(h.convo.Messages()) > 0 })
		h.queue(frames.NewLLMRunFrame())
	}})
	if texts := h.peer.createdTexts(); !slices.Equal(texts, []string{"Early"}) {
		t.Errorf("created = %v", texts)
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestReplyHeldWhileTheBotSpeaksIsSkippedOncePrismAnswers(t *testing.T) {
	for _, prismAnswers := range []bool{false, true} {
		t.Run(fmt.Sprint(prismAnswers), func(t *testing.T) {
			h := converse(t, harnessOptions{})
			release := h.output.holdAudio(t)
			h.speak()
			h.queue(appendUser("Appended", true))
			until(t, "the append written", func() bool {
				return slices.Equal(h.peer.createdTexts(), []string{"Appended"})
			})
			time.Sleep(50 * time.Millisecond)
			if prismAnswers {
				ref := h.peer.start(nil, nil)
				h.peer.message(ref, messageOptions{text: "Answer that took it in"})
				h.peer.complete(ref)
			}
			release()
			until(t, "the bot to stop", func() bool { return has[*frames.BotStoppedSpeakingFrame](h.down) })
			if prismAnswers {
				time.Sleep(100 * time.Millisecond)
				if n := len(h.peer.sent(kindResponseNew)); n != 1 {
					t.Errorf("%d replies requested, want 1", n)
				}
			} else {
				h.answer(nil, "Hello", 480)
				if n := len(h.peer.sent(kindResponseNew)); n != 2 {
					t.Errorf("%d replies requested, want 2", n)
				}
			}
			if !h.usable() {
				t.Error("unusable")
			}
		})
	}
}

func TestResponseStartedDuringTheAppendWriteIsNotItsReply(t *testing.T) {
	h := converse(t, harnessOptions{})
	release := h.output.holdAudio(t)
	written := h.peer.gate(kindItemCreate)
	h.speak()
	h.queue(appendUser("Appended", true))
	until(t, "the append sent", func() bool { return slices.Equal(h.peer.createdTexts(), []string{"Appended"}) })
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{text: "Earlier answer"})
	h.peer.complete(ref)
	time.Sleep(50 * time.Millisecond)
	written()
	until(t, "the append written", func() bool { return slices.Equal(h.peer.historyTexts(), []string{"Appended"}) })
	release()
	h.answer(nil, "Hello", 480)
	if n := len(h.peer.sent(kindResponseNew)); n != 2 || !h.usable() {
		t.Errorf("%d replies requested, usable = %v", n, h.usable())
	}
}

func TestFailedAppendKeepsTheReplyOwedToAnEarlierOne(t *testing.T) {
	h := converse(t, harnessOptions{})
	release := h.output.holdAudio(t)
	h.speak()
	h.peer.failHistoryAt(2)
	h.queue(appendUser("First", true))
	h.queue(appendUser("Second", false))
	until(t, "an error", h.failed)
	release()
	h.answer(nil, "Hello", 480)
	if texts := h.peer.historyTexts(); !slices.Equal(texts, []string{"First"}) {
		t.Errorf("history = %v", texts)
	}
	if n := len(h.peer.sent(kindResponseNew)); n != 2 || !h.usable() {
		t.Errorf("%d replies requested, usable = %v", n, h.usable())
	}
}

func TestInterruptionDropsAHeldReply(t *testing.T) {
	h := converse(t, harnessOptions{})
	release := h.output.holdAudio(t)
	h.speak()
	h.queue(appendUser("Appended", true))
	until(t, "the append written", func() bool { return slices.Equal(h.peer.createdTexts(), []string{"Appended"}) })
	interruptions := len(captured[*frames.InterruptionFrame](h.down))
	h.queue(frames.NewInterruptionFrame())
	// The interruption must reach the assistant aggregator before the bot stops
	// speaking: the bot finishing first would rightly release the held reply.
	// Both travel the same path to it, so once the interruption is there the
	// bot's stop can only come after.
	until(t, "the interruption to reach the assistant aggregator", func() bool {
		return len(captured[*frames.InterruptionFrame](h.down)) > interruptions
	})
	release()
	until(t, "the requests to settle", func() bool { return h.requestsPending() == 0 })
	replace := frames.NewLLMMessagesUpdateFrame(h.convo.Messages())
	replace.RunLLM = true
	h.queue(replace)
	time.Sleep(100 * time.Millisecond)
	if n := len(h.peer.sent(kindResponseNew)); n != 1 || !h.usable() {
		t.Errorf("%d replies requested, usable = %v", n, h.usable())
	}
}

// lookup is a tool that answers with a paid status, or as handler does.
func lookup(name string, handler llm.FunctionCallHandler) frames.Tool {
	if handler == nil {
		handler = func(ctx context.Context, p llm.FunctionCallParams) error {
			return p.Result(ctx, `{"status": "paid"}`, nil)
		}
	}
	return frames.Tool{
		Name:        name,
		Description: "Look up an order.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
		Handler:     handler,
	}
}

// asyncLookup is a tool that outlives the reply that called it.
func asyncLookup(name string, cancellable bool) frames.Tool {
	tool := lookup(name, nil)
	tool.CancelOnInterruption = new(false)
	if cancellable {
		tool.CancellableByLLM = new(true)
	}
	return tool
}

func toolset(tools ...frames.Tool) settings.Opt[frames.ToolsSchema] {
	return settings.Set(frames.ToolsSchema{Standard: tools})
}

func update(delta *Settings) *frames.LLMUpdateSettingsFrame {
	return frames.NewLLMUpdateSettingsFrame(delta)
}

func instruction(text string) *Settings {
	delta := &Settings{}
	delta.SystemInstruction = settings.Set(text)
	return delta
}

// replyCause is the cause a reply request names.
func replyCause(request object) any {
	return at(request, "response", "metadata", "prism_cause")
}

// at reads a value nested in objects, nil where a key is missing.
func at(value any, keys ...string) any {
	for _, key := range keys {
		o, ok := value.(object)
		if !ok {
			return nil
		}
		value = o[key]
	}
	return value
}

// asObject is value as an object, nil when it is not one.
func asObject(value any) object {
	o, _ := value.(object)
	return o
}

// text is value as a string, empty when it is not one.
func text(value any) string {
	s, _ := value.(string)
	return s
}

// number is value as a JSON number, zero when it is not one.
func number(value any) float64 {
	f, _ := value.(float64)
	return f
}

// lastUpdate is the session settings of the last update sent.
func (p *peer) lastUpdate() object {
	sent := p.sent(kindSessionUpdate)
	return asObject(sent[len(sent)-1]["session"])
}

func TestSettingsToolsFillersTruncationAndMetadataUpdate(t *testing.T) {
	h := converse(t, harnessOptions{})
	delta := instruction("Billing support")
	delta.Tools = toolset(lookup("lookup", nil))
	delta.ToolFillers = settings.Set(map[string]Filler{"lookup": {Type: "text", Text: new("Checking")}})
	delta.Truncation = settings.Set(TruncationAuto())
	delta.ResponseMetadata = settings.Set(map[string]string{"department": "billing"})
	h.queue(update(delta))
	until(t, "the labels confirmed", func() bool {
		info, ok := h.svc.SessionInfo()
		return ok && info.ResponseMetadata["department"] == "billing"
	})
	if got := h.peer.setting("instructions"); got != "Billing support" {
		t.Errorf("instructions = %v", got)
	}
	if !h.svc.HasFunction("lookup") {
		t.Error("the tool's handler is not registered")
	}
	tools, _ := h.peer.setting(keyTools).([]any)
	if len(tools) == 0 || at(tools[0], "prism_filler", "text") != "Checking" {
		t.Errorf("tools = %v, want the filler", tools)
	}
	if got := h.peer.setting("truncation"); got != "auto" {
		t.Errorf("truncation = %v", got)
	}
	h.queue(frames.NewLLMSetToolsFrame([]frames.Tool{}))
	until(t, "the tools removed", func() bool { return len(h.peer.toolNames()) == 0 })
	if len(h.convo.Tools()) != 0 {
		t.Error("the conversation keeps the tools")
	}
}

// TestSpeedAndInterruptionRulesOpenAndUpdate waits for each update to reach
// the peer before it asserts on it.
func TestSpeedAndInterruptionRulesOpenAndUpdate(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) {
		c.Settings.Speed = settings.Set(1.1)
		c.Settings.InterruptMinWords = settings.Set(2)
		c.Settings.AcknowledgmentPhrases = settings.Set([]string{"uh huh"})
	}})
	opening := asObject(h.peer.sent(kindSessionUpdate)[0]["session"])
	turn := asObject(opening[keyTurn])
	if opening["speed"] != 1.1 || turn[keyInterruptMin] != 2.0 || fmt.Sprint(turn[keyAckPhrases]) != "[uh huh]" {
		t.Errorf("opening = %v", opening)
	}
	if _, ok := turn[keyInterruptPhr]; ok {
		t.Error("the opening sets interrupting phrases")
	}
	updates := len(h.peer.sent(kindSessionUpdate))
	h.queue(update(&Settings{Speed: settings.Set(0.9), InterruptPhrases: settings.Set([]string{"stop", "wait"})}))
	until(t, "the update confirmed", func() bool {
		speed, _ := h.live().Speed.Value()
		return speed == 0.9 && len(h.peer.sent(kindSessionUpdate)) > updates
	})
	last := h.peer.lastUpdate()
	if last["speed"] != 0.9 || fmt.Sprint(last[keyTurn]) != "map[interrupt_phrases:[stop wait]]" {
		t.Errorf("update = %v", last)
	}
	live := h.live()
	if phrases, _ := live.InterruptPhrases.Value(); !slices.Equal(phrases, []string{"stop", "wait"}) {
		t.Errorf("phrases = %v", phrases)
	}
	if words, _ := live.InterruptMinWords.Value(); words != 2 {
		t.Errorf("min words = %v", words)
	}
	info, _ := h.svc.SessionInfo()
	detection := info.TurnDetection
	if detection.InterruptMinWords == nil || *detection.InterruptMinWords != 2 ||
		!slices.Equal(detection.InterruptPhrases, []string{"stop", "wait"}) ||
		!slices.Equal(detection.AcknowledgmentPhrases, []string{"uh huh"}) {
		t.Errorf("turn detection = %+v", detection)
	}
	updates = len(h.peer.sent(kindSessionUpdate))
	h.queue(update(&Settings{AcknowledgmentPhrases: settings.Set([]string{})}))
	until(t, "the phrases cleared", func() bool {
		phrases, _ := h.live().AcknowledgmentPhrases.Value()
		return phrases != nil && len(phrases) == 0 && len(h.peer.sent(kindSessionUpdate)) > updates
	})
	if got := fmt.Sprint(h.peer.lastUpdate()[keyTurn]); got != "map[acknowledgment_phrases:[]]" {
		t.Errorf("turn update = %v", got)
	}
	if !h.usable() {
		t.Error("unusable")
	}
}

func TestOpeningTurnDetectionRulesJoinTheSettings(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) {
		c.TurnDetection = &TurnDetection{Eagerness: "low", InterruptMinWords: new(3)}
	}})
	turn := asObject(at(h.peer.sent(kindSessionUpdate)[0], "session", keyTurn))
	if turn[keyEagerness] != "low" || turn[keyInterruptMin] != 3.0 {
		t.Errorf("turn detection = %v", turn)
	}
	if words, _ := h.live().InterruptMinWords.Value(); words != 3 {
		t.Errorf("min words = %v", words)
	}
}

func TestInterruptionRuleSetTwiceIsRejected(t *testing.T) {
	cfg := Config{
		APIKey:        "key",
		TurnDetection: &TurnDetection{InterruptMinWords: new(3)},
		Settings:      Settings{InterruptMinWords: settings.Set(2)},
	}
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "interrupt_min_words") {
		t.Errorf("Validate() = %v, want the rule named", err)
	}
}

func TestInvalidSpeedAndInterruptionRulesAreRejected(t *testing.T) {
	for name, s := range map[string]Settings{
		"speed":               {Speed: settings.Set(2.0)},
		"interrupt_min_words": {InterruptMinWords: settings.Set(0)},
	} {
		if err := (Config{APIKey: "key", Settings: s}).Validate(); err == nil {
			t.Errorf("%s: Validate() = nil, want an error", name)
		}
	}
}

func TestInvalidInterruptionRuleUpdateIsRejectedWithoutAWrite(t *testing.T) {
	h := converse(t, harnessOptions{})
	updates := len(h.peer.sent(kindSessionUpdate))
	h.queue(update(&Settings{InterruptMinWords: settings.Set(11)}))
	until(t, "an error", h.failed)
	if n := len(h.peer.sent(kindSessionUpdate)); n != updates {
		t.Errorf("%d updates sent, want %d", n, updates)
	}
	if _, ok := h.live().InterruptMinWords.Value(); ok || !h.usable() {
		t.Errorf("min words kept, or unusable")
	}
}

func TestFullPartialTranscriptsReplaceEarlierPartials(t *testing.T) {
	h := converse(t, harnessOptions{})
	event := "conversation.item.input_audio_transcription.delta"
	h.peer.emit(event, object{"item_id": "speech-1", "delta": "Wether"})
	h.peer.emit(event, object{"item_id": "speech-1", "delta": " in", "transcript": "Weather in"})
	h.peer.emit(event, object{"item_id": "speech-1", "delta": " Paris"})
	until(t, "three partials", func() bool { return len(captured[*frames.InterimTranscriptionFrame](h.down)) == 3 })
	var texts []string
	for _, f := range captured[*frames.InterimTranscriptionFrame](h.down) {
		texts = append(texts, f.Text)
	}
	if !slices.Equal(texts, []string{"Wether", "Weather in", "Weather in Paris"}) {
		t.Errorf("partials = %v", texts)
	}
}

func TestLongHistoryMessageIsLeftToPrism(t *testing.T) {
	text := strings.Repeat("word ", 1200)
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleUser, Text: text}})
	h := converse(t, harnessOptions{convo: convo})
	if texts := h.peer.createdTexts(); !slices.Equal(texts, []string{text}) || !h.usable() {
		t.Errorf("%d items created, usable = %v", len(texts), h.usable())
	}
}

func TestRejectedSettingsPreserveAcknowledgedProviderSettings(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.peer.setRefusal(kindSessionUpdate, "invalid_request")
	h.queue(update(instruction("Rejected")))
	until(t, "an error", h.failed)
	if got, _ := h.live().SystemInstruction.Value(); got != "" {
		t.Errorf("instruction = %q, want none", got)
	}
	if h.svc.SystemInstruction() != "" || !h.usable() {
		t.Errorf("composed instruction = %q, usable = %v", h.svc.SystemInstruction(), h.usable())
	}
}

func TestRuntimeInstructionKeepsTheComposedGuidance(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) {
		c.Settings.SystemInstruction = settings.Set("Support")
		c.Settings.Tools = toolset(asyncLookup("slow_lookup", false))
	}})
	if got := h.peer.setting("instructions"); got != "Support\n\n"+frames.AsyncToolInstructions {
		t.Errorf("instructions = %q", got)
	}
	h.queue(update(instruction("Billing")))
	until(t, "the new instruction", func() bool {
		text, _ := h.peer.setting("instructions").(string)
		return strings.HasPrefix(text, "Billing")
	})
	if got := h.peer.setting("instructions"); got != "Billing\n\n"+frames.AsyncToolInstructions {
		t.Errorf("instructions = %q", got)
	}
	if got := h.svc.SystemInstruction(); got != h.peer.setting("instructions") {
		t.Errorf("composed instruction = %q, Prism holds %q", got, h.peer.setting("instructions"))
	}
}

func TestToolsUpdateSendsCancelToolsAndTheirGuidance(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) { c.Settings.SystemInstruction = settings.Set("Support") }})
	h.queue(frames.NewLLMSetToolsFrame([]frames.Tool{asyncLookup("stoppable_lookup", true)}))
	until(t, "the tool advertised", func() bool { return slices.Contains(h.peer.toolNames(), "stoppable_lookup") })
	want := []string{"stoppable_lookup", llm.CancelToolName("stoppable_lookup")}
	if names := h.peer.toolNames(); !slices.Equal(names, want) {
		t.Errorf("tools = %v", names)
	}
	text, _ := h.peer.setting("instructions").(string)
	if !strings.Contains(text, "ASYNC TOOL CANCELLATION:") || !strings.Contains(text, frames.AsyncToolInstructions) {
		t.Errorf("instructions = %q, want the async guidance", text)
	}
	h.queue(frames.NewLLMSetToolsFrame([]frames.Tool{}))
	until(t, "the tools removed", func() bool { return len(h.peer.toolNames()) == 0 })
	if got := h.peer.setting("instructions"); got != "Support" {
		t.Errorf("instructions = %q", got)
	}
	if len(h.convo.Tools()) != 0 {
		t.Error("the conversation keeps the tools")
	}
}

func TestRefusedToolsUpdateRestoresToolsAndInstruction(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) { c.Settings.SystemInstruction = settings.Set("Support") }})
	h.peer.setRefusal(kindSessionUpdate, "invalid_request")
	h.queue(frames.NewLLMSetToolsFrame([]frames.Tool{asyncLookup("stoppable_lookup", true)}))
	until(t, "an error", h.failed)
	if h.svc.HasFunction("stoppable_lookup") || h.svc.HasFunction(llm.CancelToolName("stoppable_lookup")) {
		t.Error("the refused tools' handlers are registered")
	}
	if got := h.svc.SystemInstruction(); got != "Support" {
		t.Errorf("instruction = %q", got)
	}
	// The user aggregator set the refused tools on the conversation; they are
	// cleared.
	if len(h.convo.Tools()) != 0 || !h.usable() {
		t.Errorf("tools = %v, usable = %v", h.convo.Tools(), h.usable())
	}
	// The next update compares against what Prism holds, so it sends nothing
	// stale.
	h.queue(update(&Settings{Speed: settings.Set(1.1)}))
	until(t, "the speed", func() bool { return h.peer.setting("speed") == 1.1 })
	last := h.peer.lastUpdate()
	if _, ok := last[keyTools]; ok {
		t.Error("the update resends the tools")
	}
	if _, ok := last["instructions"]; ok {
		t.Error("the update resends the instructions")
	}
}

func TestContextFrameDuringToolsUpdateKeepsTheNewHandlers(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetTools([]frames.Tool{asyncLookup("slow_lookup", false)})
	h := converse(t, harnessOptions{convo: convo, config: func(c *Config) {
		c.Settings.SystemInstruction = settings.Set("Support")
	}})
	until(t, "the context's tools", func() bool { return slices.Equal(h.peer.toolNames(), []string{"slow_lookup"}) })
	open := h.peer.gate(kindSessionUpdate)
	updates := len(h.peer.sent(kindSessionUpdate))
	h.queue(update(&Settings{Tools: toolset(asyncLookup("stoppable_lookup", true))}))
	until(t, "the update sent", func() bool { return len(h.peer.sent(kindSessionUpdate)) > updates })
	// The base service syncs the handlers to the conversation's tools on this
	// frame.
	h.queue(frames.NewLLMContextFrame(convo))
	time.Sleep(100 * time.Millisecond)
	open()
	until(t, "the new tool", func() bool { return slices.Contains(h.peer.toolNames(), "stoppable_lookup") })
	time.Sleep(100 * time.Millisecond)
	if !h.svc.HasFunction("stoppable_lookup") || !h.svc.HasFunction(llm.CancelToolName("stoppable_lookup")) {
		t.Error("the new handlers are not registered")
	}
	if got := h.svc.SystemInstruction(); got != h.peer.setting("instructions") {
		t.Errorf("composed instruction = %q, Prism holds %q", got, h.peer.setting("instructions"))
	}
}

// TestUpdateWithoutInstructionOrToolChangesSendsNeither waits for the update
// to reach the peer before it asserts on it.
func TestUpdateWithoutInstructionOrToolChangesSendsNeither(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) {
		c.Settings.SystemInstruction = settings.Set("Support")
		c.Settings.Tools = toolset(asyncLookup("slow_lookup", false))
	}})
	updates := len(h.peer.sent(kindSessionUpdate))
	delta := instruction("Support")
	delta.Speed = settings.Set(0.9)
	h.queue(update(delta))
	until(t, "the update", func() bool { return len(h.peer.sent(kindSessionUpdate)) > updates })
	if got := h.peer.lastUpdate(); len(got) != 1 || got["speed"] != 0.9 {
		t.Errorf("update = %v, want the speed alone", got)
	}
}

// TestSettingsForAnotherServicePassThroughUnapplied waits for the update to be
// applied before it asserts on it.
func TestSettingsForAnotherServicePassThroughUnapplied(t *testing.T) {
	h := converse(t, harnessOptions{})
	updates := len(h.peer.sent(kindSessionUpdate))
	other := New(Config{APIKey: "other"})
	elsewhere := update(instruction("Elsewhere"))
	elsewhere.Service = other
	h.queue(elsewhere, update(instruction("Here")))
	until(t, "the update", func() bool {
		got, _ := h.live().SystemInstruction.Value()
		return got == "Here" && len(h.peer.sent(kindSessionUpdate)) > updates
	})
	if n := len(h.peer.sent(kindSessionUpdate)); n != updates+1 {
		t.Errorf("%d updates sent, want %d", n, updates+1)
	}
	passed := captured[*frames.LLMUpdateSettingsFrame](h.down)
	if len(passed) != 1 || passed[0].Service != other {
		t.Errorf("passed on = %v, want the other service's update", passed)
	}
}

func TestAudioAndEventsContinueWhileSettingsAcknowledgmentWaits(t *testing.T) {
	h := converse(t, harnessOptions{})
	open := h.peer.gate(kindSessionUpdate)
	h.queue(update(instruction("Waiting")))
	h.queue(frames.NewInputAudioRawFrame(make([]byte, 960), 24000, 1))
	if event := h.peer.next("input_audio_buffer.append"); event["audio"] == "" {
		t.Error("the audio is empty")
	}
	open()
	until(t, "the instruction", func() bool {
		got, _ := h.live().SystemInstruction.Value()
		return got == "Waiting"
	})
}

// counter counts the calls of a handler.
type counter struct {
	mu    sync.Mutex
	calls int
}

func (c *counter) add() {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
}

func (c *counter) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

func TestToolResultContinuesOnceAndIsNotImportedOnNextTurn(t *testing.T) {
	calls := &counter{}
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		calls.add()
		return p.Result(ctx, `{"status": "paid"}`, nil)
	}
	h := converse(t, harnessOptions{config: func(c *Config) { c.Settings.Tools = toolset(lookup("lookup", handler)) }})
	ref := h.peer.start(nil, nil)
	h.peer.emitCall(ref, "call-1", "lookup")
	h.peer.complete(ref)
	continuation := h.peer.nextReply()
	if replyCause(continuation) != "tool_continuation" {
		t.Errorf("cause = %v", replyCause(continuation))
	}
	if n := calls.count(); n != 1 {
		t.Errorf("%d calls, want 1", n)
	}
	if n := len(h.peer.sent(kindItemCreate)); n != 1 {
		t.Errorf("%d items, want 1", n)
	}
	next := h.peer.start(continuation, nil)
	h.peer.say(next, "It is paid")
	h.peer.complete(next)
	until(t, "the answer", func() bool { return h.hasMessage(frames.RoleAssistant, "It is paid") })
	// The output is submitted once, never imported again as history.
	outputs := 0
	for _, item := range h.peer.createdItems() {
		if item["type"] == "function_call_output" {
			outputs++
		}
	}
	if outputs != 1 || calls.count() != 1 {
		t.Errorf("%d outputs, %d calls, want one of each", outputs, calls.count())
	}
}

func TestMicrophonePCMFormats(t *testing.T) {
	for _, format := range [][2]int{{8000, 1}, {16000, 2}, {24000, 1}, {44100, 2}, {48000, 3}} {
		t.Run(fmt.Sprint(format), func(t *testing.T) {
			rate, channels := format[0], format[1]
			h := converse(t, harnessOptions{})
			h.queue(frames.NewInputAudioRawFrame(make([]byte, 2*(rate/10)*channels), rate, channels))
			if event := h.peer.next("input_audio_buffer.append"); event["audio"] == "" {
				t.Error("the audio is empty")
			}
		})
	}
}

func TestInterruptionCancelsReplyAndSendsEstimatedPosition(t *testing.T) {
	h := converse(t, harnessOptions{})
	ref := h.peer.start(nil, nil)
	h.peer.message(ref, messageOptions{samples: 24000, open: true})
	until(t, "audio written", func() bool { return h.output.written() > 0 })
	h.queue(frames.NewInterruptionFrame())
	until(t, "a truncation", func() bool { return len(h.peer.sent(kindItemTruncate)) > 0 })
	truncations := h.peer.sent(kindItemTruncate)
	if position := number(truncations[len(truncations)-1]["audio_end_ms"]); position < 0 || position > 1000 {
		t.Errorf("position = %v", position)
	}
	cancels := h.peer.sent(kindResponseStop)
	if len(cancels) == 0 || cancels[len(cancels)-1]["response_id"] != ref {
		t.Errorf("cancels = %v", cancels)
	}
	before := h.output.written()
	h.peer.emit("response.audio.delta", object{
		"response_id": ref, "output_index": 0, "item_id": "message-" + ref, "content_index": 0, "delta": "EAA=",
	})
	h.queue(appendUser("Go on", true))
	request := h.peer.nextReply()
	if h.output.written() != before {
		t.Error("audio of the canceled reply was played")
	}
	h.answer(request, "After interruption", 480)
	if len(h.peer.sent("prism.playback.drained")) > 0 {
		t.Error("a playback receipt was sent")
	}
}

// observe records the payloads of type T the service's events carry.
func observe[T EventPayload](s *LLMService) func() []T {
	var mu sync.Mutex
	var seen []T
	event.On(s.Events(), EventRimeEvent, func(_ context.Context, event *Event) {
		if payload, ok := event.Payload.(T); ok {
			mu.Lock()
			seen = append(seen, payload)
			mu.Unlock()
		}
	})
	return func() []T {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(seen)
	}
}

func TestRetainedTextEventIsForwardedWithoutReplacingContext(t *testing.T) {
	for _, kept := range []*string{nil, new(""), new("Retained words")} {
		t.Run(fmt.Sprint(kept != nil), func(t *testing.T) {
			var observed func() []ItemTruncated
			h := converse(t, harnessOptions{setup: func(s *LLMService) { observed = observe[ItemTruncated](s) }})
			ref := h.peer.start(nil, nil)
			h.peer.message(ref, messageOptions{text: "Generated answer"})
			h.peer.complete(ref)
			until(t, "the reply in the conversation", func() bool { return len(h.convo.Messages()) > 0 })
			body := object{"item_id": "message-" + ref, "content_index": 0, "audio_end_ms": 120}
			if kept != nil {
				body["prism_kept_text"] = *kept
			}
			h.peer.emit("conversation.item.truncated", body)
			until(t, "the event", func() bool { return len(observed()) > 0 })
			got := observed()[0]
			if got.ItemID != "message-"+ref || got.AudioEndMS != 120 || got.ContentIndex != 0 ||
				(got.KeptText == nil) != (kept == nil) || (kept != nil && *got.KeptText != *kept) {
				t.Errorf("event = %+v, want the truncation with the kept text %v", got, kept)
			}
			if messages := h.convo.Messages(); len(messages) != 1 || messages[0].Text != "Generated answer" {
				t.Errorf("messages = %v", messages)
			}
			if len(h.peer.sent(kindItemCreate)) > 0 || len(h.peer.sent(kindItemDelete)) > 0 || !h.usable() {
				t.Error("history edited, or unusable")
			}
		})
	}
}

func TestPromptTruncationSummaryIsForwardedWithoutDeletingHistory(t *testing.T) {
	summaries := []object{nil, {"items_left_out": 3}, {"items_left_out": 3, "first_kept_item_id": "kept-item"}}
	for _, summary := range summaries {
		t.Run(fmt.Sprint(summary), func(t *testing.T) {
			var observed func() []ResponseEnded
			convo := frames.NewLLMContext("")
			convo.SetMessages([]frames.Message{{Role: frames.RoleUser, Text: "Keep this history"}})
			h := converse(t, harnessOptions{convo: convo, setup: func(s *LLMService) {
				observed = observe[ResponseEnded](s)
			}})
			ref := h.peer.start(nil, nil)
			h.peer.message(ref, messageOptions{text: "Answer"})
			response := object{"id": ref, "status": "completed"}
			if summary != nil {
				response["prism_truncation"] = summary
			}
			h.peer.emit("response.done", object{"response": response})
			h.peer.emit("prism.typed_input.ready", nil)
			var ended ResponseEnded
			until(t, "the reply's end", func() bool {
				for _, event := range observed() {
					if event.Response.ResponseID == ref {
						ended = event
						return true
					}
				}
				return false
			})
			var want *ResponseTruncation
			if summary != nil {
				first, _ := summary["first_kept_item_id"].(string)
				want = &ResponseTruncation{ItemsLeftOut: 3, FirstKeptItemID: first}
			}
			if (want == nil) != (ended.Truncation == nil) || (want != nil && *want != *ended.Truncation) {
				t.Errorf("truncation = %+v, want %+v", ended.Truncation, want)
			}
			if texts := h.peer.historyTexts(); len(texts) == 0 || texts[0] != "Keep this history" {
				t.Errorf("history = %v", texts)
			}
			if messages := h.convo.Messages(); messages[0].Text != "Keep this history" {
				t.Errorf("messages = %v", messages)
			}
			if len(h.peer.sent(kindItemCreate)) != 1 || len(h.peer.sent(kindItemDelete)) > 0 || !h.usable() {
				t.Error("history edited, or unusable")
			}
		})
	}
}

func TestUnversionedSettingsChangesDoNotRequireRevisionTracking(t *testing.T) {
	h := converse(t, harnessOptions{unversioned: true})
	h.queue(update(&Settings{InterruptOnSpeech: settings.Set(false)}))
	until(t, "the setting", func() bool {
		on, ok := h.live().InterruptOnSpeech.Value()
		return ok && !on
	})
	ref := h.peer.start(nil, nil)
	h.peer.say(ref, "Hello")
	h.peer.complete(ref)
	until(t, "the reply", func() bool { return len(captured[*frames.LLMFullResponseEndFrame](h.down)) == 2 })
	if !h.usable() {
		t.Error("unusable")
	}
}

func TestCleanupLeavesNoOwnedTasks(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.close()
	waited := make(chan struct{})
	go func() {
		h.svc.tasks.Wait()
		close(waited)
	}()
	waitFor(t, "the service's tasks", waited)
	if h.requestsPending() != 0 || !h.svc.closing.Load() {
		t.Errorf("%d requests pending, closing = %v", h.requestsPending(), h.svc.closing.Load())
	}
}

func TestSystemAppendWithoutInferenceIsWrittenWithoutAReply(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.queue(frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleSystem, Text: "Remember the order"}}))
	until(t, "the append written", func() bool { return len(h.peer.sent(kindItemCreate)) > 0 })
	if role := h.peer.createdItems()[0]["role"]; role != "system" {
		t.Errorf("role = %v", role)
	}
	until(t, "the append in the conversation", func() bool { return len(h.convo.Messages()) > 0 })
	time.Sleep(50 * time.Millisecond)
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested, want 1", n)
	}
}

func TestInvalidAppendIsRejectedWithoutWrites(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.SetMessages([]frames.Message{{Role: frames.RoleUser, Text: "Keep me"}})
	h := converse(t, harnessOptions{convo: convo})
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Content: []frames.ContentPart{
		{Type: frames.ContentPartImageURL, ImageURL: "invalid"},
	}}})
	f.RunLLM = true
	h.queue(f)
	until(t, "an error", h.failed)
	if !h.usable() || !slices.Equal(h.peer.historyTexts(), []string{"Keep me"}) {
		t.Errorf("usable = %v, history = %v", h.usable(), h.peer.historyTexts())
	}
}

func TestMissingRequestAcknowledgmentFailsWithoutReplay(t *testing.T) {
	h := converse(t, harnessOptions{config: func(c *Config) { c.Timeouts.Request = 50 * time.Millisecond }})
	h.peer.gate(kindResponseNew)
	h.queue(appendUser("Hi", true))
	until(t, "the service to fail", func() bool { return !h.usable() })
	if n := len(h.peer.sent(kindResponseNew)); n != 2 {
		t.Errorf("%d replies requested, want 2", n)
	}
}

func TestTextOnlyOutputSurvivesRestorationOfOutputSettings(t *testing.T) {
	h := converse(t, harnessOptions{})
	h.queue(
		frames.NewLLMConfigureOutputFrame(true), appendUser("Text only", true), frames.NewLLMConfigureOutputFrame(false),
	)
	h.answer(nil, "Text reply", 960)
	until(t, "the reply", func() bool { return h.hasMessage(frames.RoleAssistant, "Text reply") })
	if h.output.written() > 0 {
		t.Error("audio was played")
	}
	if !slices.ContainsFunc(captured[*frames.LLMTextFrame](h.down), func(f *frames.LLMTextFrame) bool {
		return f.Text == "Text reply"
	}) {
		t.Error("the text never reached the aggregator")
	}
}

func TestMultipleToolResultsStartOneContinuationAfterGenerationEnds(t *testing.T) {
	calls := &counter{}
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		var args struct{ Value string }
		_ = json.Unmarshal(p.Arguments, &args)
		calls.add()
		return p.Result(ctx, fmt.Sprintf(`{"value": %q}`, args.Value), nil)
	}
	h := converse(t, harnessOptions{config: func(c *Config) { c.Settings.Tools = toolset(lookup("lookup", handler)) }})
	ref := h.peer.start(nil, nil)
	for index := range 2 {
		h.peer.emit("response.function_call_arguments.done", object{
			"response_id": ref, "item_id": fmt.Sprintf("call-item-%d", index), "output_index": index,
			"call_id": fmt.Sprintf("call-%d", index), "name": "lookup",
			"arguments": fmt.Sprintf(`{"value": "%d"}`, index),
		})
		until(t, "the call", func() bool { return calls.count() == index+1 })
	}
	until(t, "both results", func() bool { return len(h.peer.sent(kindItemCreate)) == 2 })
	if n := len(h.peer.sent(kindResponseNew)); n != 1 {
		t.Errorf("%d replies requested before the end, want 1", n)
	}
	h.peer.complete(ref)
	continuation := h.peer.nextReply()
	if replyCause(continuation) != "tool_continuation" {
		t.Errorf("cause = %v", replyCause(continuation))
	}
	h.answer(continuation, "Hello", 480)
	until(t, "the answer", func() bool { return h.hasMessage(frames.RoleAssistant, "Hello") })
	if n := len(h.peer.sent(kindResponseNew)); n != 2 {
		t.Errorf("%d replies requested, want 2", n)
	}
}

func TestRemovedToolUsesStandardMissingHandler(t *testing.T) {
	calls := &counter{}
	handler := func(ctx context.Context, p llm.FunctionCallParams) error {
		calls.add()
		return p.Result(ctx, "Done", nil)
	}
	h := converse(t, harnessOptions{config: func(c *Config) { c.Settings.Tools = toolset(lookup("lookup", handler)) }})
	ref := h.peer.start(nil, nil)
	h.queue(frames.NewLLMSetToolsFrame([]frames.Tool{}))
	until(t, "the handler removed", func() bool { return !h.svc.HasFunction("lookup") })
	h.peer.emit("response.function_call_arguments.done", object{
		"response_id": ref, "item_id": "missing-call", "output_index": 0,
		"call_id": "call-1", "name": "lookup", "arguments": "{}",
	})
	h.peer.complete(ref)
	request := h.peer.nextReply()
	if calls.count() != 0 {
		t.Error("the removed handler ran")
	}
	items := h.peer.createdItems()
	if output, _ := items[len(items)-1]["output"].(string); !strings.Contains(output, "not currently available") {
		t.Errorf("output = %q", output)
	}
	h.answer(request, "Hello", 480)
}

func TestToolResultEncodingMatchesNormalContextAggregation(t *testing.T) {
	// The results as a handler gives them: JSON text, or nothing at all.
	results := []*string{new(`"done"`), new(`""`), nil, new("false"), new("0"), new("{}"), new(`{"status": "paid"}`)}
	for _, result := range results {
		name := "none"
		if result != nil {
			name = *result
		}
		t.Run(name, func(t *testing.T) {
			handler := func(ctx context.Context, p llm.FunctionCallParams) error {
				if result == nil {
					return p.Result(ctx, "", nil)
				}
				return p.Result(ctx, *result, nil)
			}
			h := converse(t, harnessOptions{config: func(c *Config) { c.Settings.Tools = toolset(lookup("lookup", handler)) }})
			ref := h.peer.start(nil, nil)
			h.peer.emitCall(ref, "call-1", "lookup")
			h.peer.complete(ref)
			if result != nil {
				h.answer(h.peer.nextReply(), "Tool answer", 480)
				until(t, "the answer", func() bool { return h.hasMessage(frames.RoleAssistant, "Tool answer") })
			} else {
				until(t, "the result", func() bool { return len(h.peer.sent(kindItemCreate)) == 1 })
				if n := len(h.peer.sent(kindResponseNew)); n != 1 {
					t.Errorf("%d replies requested, want 1", n)
				}
			}
			if n := len(h.peer.sent(kindItemCreate)); n != 1 || !h.usable() {
				t.Errorf("%d items, usable = %v", n, h.usable())
			}
		})
	}
}
