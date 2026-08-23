package aggregators_test

// Tests for the tool-change announcements WithToolChangeMessages enables: a
// developer message describing what an LLMSetToolsFrame adds and removes.
// Ported from upstream's tool-change tests.

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
)

func toolset(names ...string) []frames.Tool {
	out := make([]frames.Tool, len(names))
	for i, n := range names {
		out[i] = frames.Tool{Name: n}
	}
	return out
}

// developerMessages returns the out-of-band instructions written to the
// conversation, which is where a tool change is announced.
func developerMessages(convo *frames.LLMContext) []string {
	var out []string
	for _, m := range convo.Messages() {
		if m.Role == frames.RoleDeveloper {
			out = append(out, m.Text)
		}
	}
	return out
}

// sendSetTools drives one toolset change through the given processors and waits
// for it to have been handled.
func sendSetTools(t *testing.T, convo *frames.LLMContext, tools []frames.Tool, ps ...processor.Processor) {
	t.Helper()
	task, seen, runDone := runPair(t, ps...)
	defer func() { task.StopWhenDone(); <-runDone }()
	task.QueueFrame(frames.NewLLMSetToolsFrame(tools))
	// Drain so the change has certainly been applied before the assertions.
	drainCount(seen, func(frames.Frame) bool { return false })
}

func TestToolChangeMessagesOffByDefault(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a"))
	pair := aggregators.New(convo)
	sendSetTools(t, convo, toolset("a", "b"), pair.User())

	if msgs := developerMessages(convo); len(msgs) != 0 {
		t.Errorf("developer messages = %q, want none without the option", msgs)
	}
}

func TestToolChangeMessagesAnnounceAdditions(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("a", "b", "c"), pair.User())

	msgs := developerMessages(convo)
	if len(msgs) != 1 {
		t.Fatalf("developer messages = %q, want exactly one", msgs)
	}
	if !strings.Contains(msgs[0], "just been added") {
		t.Errorf("message does not announce an addition: %q", msgs[0])
	}
	if strings.Contains(msgs[0], "removed") {
		t.Errorf("message announces a removal that did not happen: %q", msgs[0])
	}
	// Sorted, so the order is the same every time.
	if i, j := strings.Index(msgs[0], "`b`"), strings.Index(msgs[0], "`c`"); i < 0 || j < 0 || i > j {
		t.Errorf("names are not listed sorted: %q", msgs[0])
	}
}

func TestToolChangeMessagesAnnounceRemovals(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a", "b", "c"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("a"), pair.User())

	msgs := developerMessages(convo)
	if len(msgs) != 1 {
		t.Fatalf("developer messages = %q, want exactly one", msgs)
	}
	if !strings.Contains(msgs[0], "just been removed") {
		t.Errorf("message does not announce a removal: %q", msgs[0])
	}
	if strings.Contains(msgs[0], "just been added") {
		t.Errorf("message announces an addition that did not happen: %q", msgs[0])
	}
	for _, want := range []string{"`b`", "`c`"} {
		if !strings.Contains(msgs[0], want) {
			t.Errorf("message does not name %s: %q", want, msgs[0])
		}
	}
}

func TestToolChangeMessagesCombineAdditionsAndRemovals(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a", "b"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("b", "c"), pair.User())

	msgs := developerMessages(convo)
	if len(msgs) != 1 {
		t.Fatalf("developer messages = %q, want exactly one", msgs)
	}
	added, removed := strings.Index(msgs[0], "just been added"), strings.Index(msgs[0], "just been removed")
	if added < 0 || removed < 0 {
		t.Fatalf("message does not carry both halves: %q", msgs[0])
	}
	if added > removed {
		t.Errorf("the addition should be announced before the removal: %q", msgs[0])
	}
	if !strings.Contains(msgs[0], "`c`") || !strings.Contains(msgs[0], "`a`") {
		t.Errorf("message does not name what changed: %q", msgs[0])
	}
}

func TestToolChangeMessagesSaysNothingWhenNothingChanged(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a", "b"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("a", "b"), pair.User())

	if msgs := developerMessages(convo); len(msgs) != 0 {
		t.Errorf("developer messages = %q, want none for an empty diff", msgs)
	}
}

func TestToolChangeMessagesEmptyToolsetRemovesEverything(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a", "b"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, nil, pair.User())

	msgs := developerMessages(convo)
	if len(msgs) != 1 || !strings.Contains(msgs[0], "just been removed") {
		t.Fatalf("developer messages = %q, want one announcing both removals", msgs)
	}
	for _, want := range []string{"`a`", "`b`"} {
		if !strings.Contains(msgs[0], want) {
			t.Errorf("message does not name %s: %q", want, msgs[0])
		}
	}
}

func TestToolChangeMessagesFirstToolsetIsAllAdditions(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("x", "y"), pair.User())

	msgs := developerMessages(convo)
	if len(msgs) != 1 || !strings.Contains(msgs[0], "just been added") {
		t.Fatalf("developer messages = %q, want one announcing both additions", msgs)
	}
	for _, want := range []string{"`x`", "`y`"} {
		if !strings.Contains(msgs[0], want) {
			t.Errorf("message does not name %s: %q", want, msgs[0])
		}
	}
}

// Both halves take part, and they share the conversation, so whichever one
// handles the frame first writes the announcement and the other one's diff is
// empty by the time it looks.
func TestToolChangeMessagesAnnounceOnceAcrossThePair(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("a", "b"), pair.User(), pair.Assistant())

	msgs := developerMessages(convo)
	if len(msgs) != 1 {
		t.Fatalf("developer messages = %q, want exactly one announcement", msgs)
	}
	if !strings.Contains(msgs[0], "`b`") {
		t.Errorf("message does not name the added tool: %q", msgs[0])
	}
}

// Order-independence: the assistant half announces when it is the one that gets
// there first.
func TestToolChangeMessagesAnnouncedByTheAssistantHalfWhenItHandlesItFirst(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.SetTools(toolset("a"))
	pair := aggregators.New(convo, aggregators.WithToolChangeMessages())
	sendSetTools(t, convo, toolset("a", "b"), pair.Assistant())

	msgs := developerMessages(convo)
	if len(msgs) != 1 {
		t.Fatalf("developer messages = %q, want exactly one announcement", msgs)
	}
	if !strings.Contains(msgs[0], "`b`") {
		t.Errorf("message does not name the added tool: %q", msgs[0])
	}
}
