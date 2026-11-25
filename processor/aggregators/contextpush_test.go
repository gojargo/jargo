package aggregators_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
)

// Tests for the run a tool result or an appended message asks for when the push
// cannot be made at once. A push held for frames still queued, or for a user who
// is speaking, is owed: the next result or append to find the way clear makes it
// whatever it asks itself, and a response starting settles it.
//
// Ported from upstream's assistant-aggregator suite.

// assistantHarness runs an assistant aggregator and records the context frames it
// pushes upstream, which is how it asks for generation to run, and every frame
// that reaches the end of the pipeline.
type assistantHarness struct {
	task      *pipeline.Worker
	assistant *aggregators.AssistantAggregator

	mu   sync.Mutex
	runs int
	down []frames.Frame
}

func newAssistantHarness(t *testing.T, convo *frames.LLMContext) *assistantHarness {
	t.Helper()

	h := &assistantHarness{}
	h.assistant = aggregators.New(convo).Assistant()
	h.task = pipeline.NewWorker(pipeline.New(h.assistant), pipeline.WorkerConfig{
		ReachedUpstreamFilter:   pipeline.AnyFrame,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&h.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.LLMContextFrame); ok {
			h.mu.Lock()
			h.runs++
			h.mu.Unlock()
		}
	})
	started := make(chan struct{})
	event.On(&h.task.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) {
		close(started)
	})
	event.On(&h.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		h.mu.Lock()
		h.down = append(h.down, f)
		h.mu.Unlock()
	})

	runDone := make(chan error, 1)
	go func() { runDone <- h.task.Run(context.Background()) }()
	t.Cleanup(func() {
		h.task.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Error("task did not finish")
		}
	})
	// Started before a test drives it, so a pause takes the frames that follow
	// into the aggregator's queue rather than holding them ahead of the start.
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("the aggregator never started")
	}
	return h
}

// generations is how many times the aggregator has asked for generation to run.
func (h *assistantHarness) generations() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.runs
}

// downstream is the frames that reached the end of the pipeline, by type name.
func (h *assistantHarness) downstream() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	names := make([]string, 0, len(h.down))
	for _, f := range h.down {
		name, _, _ := strings.Cut(f.Name(), "#")
		names = append(names, name)
	}
	return names
}

// queueTogether queues fs so the aggregator finds them waiting together, as a
// burst arriving faster than it is handled would be: processing is paused until
// the last of them is in the aggregator's queue.
func (h *assistantHarness) queueTogether(t *testing.T, fs ...frames.Frame) {
	t.Helper()
	h.assistant.PauseProcessingFrames()
	h.task.QueueFrames(fs)
	last := fs[len(fs)-1]
	if !waitFor(3*time.Second, func() bool {
		return h.assistant.HasQueuedFrame(func(f frames.Frame) bool { return f == last })
	}) {
		t.Fatal("the burst never reached the aggregator's queue")
	}
	h.assistant.ResumeProcessingFrames()
}

// awaitMessageCount waits for the conversation to hold n messages, which is how
// a test knows a data frame was handled before it sends a system frame that
// would otherwise overtake it.
func awaitMessageCount(t *testing.T, convo *frames.LLMContext, n int) {
	t.Helper()
	if !waitFor(3*time.Second, func() bool { return len(convo.Messages()) == n }) {
		t.Fatalf("conversation = %+v, want %d messages", convo.Messages(), n)
	}
}

func developer(text string) []frames.Message {
	return []frames.Message{{Role: frames.RoleDeveloper, Text: text}}
}

func appendFrame(text string, run bool) *frames.LLMMessagesAppendFrame {
	f := frames.NewLLMMessagesAppendFrame(developer(text))
	f.RunLLM = run
	return f
}

// TestRunSurvivesALaterResultThatDoesNotAskToRun is upstream's
// test_run_survives_a_later_result_that_does_not_ask_to_run: a burst runs
// inference once, even when its last result does not ask for it.
func TestRunSurvivesALaterResultThatDoesNotAskToRun(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.task.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: "1", Name: "lookup"}}))
	h.task.QueueFrame(inProgressAsync("1", "lookup"))
	if !waitFor(3*time.Second, func() bool { return len(asyncKinds(convo.Messages())) == 1 }) {
		t.Fatalf("conversation = %+v, want the call started", convo.Messages())
	}

	// Queued together, so the first result's push is held for the second.
	first := frames.NewFunctionCallResultFrame("1", "lookup", nil, `{"progress":"on it"}`)
	first.Properties = &frames.FunctionCallResultProperties{IsFinal: new(false), RunLLM: new(true)}
	second := frames.NewFunctionCallResultFrame("1", "lookup", nil, `{"progress":"still on it"}`)
	second.Properties = &frames.FunctionCallResultProperties{IsFinal: new(false), RunLLM: new(false)}
	h.queueTogether(t, first, second)

	awaitCount(t, h.generations, 1, "a burst whose last result does not ask to run")
	settle()
	if got := h.generations(); got != 1 {
		t.Errorf("generation ran %d times for one burst, want 1", got)
	}
}

// TestResultOwedThroughAnEmptyUserTurnIsPushedOnTheNextArrival is the tool
// result form of upstream's
// test_an_append_owed_through_an_empty_user_turn_is_pushed_on_the_next_arrival: a
// result asking to run while the user speaks is owed, a user turn with nothing
// in it brings no run, and the next result makes the push though it declines to
// run itself.
func TestResultOwedThroughAnEmptyUserTurnIsPushedOnTheNextArrival(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.task.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.ToolCall{
		{ID: "1", Name: "check_stock"}, {ID: "2", Name: "check_price"},
	}))
	// No group, so each result asks to run on its own.
	h.task.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "check_stock", nil, true, ""))
	h.task.QueueFrame(frames.NewFunctionCallInProgressFrame("2", "check_price", nil, true, ""))
	if !waitFor(3*time.Second, func() bool { return len(toolResults(convo.Messages())) == 2 }) {
		t.Fatalf("conversation = %+v, want both calls started", convo.Messages())
	}

	h.task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	h.task.QueueFrame(frames.NewFunctionCallResultFrame("1", "check_stock", nil, "in stock"))
	if !waitFor(3*time.Second, func() bool {
		return toolResults(convo.Messages())[0].Content == "in stock"
	}) {
		t.Fatalf("conversation = %+v, want the first result in place", convo.Messages())
	}
	settle()
	if got := h.generations(); got != 0 {
		t.Fatalf("generation ran %d times while the user was speaking, want 0", got)
	}

	h.task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	settle()
	h.task.QueueFrame(noRun(frames.NewFunctionCallResultFrame("2", "check_price", nil, "4.50")))

	awaitCount(t, h.generations, 1, "the result after an empty user turn")
	settle()
	if got := h.generations(); got != 1 {
		t.Errorf("generation ran %d times, want 1", got)
	}
}

// TestAnAppendAskingToRunWaitsForTheBotToStopSpeaking is upstream's
// test_an_append_asking_to_run_waits_for_the_bot_to_stop_speaking. Run
// mid-sentence, the model would answer a conversation that lacks its answer in
// progress.
func TestAnAppendAskingToRunWaitsForTheBotToStopSpeaking(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	h.task.QueueFrame(appendFrame("Backend: the coffee order is ready.", true))
	awaitMessageCount(t, convo, 1)
	settle()
	if got := h.generations(); got != 0 {
		t.Fatalf("generation ran %d times while the bot was speaking, want 0", got)
	}

	h.task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	awaitCount(t, h.generations, 1, "the bot having stopped")
	settle()
	if got := h.generations(); got != 1 {
		t.Errorf("generation ran %d times, want 1", got)
	}
}

// TestAnAppendAskingToRunWhileTheUserSpeaksIsSettledByTheTurnsRun is upstream's
// test_an_append_asking_to_run_while_the_user_speaks_is_settled_by_the_turns_run:
// the run the user's turn brings sees the conversation as it stands, so nothing
// is owed after it.
func TestAnAppendAskingToRunWhileTheUserSpeaksIsSettledByTheTurnsRun(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	h.task.QueueFrame(appendFrame("Backend: done.", true))
	awaitMessageCount(t, convo, 1)

	h.task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	h.task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	h.task.QueueFrame(appendFrame("Backend (thinking): ...", false))
	awaitMessageCount(t, convo, 2)
	settle()

	if got := h.generations(); got != 0 {
		t.Errorf("generation ran %d times after the turn's own run had started, want 0", got)
	}
}

// TestAnAppendOwedThroughAnEmptyUserTurnIsPushedOnTheNextArrival is upstream's
// test of the same name: a user turn with nothing in it brings no run, so the
// push is still owed and the next append makes it.
func TestAnAppendOwedThroughAnEmptyUserTurnIsPushedOnTheNextArrival(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	h.task.QueueFrame(appendFrame("Backend: done.", true))
	awaitMessageCount(t, convo, 1)
	settle()
	if got := h.generations(); got != 0 {
		t.Fatalf("generation ran %d times while the user was speaking, want 0", got)
	}

	h.task.QueueFrame(frames.NewUserStoppedSpeakingFrame())
	settle()
	h.task.QueueFrame(appendFrame("Backend (thinking): ...", false))
	awaitMessageCount(t, convo, 2)

	awaitCount(t, h.generations, 1, "the append after an empty user turn")
	settle()
	if got := h.generations(); got != 1 {
		t.Errorf("generation ran %d times, want 1", got)
	}
}

// TestAppendsQueuedTogetherRunInferenceOnce is upstream's
// test_appends_queued_together_run_inference_once: a burst of messages is one
// run, whatever the last one asks.
func TestAppendsQueuedTogetherRunInferenceOnce(t *testing.T) {
	convo := frames.NewLLMContext("")
	h := newAssistantHarness(t, convo)

	h.queueTogether(t,
		appendFrame("Backend: one.", true),
		appendFrame("Backend: two.", true),
		appendFrame("Backend (thinking): three.", false),
	)

	awaitCount(t, h.generations, 1, "a burst of appends")
	settle()
	if got := h.generations(); got != 1 {
		t.Errorf("generation ran %d times for one burst, want 1", got)
	}
	if got := len(convo.Messages()); got != 3 {
		t.Errorf("conversation holds %d messages, want 3", got)
	}
}

// TestOnlyASynchronousCallHoldsTheModel checks an asynchronous call in flight,
// which has its placeholder in the context, does not count as synchronous,
// while a synchronous one, and one announced but not yet started, do.
func TestOnlyASynchronousCallHoldsTheModel(t *testing.T) {
	cases := []struct {
		name string
		feed []frames.Frame
		want bool
	}{
		{"asynchronous", []frames.Frame{
			frames.NewFunctionCallInProgressFrame("1", "lookup", nil, false, "g1"),
		}, false},
		{"synchronous", []frames.Frame{
			frames.NewFunctionCallInProgressFrame("1", "lookup", nil, true, "g1"),
		}, true},
		{"announced", []frames.Frame{
			frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: "1", Name: "lookup"}}),
		}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newAssistantHarness(t, frames.NewLLMContext(""))
			h.task.QueueFrames(tc.feed)
			if !waitFor(3*time.Second, h.assistant.HasFunctionCallsInProgress) {
				t.Fatal("the call never reached the aggregator")
			}
			if got := h.assistant.HasSynchronousFunctionCallsInProgress(); got != tc.want {
				t.Errorf("synchronous call in progress = %t, want %t", got, tc.want)
			}
		})
	}
}
