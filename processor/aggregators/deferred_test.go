package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
)

// Tests for when a tool result runs the model again. A result arriving while the
// bot is mid-sentence must not start a second answer over the top of the first,
// so the re-run waits for the floor to be free.

// assistantUpstream runs an assistant aggregator and records the context frames
// it pushes upstream, which is how it asks for generation to run.
func assistantUpstream(t *testing.T, convo *frames.LLMContext) (*pipeline.Task, func() int, func()) {
	t.Helper()

	var (
		mu sync.Mutex
		n  int
	)
	pair := aggregators.New(convo)
	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{
		ReachedUpstreamFilter: pipeline.AnyFrame,
		OnReachedUpstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				mu.Lock()
				n++
				mu.Unlock()
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	count := func() int {
		mu.Lock()
		defer mu.Unlock()
		return n
	}
	stop := func() {
		t.Helper()
		task.StopWhenDone()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Fatal("task did not finish")
		}
	}
	return task, count, stop
}

// awaitCount waits for the generation count to reach n.
func awaitCount(t *testing.T, count func() int, n int, what string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for count() < n {
		if time.Now().After(deadline) {
			t.Fatalf("%s: generation ran %d times, want %d", what, count(), n)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// toolTurn is the frames a completed tool call produces, without the response
// text around it.
func toolTurn(id, name, result string) []frames.Frame {
	return []frames.Frame{
		frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: id, Name: name}}),
		inProgress(id, name, nil),
		frames.NewFunctionCallResultFrame(id, name, nil, result),
	}
}

// TestToolResultRunsGenerationWhenNobodyIsSpeaking checks the ordinary case: the
// floor is free, so the result runs the model straight away.
func TestToolResultRunsGenerationWhenNobodyIsSpeaking(t *testing.T) {
	task, count, stop := assistantUpstream(t, frames.NewLLMContext("system"))

	for _, f := range toolTurn("c1", "get_weather", "sunny") {
		task.QueueFrame(f)
	}

	awaitCount(t, count, 1, "a result with the floor free")
	stop()
}

// TestToolResultWaitsForTheBotToStopSpeaking checks a result that lands while
// the bot is mid-sentence defers the re-run rather than talking over it, and
// that the bot falling silent is what releases it.
func TestToolResultWaitsForTheBotToStopSpeaking(t *testing.T) {
	task, count, stop := assistantUpstream(t, frames.NewLLMContext("system"))

	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	for _, f := range toolTurn("c1", "get_weather", "sunny") {
		task.QueueFrame(f)
	}

	// Nothing should run while the bot holds the floor.
	time.Sleep(200 * time.Millisecond)
	if got := count(); got != 0 {
		t.Fatalf("generation ran %d times while the bot was speaking, want 0", got)
	}

	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	awaitCount(t, count, 1, "the bot having stopped")
	stop()
}

// TestSeveralResultsShareOneDeferredRun checks results arriving while the bot
// speaks accumulate into a single re-run. Answering once per result would have
// the bot reply several times to one turn.
func TestSeveralResultsShareOneDeferredRun(t *testing.T) {
	task, count, stop := assistantUpstream(t, frames.NewLLMContext("system"))

	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	task.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.ToolCall{
		{ID: "c1", Name: "get_weather"},
		{ID: "c2", Name: "get_traffic"},
	}))
	for _, f := range []frames.Frame{
		inProgress("c1", "get_weather", nil),
		inProgress("c2", "get_traffic", nil),
		frames.NewFunctionCallResultFrame("c1", "get_weather", nil, "sunny"),
		frames.NewFunctionCallResultFrame("c2", "get_traffic", nil, "clear"),
	} {
		task.QueueFrame(f)
	}

	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())
	awaitCount(t, count, 1, "two deferred results")

	// Give a second run the chance to appear before ruling it out.
	time.Sleep(200 * time.Millisecond)
	if got := count(); got != 1 {
		t.Errorf("generation ran %d times for two deferred results, want 1", got)
	}
	stop()
}

// TestDeferredRunStaysDeferredWhileTheUserSpeaks checks the re-run is not
// released into the user's turn. The user finishing runs the model anyway, so
// answering here would be a second generation over the top of that one.
func TestDeferredRunStaysDeferredWhileTheUserSpeaks(t *testing.T) {
	task, count, stop := assistantUpstream(t, frames.NewLLMContext("system"))

	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	for _, f := range toolTurn("c1", "get_weather", "sunny") {
		task.QueueFrame(f)
	}

	// The user cuts in, then the bot's speech ends: the floor is the user's.
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewBotStoppedSpeakingFrame())

	time.Sleep(200 * time.Millisecond)
	if got := count(); got != 0 {
		t.Fatalf("generation ran %d times while the user was speaking, want 0", got)
	}
	stop()
}

// TestSpeakingFramesTravelOn checks the aggregator only watches the speaking
// frames: everything downstream still sees them.
func TestSpeakingFramesTravelOn(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []string
	)
	pair := aggregators.New(frames.NewLLMContext("system"))
	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			switch f.(type) {
			case *frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame,
				*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame:
				mu.Lock()
				seen = append(seen, f.Name())
				mu.Unlock()
			}
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for _, f := range []frames.Frame{
		frames.NewBotStartedSpeakingFrame(),
		frames.NewUserStartedSpeakingFrame(),
		frames.NewUserStoppedSpeakingFrame(),
		frames.NewBotStoppedSpeakingFrame(),
	} {
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 4 {
		t.Errorf("the far end saw %v, want all four speaking frames", seen)
	}
}
