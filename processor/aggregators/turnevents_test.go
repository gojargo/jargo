package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter"
	anthropicadapter "github.com/gojargo/jargo/adapter/anthropic"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for what an aggregator reports about the turns it assembles: the events
// something outside the pipeline listens to, and the frames the pipeline itself
// is told.

// turnLog collects what a pair reports, so a test can read it once the pipeline
// has settled.
type turnLog struct {
	mu        sync.Mutex
	assistant []aggregators.AssistantTurnStopped
	started   int
	user      []aggregators.UserTurnStopped
	messages  []aggregators.UserTurnMessageAdded
}

func (l *turnLog) assistantTurns() []aggregators.AssistantTurnStopped {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]aggregators.AssistantTurnStopped(nil), l.assistant...)
}

func (l *turnLog) userTurns() []aggregators.UserTurnStopped {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]aggregators.UserTurnStopped(nil), l.user...)
}

func (l *turnLog) userMessages() []aggregators.UserTurnMessageAdded {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]aggregators.UserTurnMessageAdded(nil), l.messages...)
}

func (l *turnLog) assistantStarts() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.started
}

// watchAssistant attaches to the assistant half's turn events.
func watchAssistant(a *aggregators.AssistantAggregator) *turnLog {
	l := &turnLog{}
	// The event carries no value, so it is attached to directly rather than
	// through the typed helper.
	a.Events().Add(aggregators.EventAssistantTurnStarted, func(context.Context, any, ...any) {
		l.mu.Lock()
		l.started++
		l.mu.Unlock()
	})
	events.On(a.Events(), aggregators.EventAssistantTurnStopped,
		func(_ context.Context, t aggregators.AssistantTurnStopped) {
			l.mu.Lock()
			l.assistant = append(l.assistant, t)
			l.mu.Unlock()
		})
	return l
}

// settle gives the event handlers, which run off the frame path, a moment to run.
func settle() { time.Sleep(200 * time.Millisecond) }

// TestAssistantReportsTheTurnItWrote checks that a finished bot turn is reported
// with what it said and when it began.
func TestAssistantReportsTheTurnItWrote(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	log := watchAssistant(pair.Assistant())

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	<-runDone
	settle()

	if got := log.assistantStarts(); got != 1 {
		t.Errorf("assistant turn started %d times, want once", got)
	}
	got := log.assistantTurns()
	if len(got) != 1 {
		t.Fatalf("assistant turns reported = %d, want one", len(got))
	}
	if got[0].Content != "Hello there." {
		t.Errorf("turn content = %q, want %q", got[0].Content, "Hello there.")
	}
	if got[0].Interrupted {
		t.Error("a turn that finished should not report as interrupted")
	}
	if got[0].Timestamp == "" {
		t.Error("the turn carries no timestamp")
	}
}

// TestAssistantReportsAnInterruptedTurn checks that a turn cut off mid-sentence
// says so, and carries what the bot had said by then.
func TestAssistantReportsAnInterruptedTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	log := watchAssistant(pair.Assistant())

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Let me check"))
	// The interruption is a system frame, so it overtakes anything still queued.
	// Letting the turn's own frames land first is what makes this an
	// interruption of a turn rather than one arriving before it.
	settle()
	task.QueueFrame(frames.NewInterruptionFrame())
	task.StopWhenDone()
	<-runDone
	settle()

	got := log.assistantTurns()
	if len(got) != 1 {
		t.Fatalf("assistant turns reported = %d, want one", len(got))
	}
	if !got[0].Interrupted {
		t.Error("a turn cut off should report as interrupted")
	}
	if got[0].Content != "Let me check" {
		t.Errorf("turn content = %q, want what the bot had said", got[0].Content)
	}
}

// TestAssistantAnnouncesTheTurnToThePipeline checks that a finished turn is also
// announced as a frame, both ways, so processors on either side hear it without
// having to attach a handler.
func TestAssistantAnnouncesTheTurnToThePipeline(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	seen := make(chan frames.Frame, 64)
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	turnFrame := awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextAssistantTurnFrame)
		return ok
	}, "the assistant turn frame")
	got, ok := turnFrame.(*frames.LLMContextAssistantTurnFrame)
	if !ok {
		t.Fatalf("frame = %T, want an assistant turn frame", turnFrame)
	}
	if got.Text != "Hello there." {
		t.Errorf("turn frame text = %q, want %q", got.Text, "Hello there.")
	}

	task.StopWhenDone()
	<-runDone
}

// TestAssistantStampsTheMessageItWrote checks that writing the turn to the
// conversation is followed by the moment it was written, for anything keeping a
// transcript alongside the conversation.
func TestAssistantStampsTheMessageItWrote(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	seen := make(chan frames.Frame, 64)
	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	stamp := awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextAssistantTimestampFrame)
		return ok
	}, "the assistant timestamp frame")
	got, ok := stamp.(*frames.LLMContextAssistantTimestampFrame)
	if !ok {
		t.Fatalf("frame = %T, want an assistant timestamp frame", stamp)
	}
	if got.Timestamp == "" {
		t.Error("the timestamp frame carries no timestamp")
	}

	task.StopWhenDone()
	<-runDone
}

// TestUserReportsTheTurnItCollected checks that the user's turn is reported with
// everything they said, and that the write to the conversation is reported too.
func TestUserReportsTheTurnItCollected(t *testing.T) {
	convo := frames.NewLLMContext("system")
	// A model-free stop chain, so what closes the turn is not left to whether
	// the end-of-turn model loads: this covers a turn nothing closed, reported
	// because the session ended, and the defaults would close it themselves.
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
		},
	}))
	log := &turnLog{}
	events.On(pair.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, tr aggregators.UserTurnStopped) {
			log.mu.Lock()
			log.user = append(log.user, tr)
			log.mu.Unlock()
		})
	events.On(pair.User().Events(), aggregators.EventUserTurnMessageAdded,
		func(_ context.Context, m aggregators.UserTurnMessageAdded) {
			log.mu.Lock()
			log.messages = append(log.messages, m)
			log.mu.Unlock()
		})

	task := pipeline.NewWorker(pipeline.New(pair.User()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	tf := frames.NewTranscriptionFrame("hello there", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)
	task.StopWhenDone()
	<-runDone
	settle()

	msgs := log.userMessages()
	if len(msgs) != 1 || msgs[0].Content != "hello there" {
		t.Fatalf("messages written = %+v, want one saying 'hello there'", msgs)
	}
	// The default strategies opened the turn on the transcript, and the session
	// ended before any of them closed it, so the end of the session is what
	// reports it and no strategy is named.
	got := log.userTurns()
	if len(got) != 1 {
		t.Fatalf("user turns reported = %+v, want exactly one", got)
	}
	if got[0].Strategy != nil {
		t.Errorf("turn reported strategy %v, want none: nothing decided it was over", got[0].Strategy)
	}
	if got[0].Content != "hello there" {
		t.Errorf("turn content = %q, want %q", got[0].Content, "hello there")
	}
}

// assistantTurnsFor runs frames through an assistant aggregator and returns the
// turns it reported, with the conversation it wrote.
func assistantTurnsFor(t *testing.T, in ...frames.Frame) ([]aggregators.AssistantTurnStopped, *frames.LLMContext) {
	t.Helper()
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	log := watchAssistant(pair.Assistant())

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrames(in)
	task.StopWhenDone()
	<-runDone
	settle()
	return log.assistantTurns(), convo
}

// Ported from upstream (test_turn_completion_markers_stripped_from_transcript).
// The markers a gated model prefixes its replies with arrive as marker frames,
// and do not reach a transcript.
func TestTurnCompletionMarkersAreStrippedFromTheReport(t *testing.T) {
	complete := frames.NewLLMMarkerFrame(frames.UserTurnCompleteMarker)
	complete.AppendToContextImmediately = false
	got, _ := assistantTurnsFor(t,
		frames.NewLLMFullResponseStartFrame(),
		complete,
		frames.NewLLMTextFrame("Hello there!"),
		frames.NewLLMFullResponseEndFrame(),
	)
	if len(got) != 1 {
		t.Fatalf("assistant turns reported = %d, want one", len(got))
	}
	if got[0].Content != "Hello there!" {
		t.Errorf("reported content = %q, want the marker stripped", got[0].Content)
	}
}

// Ported from upstream (the incomplete half of the same test). A marker that is
// the whole reply is a context message of its own, and nothing was said.
func TestAMarkerOnItsOwnReportsNothingSaid(t *testing.T) {
	short := frames.NewLLMMarkerFrame(frames.UserTurnIncompleteShortMarker)
	short.AppendToContextImmediately = true
	got, convo := assistantTurnsFor(t,
		frames.NewLLMFullResponseStartFrame(),
		short,
		frames.NewLLMFullResponseEndFrame(),
	)
	if len(got) != 1 {
		t.Fatalf("assistant turns reported = %d, want one", len(got))
	}
	if got[0].Content != "" {
		t.Errorf("reported content = %q, want nothing", got[0].Content)
	}
	msgs := convo.Messages()
	if len(msgs) == 0 || msgs[len(msgs)-1].Text != frames.UserTurnIncompleteShortMarker {
		t.Errorf("context messages = %+v, want the marker as the last", msgs)
	}
}

// Ported from upstream (test_marker_characters_kept_without_marker_frames).
// Marker characters in plain text, with no marker frame, are content: the
// report says what the context says.
func TestMarkerCharactersKeptWithoutMarkerFrames(t *testing.T) {
	const reply = "○ Option A ● Option B"
	got, convo := assistantTurnsFor(t,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame(reply),
		frames.NewLLMFullResponseEndFrame(),
	)
	if len(got) != 1 || got[0].Content != reply {
		t.Fatalf("reported %+v, want one turn saying %q", got, reply)
	}
	msgs := convo.Messages()
	if len(msgs) == 0 || msgs[len(msgs)-1].Text != got[0].Content {
		t.Errorf("context messages = %+v, want the same text as reported", msgs)
	}
}

// TestAssistantRecordsAThought checks that a reasoning model's thinking is
// reported and, when the provider asks for it back, kept in the conversation as
// that provider's own message.
func TestAssistantRecordsAThought(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	var mu sync.Mutex
	var thoughts []aggregators.AssistantThought
	events.On(pair.Assistant().Events(), aggregators.EventAssistantThought,
		func(_ context.Context, th aggregators.AssistantThought) {
			mu.Lock()
			thoughts = append(thoughts, th)
			mu.Unlock()
		})

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	start := frames.NewLLMThoughtStartFrame()
	start.AppendToContext = true
	start.LLM = "anthropic"
	task.QueueFrame(start)
	task.QueueFrame(frames.NewLLMThoughtTextFrame("Let me work"))
	task.QueueFrame(frames.NewLLMThoughtTextFrame(" this out."))
	end := frames.NewLLMThoughtEndFrame()
	end.Signature = "sig"
	task.QueueFrame(end)
	task.StopWhenDone()
	<-runDone
	settle()

	mu.Lock()
	defer mu.Unlock()
	if len(thoughts) != 1 {
		t.Fatalf("thoughts reported = %d, want one", len(thoughts))
	}
	if thoughts[0].Content != "Let me work this out." {
		t.Errorf("thought = %q, want the chunks joined as they arrived", thoughts[0].Content)
	}
	msgs := convo.Messages()
	if len(msgs) != 1 || !msgs[0].IsLLMSpecific() || msgs[0].LLM != "anthropic" {
		t.Fatalf("context messages = %+v, want one written as the provider's own", msgs)
	}
}

// TestAThoughtRoundTripsToAnthropic checks a thought the Anthropic service asked
// back is kept in a form its adapter reads: on the next request it is a thinking
// block, signature intact, ahead of the reply it preceded in the same turn.
func TestAThoughtRoundTripsToAnthropic(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.AddUserMessage("Is there tea?")
	pair := aggregators.New(convo)

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	start := frames.NewLLMThoughtStartFrame()
	start.AppendToContext = true
	start.LLM = "anthropic"
	task.QueueFrame(start)
	task.QueueFrame(frames.NewLLMThoughtTextFrame("The menu has tea."))
	end := frames.NewLLMThoughtEndFrame()
	end.Signature = "sig"
	task.QueueFrame(end)
	task.QueueFrame(frames.NewLLMTextFrame("We have tea."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	<-runDone
	settle()

	params, err := (&anthropicadapter.Adapter{}).LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("LLMInvocationParams: %v", err)
	}
	if len(params.Messages) != 2 {
		t.Fatalf("messages = %+v, want the user turn and one assistant turn", params.Messages)
	}
	content := params.Messages[1].Content
	if len(content) != 2 {
		t.Fatalf("assistant content = %+v, want the thinking block and the reply", content)
	}
	if th := content[0].OfThinking; th == nil || th.Thinking != "The menu has tea." || th.Signature != "sig" {
		t.Errorf("block 0 = %+v, want the thinking block with its signature", content[0])
	}
	if tx := content[1].OfText; tx == nil || tx.Text != "We have tea." {
		t.Errorf("block 1 = %+v, want the reply", content[1])
	}
}

// TestAThoughtIsNotSpoken checks that reasoning stays out of what the bot said.
// A thought is the model reasoning with itself, so it must never reach the turn
// the conversation records as the reply.
func TestAThoughtIsNotSpoken(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	log := watchAssistant(pair.Assistant())

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMThoughtStartFrame())
	task.QueueFrame(frames.NewLLMThoughtTextFrame("thinking hard"))
	task.QueueFrame(frames.NewLLMThoughtEndFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello there."))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	<-runDone
	settle()

	got := log.assistantTurns()
	if len(got) != 1 || got[0].Content != "Hello there." {
		t.Fatalf("assistant turns = %+v, want one saying only the reply", got)
	}
}

// TestUserAnnouncesTheTurnToThePipeline checks that a user turn written to the
// conversation is announced as a frame, both ways, the user's side of the
// assistant turn frame, and ahead of the conversation it was written to.
func TestUserAnnouncesTheTurnToThePipeline(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
		},
	}))

	var mu sync.Mutex
	var down, up []frames.Frame
	task := pipeline.NewWorker(pipeline.New(pair.User()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	tf := frames.NewTranscriptionFrame("hello there", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)
	// The stop is a system frame and would overtake the transcription still
	// queued, so the pause lets the transcription reach the turn first.
	time.Sleep(50 * time.Millisecond)
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	// The speech timeout closes the turn, writing it, before the session ends.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		announced := false
		for _, f := range up {
			if _, ok := f.(*frames.LLMContextUserTurnFrame); ok {
				announced = true
			}
		}
		mu.Unlock()
		if announced {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	task.StopWhenDone()
	<-runDone
	settle()

	mu.Lock()
	defer mu.Unlock()
	turnAt, contextAt := -1, -1
	for i, f := range down {
		switch fr := f.(type) {
		case *frames.LLMContextUserTurnFrame:
			if turnAt < 0 {
				turnAt = i
			}
			if fr.Text != "hello there" || fr.Timestamp == "" {
				t.Errorf("turn frame = %v, want the turn's text and when it began", fr)
			}
		case *frames.LLMContextFrame:
			if contextAt < 0 {
				contextAt = i
			}
		}
	}
	if turnAt < 0 {
		t.Fatal("no user turn frame went downstream")
	}
	if contextAt >= 0 && contextAt < turnAt {
		t.Error("the conversation went out ahead of the turn frame announcing what was written to it")
	}
	found := false
	for _, f := range up {
		if _, ok := f.(*frames.LLMContextUserTurnFrame); ok {
			found = true
		}
	}
	if !found {
		t.Error("no user turn frame went upstream")
	}
}
