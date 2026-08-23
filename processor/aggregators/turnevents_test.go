package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
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
	pair := aggregators.New(convo)
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

// TestTurnCompletionMarkersAreStrippedFromTheReport checks that the protocol
// markers a gated model prefixes its replies with do not reach a transcript. The
// conversation keeps them, since the model reads its own earlier verdicts back.
func TestTurnCompletionMarkersAreStrippedFromTheReport(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	log := watchAssistant(pair.Assistant())

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame(frames.UserTurnCompleteMarker + " Hello there!"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	<-runDone
	settle()

	got := log.assistantTurns()
	if len(got) != 1 {
		t.Fatalf("assistant turns reported = %d, want one", len(got))
	}
	if got[0].Content != "Hello there!" {
		t.Errorf("reported content = %q, want the marker stripped", got[0].Content)
	}
	// The conversation keeps the marker.
	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != frames.UserTurnCompleteMarker+" Hello there!" {
		t.Errorf("context messages = %+v, want the marker kept", msgs)
	}
}

// TestAMarkerOnItsOwnReportsNothingSaid checks the turn where the marker is the
// whole reply: the spoken answer was suppressed, so there is nothing to report
// as having been said.
func TestAMarkerOnItsOwnReportsNothingSaid(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)
	log := watchAssistant(pair.Assistant())

	task := pipeline.NewWorker(pipeline.New(pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame(frames.UserTurnIncompleteShortMarker))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()
	<-runDone
	settle()

	got := log.assistantTurns()
	if len(got) != 1 {
		t.Fatalf("assistant turns reported = %d, want one", len(got))
	}
	if got[0].Content != "" {
		t.Errorf("reported content = %q, want nothing", got[0].Content)
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
