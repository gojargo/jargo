package aggregators_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

func TestUserAggregatorTriggersLLMOnFinal(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	triggered := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case triggered <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// An interim transcription must not trigger the LLM.
	task.QueueFrame(frames.NewInterimTranscriptionFrame("hel", "u", "ts"))
	// A finalized transcription ends the turn and triggers the LLM.
	tf := frames.NewTranscriptionFrame("hello there", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)

	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("user aggregator did not emit an LLMContextFrame")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || msgs[0].Text != "hello there" {
		t.Fatalf("context messages = %+v, want one user 'hello there'", msgs)
	}

	task.StopWhenDone()
	<-runDone
}

func TestUserAggregatorTurnTakingGatesOnEndOfTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 2 * time.Second,
	}))

	triggered := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case triggered <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// A finalized transcript without an end-of-turn must NOT trigger the LLM.
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	tf := frames.NewTranscriptionFrame("hello there", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)

	select {
	case <-triggered:
		t.Fatal("LLM triggered before end-of-turn")
	case <-time.After(300 * time.Millisecond):
	}

	// The user falls silent. A turn is never finalized while they are still
	// audibly speaking, since a verdict that lands mid-speech is already stale,
	// so the VAD stop has to come first as it does in a real pipeline.
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	// The end-of-turn decision, with a finalized transcript in hand, triggers it.
	task.QueueFrame(frames.NewUserTurnInferenceCompletedFrame())
	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("LLM not triggered after end-of-turn")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Text != "hello there" {
		t.Fatalf("context messages = %+v, want one user 'hello there'", msgs)
	}
	task.StopWhenDone()
	<-runDone
}

// The transcript that ends a turn belongs to that turn's message. It is the
// last thing the user said, and losing it means answering a question they only
// half asked: the strategies run inside this aggregator so the transcript is
// folded in before any of them can call the turn over.
func TestUserAggregatorKeepsTheTranscriptThatEndsTheTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{
				UserSpeechTimeout: 20 * time.Millisecond,
			})},
		},
		StopTimeout: 2 * time.Second,
	}))

	triggered := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case triggered <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	first := frames.NewTranscriptionFrame("j'ai un ami qui me demande", "u", "ts")
	first.Finalized = true
	task.QueueFrame(first)
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	// The turn ends on this transcript, so it must be in the message.
	last := frames.NewTranscriptionFrame("c'est qui Clovis", "u", "ts")
	last.Finalized = true
	task.QueueFrame(last)

	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("LLM not triggered")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 {
		t.Fatalf("context messages = %+v, want one", msgs)
	}
	if !strings.Contains(msgs[0].Text, "c'est qui Clovis") {
		t.Fatalf("user message = %q, want it to carry the transcript that ended the turn", msgs[0].Text)
	}
	task.StopWhenDone()
	<-runDone
}

// A streaming STT is entitled to deliver one utterance as more than one final
// transcript. The tail of it arrives after the turn it belongs to has closed,
// and it must not be answered on its own: with turn taking, only the turn
// controller decides when what the user said becomes a message. Answering the
// tail costs an inference, a second user message holding half a sentence, and,
// where the turn calls a tool, a second call to whatever is behind it.
func TestUserAggregatorDoesNotAnswerATranscriptOutsideATurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			// VAD alone opens turns, so the trailing transcript below opens none:
			// what it must not do is produce an answer without one.
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{
				UserSpeechTimeout: 20 * time.Millisecond,
			})},
		},
		StopTimeout: 3 * time.Second,
	}))

	runs := make(chan struct{}, 4)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				runs <- struct{}{}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// One turn, answered once.
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	first := frames.NewTranscriptionFrame("what time do you close", "u", "ts")
	first.Finalized = true
	task.QueueFrame(first)
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	select {
	case <-runs:
	case <-time.After(3 * time.Second):
		t.Fatal("the turn was never answered")
	}

	// The tail of the same utterance, transcribed late. No VAD onset stands
	// behind it, and the turn it belongs to has already been answered.
	tail := frames.NewTranscriptionFrame("on sundays", "u", "ts")
	tail.Finalized = true
	task.QueueFrame(tail)

	select {
	case <-runs:
		t.Fatal("a transcript arriving outside a turn was answered on its own")
	case <-time.After(300 * time.Millisecond):
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || msgs[0].Text != "what time do you close" {
		t.Fatalf("messages = %+v, want one user 'what time do you close'", msgs)
	}

	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

func TestAssistantAggregatorCommitsPartialOnInterruption(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	task.QueueFrame(frames.NewLLMTextFrame("wor"))
	// Let the text be aggregated before the interruption arrives.
	time.Sleep(300 * time.Millisecond)
	task.QueueFrame(frames.NewInterruptionFrame())
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello wor" {
		t.Fatalf("context messages = %+v, want one assistant 'Hello wor'", msgs)
	}
}

func TestAssistantAggregatorCollectsResponse(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo)

	task := pipeline.NewTask(pipeline.New(pair.Assistant()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("Hello "))
	task.QueueFrame(frames.NewLLMTextFrame("world"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello world" {
		t.Fatalf("context messages = %+v, want one assistant 'Hello world'", msgs)
	}
}

// The speech that opens a turn is the turn's first words, and must survive the
// opening. A start strategy that holds out for a few words only decides once
// they have been transcribed, so by the time the turn opens they are already
// aggregated: clearing then would throw away exactly the speech that caused it,
// and the turn would close having produced nothing at all.
//
// Speech that must not count is dropped explicitly instead. Here the one-word
// barge-in is below the strategy's threshold, so the strategy asks for the
// aggregation to be reset and those words stay out of the conversation.
func TestUserAggregatorKeepsTheSpeechThatOpensTheTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewMinWordsStart(turns.MinWordsStartConfig{MinWords: 3})},
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{
				UserSpeechTimeout: 30 * time.Millisecond,
			})},
		},
		StopTimeout: 3 * time.Second,
	}))

	triggered := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case triggered <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The bot is talking, so it takes several words to barge in.
	task.QueueFrame(frames.NewBotStartedSpeakingFrame())

	// One word is not enough to interrupt: it opens no turn and must not end up
	// in the conversation either.
	short := frames.NewTranscriptionFrame("wait", "u", "ts")
	short.Finalized = true
	task.QueueFrame(short)
	time.Sleep(50 * time.Millisecond)

	// These words do open the turn, and they are the words it was opened for.
	long := frames.NewTranscriptionFrame("cancel the order", "u", "ts")
	long.Finalized = true
	task.QueueFrame(long)

	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("LLM never ran: the turn produced nothing")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser {
		t.Fatalf("messages = %+v, want one user message", msgs)
	}
	if msgs[0].Text != "cancel the order" {
		t.Fatalf("user text = %q, want %q", msgs[0].Text, "cancel the order")
	}

	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

// Deferred finalization exists so the answer can start being written while a
// separate judge is still deciding whether the turn is really over. The detector
// says there is enough to answer and inference begins; the judge finalizes
// later. Acting only on the finalization would mean waiting for the judge before
// starting at all, which is the delay the arrangement is meant to remove.
func TestUserAggregatorRunsInferenceBeforeDeferredFinalization(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop: []turns.StopStrategy{
				// The detector triggers inference but cannot finalize.
				turns.Deferred(turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{
					UserSpeechTimeout: 20 * time.Millisecond,
				})),
				// Only the judge finalizes.
				turns.NewExternalCompletionStop(),
			},
		},
		StopTimeout: 3 * time.Second,
	}))

	triggered := make(chan struct{}, 4)
	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.LLMContextFrame); ok {
				select {
				case triggered <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	tf := frames.NewTranscriptionFrame("what time do you close", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	// No completion verdict has been sent, so the judge has not finalized. The
	// answer must already be under way.
	select {
	case <-triggered:
	case <-time.After(3 * time.Second):
		t.Fatal("inference never started: it waited on a finalization that had not happened")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || msgs[0].Text != "what time do you close" {
		t.Fatalf("messages = %+v, want one user 'what time do you close'", msgs)
	}

	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}
}

// What the user said last is committed when the session ends. A transcript that
// arrived without the turn being finalized, because the call dropped or the
// scenario's last turn was the user's, would otherwise be lost with the
// processor, leaving a saved conversation ending on the bot's turn with no
// record that the user answered.
func TestUserAggregatorCommitsWhatIsHeldWhenTheSessionEnds(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
		},
		StopTimeout: 3 * time.Second,
	}))

	task := pipeline.NewTask(pipeline.New(pair.User()), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The user speaks, and the turn is never finalized before the end.
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2))
	tf := frames.NewTranscriptionFrame("actually make it two", "u", "ts")
	tf.Finalized = true
	task.QueueFrame(tf)

	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleUser || msgs[0].Text != "actually make it two" {
		t.Fatalf("messages = %+v, want one user 'actually make it two': the last thing "+
			"the user said was dropped at the end of the session", msgs)
	}
}
