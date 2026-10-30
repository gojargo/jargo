package aggregators_test

import (
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

// eagerTurn drives one turn through an aggregator running the eager strategies
// and returns the frames the aggregator pushed.
//
// The frames are sent with a pause between them, because the decisions under
// test are made across frame boundaries: a speculation starts on one and is
// settled by another.
func eagerTurn(t *testing.T, send ...frames.Frame) (*frames.LLMContext, []frames.Frame) {
	t.Helper()
	return eagerTurnWith(t, nil, send...)
}

// eagerTurnWith is eagerTurn with the user muted by the given strategies.
func eagerTurnWith(
	t *testing.T, mute []turns.MuteStrategy, send ...frames.Frame,
) (*frames.LLMContext, []frames.Frame) {
	t.Helper()

	convo := frames.NewLLMContext("Be brief.")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies:     turns.EagerStrategies(turns.EagerStrategiesConfig{}),
		MuteStrategies: mute,
	}))
	task, seen, runDone := runPair(t, pair.User())

	for _, f := range send {
		task.QueueFrame(f)
		time.Sleep(60 * time.Millisecond)
	}
	time.Sleep(400 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	var got []frames.Frame
	for {
		select {
		case f := <-seen:
			got = append(got, f)
		default:
			return convo, got
		}
	}
}

// eagerFrame is the frame a service pushes when it predicts the turn has ended.
func eagerFrame(text string) *frames.EagerTranscriptionFrame {
	return frames.NewEagerTranscriptionFrame(text, "u", "ts")
}

// contextFrames are the inferences the aggregator asked for, in order.
func contextFrames(fs []frames.Frame) []*frames.LLMContextFrame {
	var out []*frames.LLMContextFrame
	for _, f := range fs {
		if cf, ok := f.(*frames.LLMContextFrame); ok {
			out = append(out, cf)
		}
	}
	return out
}

// countOf is how many frames of a kind the aggregator pushed.
func countOf[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

// TestEagerEndOfTurnRunsInferenceWithoutTouchingTheConversation covers the whole
// point of speculating: the model is asked to answer a turn that has not ended,
// and the conversation records nothing, because the turn may yet continue.
func TestEagerEndOfTurnRunsInferenceWithoutTouchingTheConversation(t *testing.T) {
	convo, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("book a flight"),
	)

	cfs := contextFrames(got)
	if len(cfs) != 1 {
		t.Fatalf("inferences = %d, want 1", len(cfs))
	}
	if !cfs[0].Speculation {
		t.Error("the inference is not marked speculative")
	}
	if cfs[0].Context == convo {
		t.Error("the inference ran against the real conversation, want a provisional copy")
	}
	if msgs := cfs[0].Context.Messages(); len(msgs) == 0 || msgs[len(msgs)-1].Text != "book a flight" {
		t.Errorf("the provisional conversation does not end with the eager transcript: %v", msgs)
	}

	// The real conversation is untouched: the turn has not ended.
	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("the conversation holds %v, want nothing until the turn ends", msgs)
	}

	// The turn the inference answers goes ahead of it, as a speculation.
	turnAt, inferenceAt := -1, -1
	for i, f := range got {
		switch fr := f.(type) {
		case *frames.LLMContextUserTurnFrame:
			if turnAt < 0 {
				turnAt = i
				if fr.Text != "book a flight" || !fr.Speculation {
					t.Errorf("turn = (%q, %t), want (book a flight, true)", fr.Text, fr.Speculation)
				}
			}
		case *frames.LLMContextFrame:
			if inferenceAt < 0 {
				inferenceAt = i
			}
		}
	}
	if turnAt < 0 || turnAt > inferenceAt {
		t.Errorf("the turn landed at %d and the inference at %d, want the turn first", turnAt, inferenceAt)
	}
}

// userTurns are the user turns the aggregator announced, by text and whether
// each was a speculation.
func userTurns(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if tf, ok := f.(*frames.LLMContextUserTurnFrame); ok {
			out = append(out, fmt.Sprintf("%s/%t", tf.Text, tf.Speculation))
		}
	}
	return out
}

// TestMatchingTranscriptEndsTheTurnAndKeepsTheReply covers the happy path: the
// committed transcript says what the prediction said, so the reply already
// generated stands and the model is not asked a second time.
func TestMatchingTranscriptEndsTheTurnAndKeepsTheReply(t *testing.T) {
	convo, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("book a flight"),
		frames.NewTranscriptionFrame("book a flight", "u", "ts"),
		frames.NewProposedUserStoppedSpeakingFrame(),
	)

	if n := countOf[*frames.EagerEndOfTurnCancelFrame](got); n != 0 {
		t.Errorf("withdrawals = %d, want none: the prediction was right", n)
	}
	cfs := contextFrames(got)
	if len(cfs) != 1 {
		t.Fatalf("inferences = %d, want 1: the confirmed turn is not answered twice", len(cfs))
	}
	if !cfs[0].Speculation {
		t.Error("the one inference is not the speculative one")
	}
	// The turn end is what releases the held reply.
	if n := countOf[*frames.UserStoppedSpeakingFrame](got); n != 1 {
		t.Errorf("turn ends = %d, want 1", n)
	}
	if msgs := convo.Messages(); len(msgs) != 1 || msgs[0].Text != "book a flight" {
		t.Errorf("the conversation holds %v, want the committed transcript", msgs)
	}
	// The turn is broadcast as a speculation, then as written once it ends.
	if got, want := userTurns(got), []string{"book a flight/true", "book a flight/false"}; !slices.Equal(got, want) {
		t.Errorf("user turns = %v, want %v", got, want)
	}
}

// TestDifferingTranscriptWithdrawsTheSpeculation covers the prediction being
// wrong: the reply answered something the user did not finish saying, so it is
// withdrawn and the committed transcript is answered instead.
func TestDifferingTranscriptWithdrawsTheSpeculation(t *testing.T) {
	convo, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("i want to cancel"),
		frames.NewTranscriptionFrame("i want to cancel, actually reschedule it", "u", "ts"),
		frames.NewProposedUserStoppedSpeakingFrame(),
	)

	if n := countOf[*frames.EagerEndOfTurnCancelFrame](got); n != 1 {
		t.Errorf("withdrawals = %d, want 1", n)
	}
	cfs := contextFrames(got)
	if len(cfs) != 2 {
		t.Fatalf("inferences = %d, want 2: the speculative one, then the committed one", len(cfs))
	}
	if !cfs[0].Speculation || cfs[1].Speculation {
		t.Errorf("speculative flags = %v/%v, want true then false", cfs[0].Speculation, cfs[1].Speculation)
	}
	if msgs := cfs[1].Context.Messages(); len(msgs) == 0 ||
		msgs[len(msgs)-1].Text != "i want to cancel, actually reschedule it" {
		t.Errorf("the second inference does not carry the committed transcript: %v", msgs)
	}

	// Only the committed transcript reaches the conversation.
	if msgs := convo.Messages(); len(msgs) != 1 ||
		msgs[0].Text != "i want to cancel, actually reschedule it" {
		t.Errorf("the conversation holds %v, want only the committed transcript", msgs)
	}
}

// TestFormattingDifferencesAreToleratedByDefault covers the default match
// policy: a service that capitalizes and punctuates the committed transcript has
// not said anything different, so the reply stands.
func TestFormattingDifferencesAreToleratedByDefault(t *testing.T) {
	_, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("book a flight"),
		frames.NewTranscriptionFrame("Book a flight.", "u", "ts"),
		frames.NewProposedUserStoppedSpeakingFrame(),
	)

	if n := countOf[*frames.EagerEndOfTurnCancelFrame](got); n != 0 {
		t.Errorf("withdrawals = %d, want none: only the formatting differed", n)
	}
	if cfs := contextFrames(got); len(cfs) != 1 {
		t.Errorf("inferences = %d, want 1: the reply still applies", len(cfs))
	}
}

// TestTurnWithoutAnEagerPredictionBehavesNormally covers a service that predicts
// nothing: the strategies are the external ones with nothing speculating, so the
// turn runs exactly as it would without them.
func TestTurnWithoutAnEagerPredictionBehavesNormally(t *testing.T) {
	convo, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		frames.NewTranscriptionFrame("book a flight", "u", "ts"),
		frames.NewProposedUserStoppedSpeakingFrame(),
	)

	if n := countOf[*frames.EagerEndOfTurnCancelFrame](got); n != 0 {
		t.Errorf("withdrawals = %d, want none: nothing was ever speculated", n)
	}
	cfs := contextFrames(got)
	if len(cfs) != 1 {
		t.Fatalf("inferences = %d, want 1", len(cfs))
	}
	if cfs[0].Speculation {
		t.Error("the inference is marked speculative, but nothing predicted the turn")
	}
	if msgs := convo.Messages(); len(msgs) != 1 || msgs[0].Text != "book a flight" {
		t.Errorf("the conversation holds %v, want the transcript", msgs)
	}
}

// TestWithdrawalPrecedesTheTurnEnd covers the ordering the whole design rests
// on: a withdrawal must reach the model service ahead of the turn end that
// follows it. Arriving after, it would find the reply already released and the
// bot would speak an answer to something the user never said.
func TestWithdrawalPrecedesTheTurnEnd(t *testing.T) {
	_, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("i want to cancel"),
		frames.NewTranscriptionFrame("i want to reschedule", "u", "ts"),
		frames.NewProposedUserStoppedSpeakingFrame(),
	)

	withdrawal, turnEnd := -1, -1
	for i, f := range got {
		switch f.(type) {
		case *frames.EagerEndOfTurnCancelFrame:
			if withdrawal < 0 {
				withdrawal = i
			}
		case *frames.UserStoppedSpeakingFrame:
			if turnEnd < 0 {
				turnEnd = i
			}
		}
	}
	if withdrawal < 0 {
		t.Fatalf("the mismatched prediction was never withdrawn")
	}
	if turnEnd >= 0 && withdrawal > turnEnd {
		t.Error("the withdrawal followed the turn end, want it ahead: the reply would already be released")
	}
}

// TestEagerTranscriptIsNotPushedDownstream checks the transcript of a predicted
// end of turn is consumed by the aggregator. It carries text, so anything
// downstream that speaks or aggregates text must never see it.
func TestEagerTranscriptIsNotPushedDownstream(t *testing.T) {
	_, got := eagerTurn(t,
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("book a flight"),
	)
	if n := countOf[*frames.EagerTranscriptionFrame](got); n != 0 {
		t.Errorf("eager transcripts pushed downstream = %d, want none", n)
	}
}

// muteAfterTurnStart mutes the user from the frame after the turn starts on, so
// the turn is under way when the eager transcript arrives.
type muteAfterTurnStart struct{ started bool }

func (m *muteAfterTurnStart) ShouldMute(f frames.Frame) bool {
	if _, ok := f.(*frames.ProposedUserStartedSpeakingFrame); ok {
		m.started = true
	}
	return m.started
}

// TestMutedEagerTranscriptDoesNotSpeculate checks the transcript of a predicted
// end of turn is user input like any other transcript: while the user is muted
// it is dropped, and no speculative inference runs on it.
func TestMutedEagerTranscriptDoesNotSpeculate(t *testing.T) {
	convo, got := eagerTurnWith(t, []turns.MuteStrategy{&muteAfterTurnStart{}},
		frames.NewProposedUserStartedSpeakingFrame(),
		eagerFrame("book a flight"),
	)
	if cfs := contextFrames(got); len(cfs) != 0 {
		t.Errorf("inferences = %d, want none: the eager transcript arrived while muted", len(cfs))
	}
	if msgs := convo.Messages(); len(msgs) != 0 {
		t.Errorf("the conversation holds %v, want nothing", msgs)
	}
}
