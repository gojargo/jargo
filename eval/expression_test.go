package eval

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// Expressive conversation reports what the TTS performed: where the delivery of
// the spoken text changes, and each sound and pause.
func TestTranslateExpressionFromBotOutput(t *testing.T) {
	s := &session{events: make(chan Event, eventBuffer)}
	output := func(segmentID uint64, textType frames.TextType, text string, delivery []string,
		status rtvi.SpokenStatus,
	) *Event {
		willBeSpoken := true
		return s.translate(msg(t, rtvi.TypeBotOutput, rtvi.BotOutputData{
			Text: text, OutputType: textType, SegmentID: &segmentID, WillBeSpoken: &willBeSpoken,
			SpokenStatus: status, Delivery: delivery,
		}))
	}
	words := func(segmentID uint64, delivery []string, status rtvi.SpokenStatus) *Event {
		return output(segmentID, frames.TextTypeSentence, "words", delivery, status)
	}

	s.translate(msg(t, rtvi.TypeBotLLMStarted, nil))
	// Spoken text counts as it is spoken, not when it is announced.
	wantNoEvent(t, words(1, []string{"excited"}, rtvi.SpokenNew))
	wantEvent(t, words(1, []string{"excited"}, rtvi.SpokenInProgress), EventExpression, "<<excited>>")
	wantNoEvent(t, words(1, []string{"excited"}, rtvi.SpokenCompleted))
	wantEvent(t, words(2, []string{"excited", "emphasis"}, rtvi.SpokenInProgress),
		EventExpression, "<<excited, emphasis>>")
	// Delivery can change within a segment, word by word.
	wantEvent(t, words(2, []string{"calm"}, rtvi.SpokenInProgress), EventExpression, "<<calm>>")
	// A sound counts when it is heard, once.
	wantNoEvent(t, output(3, frames.TextTypeExpressionSound, "laughter", nil, rtvi.SpokenNew))
	wantEvent(t, output(3, frames.TextTypeExpressionSound, "laughter", nil, rtvi.SpokenCompleted),
		EventExpression, "[[laughter]]")
	wantNoEvent(t, output(3, frames.TextTypeExpressionSound, "laughter", nil, rtvi.SpokenCompleted))
	wantEvent(t, output(4, frames.TextTypeExpressionPause, "600", nil, rtvi.SpokenCompleted),
		EventExpression, "[[pause 600]]")
	wantEvent(t, words(5, []string{"excited"}, rtvi.SpokenInProgress), EventExpression, "<<excited>>")
	wantNoEvent(t, words(6, []string{"excited"}, rtvi.SpokenInProgress))
	wantEvent(t, words(7, []string{}, rtvi.SpokenInProgress), EventExpression, "<<neutral>>")
	// Without expressive conversation, spoken text carries no delivery.
	wantNoEvent(t, words(8, nil, rtvi.SpokenInProgress))
	// Each response starts from the baseline.
	s.translate(msg(t, rtvi.TypeBotLLMStarted, nil))
	wantNoEvent(t, words(9, []string{}, rtvi.SpokenInProgress))
}

// A criterion on the expression the bot performed is a judge's to grade, and
// accumulates what the turn performed.
func TestAnExpressionCheckIsJudgedAndAggregates(t *testing.T) {
	if !judgeableEvents[EventExpression] {
		t.Fatal("expression is not judgeable")
	}
	s := &session{}
	if !s.aggregates(Expectation{Event: EventExpression, Eval: "the bot laughs"}) {
		t.Fatal("an expression check with a criterion does not aggregate")
	}
	if s.aggregates(Expectation{Event: EventExpression}) {
		t.Fatal("a bare expression expectation aggregates")
	}
}
