package spoken_test

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/expressive/formatter"
	"github.com/gojargo/jargo/expressive/spoken"
	"github.com/gojargo/jargo/frames"
)

//nolint:gochecknoglobals // shared test fixture
var laugh = formatter.ReportedSound{TextType: frames.TextTypeExpressionSound, Text: "laugh"}

const text = "<<calm>> Hi. [[laugh]] Bye."

func spokenExpression() *spoken.SpokenExpression {
	s := spoken.New()
	s.Add(text, []formatter.ExpressionSpan{
		{Start: 8, End: 13, Delivery: []string{"calm"}},
		{Start: 13, End: 22, Sound: &laugh},
		{Start: 22, End: 27, Delivery: []string{"calm", "emphasis"}},
	})
	return s
}

func checkSpoken(
	t *testing.T, s *spoken.SpokenExpression, said string, delivery []string, sounds []formatter.ReportedSound,
) {
	t.Helper()
	gotDelivery, ok, gotSounds := s.Spoken(said)
	if delivery == nil {
		if ok {
			t.Fatalf("Spoken(%q) delivery = %v, want none", said, gotDelivery)
		}
	} else if !ok || !reflect.DeepEqual(gotDelivery, delivery) {
		t.Fatalf("Spoken(%q) delivery = %v (%t), want %v", said, gotDelivery, ok, delivery)
	}
	if len(gotSounds) != len(sounds) || (len(sounds) > 0 && !reflect.DeepEqual(gotSounds, sounds)) {
		t.Fatalf("Spoken(%q) sounds = %v, want %v", said, gotSounds, sounds)
	}
}

func TestAWordIsGivenItsDeliveryAndTheSoundsBeforeIt(t *testing.T) {
	s := spokenExpression()

	checkSpoken(t, s, "Hi.", []string{"calm"}, nil)
	checkSpoken(t, s, "[[laugh]] Bye.", []string{"calm", "emphasis"}, []formatter.ReportedSound{laugh})
	if d, _ := s.Delivery(); !reflect.DeepEqual(d, []string{"calm", "emphasis"}) {
		t.Fatalf("Delivery = %v", d)
	}
}

func TestUnitsAddedFollowOneAnother(t *testing.T) {
	s := spokenExpression()
	s.Add(" Again.", []formatter.ExpressionSpan{{Start: 0, End: 7, Delivery: []string{}}})

	s.Spoken("Bye.")

	checkSpoken(t, s, "Again.", []string{}, nil)
}

func TestSpokenMarkupStandsForTheNextSound(t *testing.T) {
	s := spokenExpression()
	s.Spoken("Hi.")

	if got, ok := s.SpokenMarkup(); !ok || got != laugh {
		t.Fatalf("SpokenMarkup = %v, %t", got, ok)
	}
	// Reported once, by its markup.
	checkSpoken(t, s, "[[laugh]] Bye.", []string{"calm", "emphasis"}, nil)
}

func TestSpokenMarkupWithNoSoundBeforeTheNextWordsIsntSpoken(t *testing.T) {
	if got, ok := spokenExpression().SpokenMarkup(); ok {
		t.Fatalf("SpokenMarkup = %v, want none", got)
	}
}

func TestASoundAfterTheLastWordIsLeftForWhatFollows(t *testing.T) {
	s := spoken.New()
	// "Bye [[laugh]]!", spoken as one word that covers the tag.
	s.Add("Bye [[laugh]]!", []formatter.ExpressionSpan{
		{Start: 0, End: 4, Delivery: []string{}},
		{Start: 4, End: 13, Sound: &laugh},
	})

	checkSpoken(t, s, "Bye [[laugh]]!", []string{}, nil)
	if got, ok := s.SpokenMarkup(); !ok || got != laugh {
		t.Fatalf("SpokenMarkup = %v, %t", got, ok)
	}
}

func TestSoundsNoTextReportedAreUnreportedAtTheEnd(t *testing.T) {
	s := spoken.New()
	s.Add("Bye [[laugh]]", []formatter.ExpressionSpan{
		{Start: 0, End: 4, Delivery: []string{}},
		{Start: 4, End: 13, Sound: &laugh},
	})

	s.Spoken("Bye")

	if got := s.Unreported(); !reflect.DeepEqual(got, []formatter.ReportedSound{laugh}) {
		t.Fatalf("Unreported = %v", got)
	}
	if got := s.Unreported(); len(got) != 0 {
		t.Fatalf("Unreported again = %v", got)
	}
}

func TestTextSpokenBeforeAnyWasAddedHasNoDelivery(t *testing.T) {
	s := spoken.New()

	checkSpoken(t, s, "Hi.", nil, nil)
	if _, ok := s.Delivery(); ok {
		t.Fatal("Delivery reported with no text added")
	}
}
