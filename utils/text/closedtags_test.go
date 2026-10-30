package text_test

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

const spelledReply = "Your code is <spell>X. Y. Z</spell>. Say <spell>St. Louis</spell> slowly. Done"

// Punctuation inside a closed tag is not a sentence boundary.
func TestSkipTagsClosedTagContentIsNotSplit(t *testing.T) {
	a := newSkipTags(t, frames.TextTypeSentence, spellTag())

	got := texts(a.Aggregate(spelledReply))

	want := []string{"Your code is <spell>X. Y. Z</spell>.", "Say <spell>St. Louis</spell> slowly."}
	if !equalStrings(got, want) {
		t.Fatalf("sentences = %q, want %q", got, want)
	}
	if rest, _ := a.Flush(); rest.Text != "Done" {
		t.Fatalf("flushed %q", rest.Text)
	}
}

// A tag is still recognized after an earlier sentence has been emitted.
func TestSkipTagsTagAfterAnEmittedSentenceIsNotSplit(t *testing.T) {
	a := newSkipTags(t, frames.TextTypeSentence, spellTag())

	got := texts(a.Aggregate("Hi <spell>AB</spell>. My name is <spell>Bob. Smith</spell> today. Done"))

	want := []string{"Hi <spell>AB</spell>.", "My name is <spell>Bob. Smith</spell> today."}
	if !equalStrings(got, want) {
		t.Fatalf("sentences = %q, want %q", got, want)
	}
}

// Punctuation inside a closed kept pattern is not a sentence boundary.
func TestPatternPairKeptPatternContentIsNotSplit(t *testing.T) {
	a := text.NewPatternPairAggregator(frames.TextTypeSentence, "")
	if err := a.AddPattern("spell_pattern", "<spell>", "</spell>", text.MatchKeep); err != nil {
		t.Fatal(err)
	}

	got := texts(a.Aggregate(spelledReply))

	want := []string{"Your code is <spell>X. Y. Z</spell>.", "Say <spell>St. Louis</spell> slowly."}
	if !equalStrings(got, want) {
		t.Fatalf("sentences = %q, want %q", got, want)
	}
	if rest, _ := a.Flush(); rest.Text != "Done" {
		t.Fatalf("flushed %q", rest.Text)
	}
}
