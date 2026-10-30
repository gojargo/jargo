package fish_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter/tts/fish"
	"github.com/gojargo/jargo/expressive"
)

func names(terms []expressive.ExpressionTerm) []string {
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		out = append(out, t.Name)
	}
	return out
}

func TestCapabilities(t *testing.T) {
	c := fish.FishAudioS2ExpressionAdapter{}.Capabilities()

	if !c.AllowsFreeTextDelivery || !c.AllowsFreeTextSounds || !c.Emphasis {
		t.Fatalf("capabilities = %+v", c)
	}
	if !slices.Contains(names(c.Sounds), "laughing") {
		t.Fatalf("sounds = %q", names(c.Sounds))
	}
	// A delivery tag separates its names by spaces.
	for _, term := range c.Delivery {
		if len(strings.Fields(term.Name)) != 1 {
			t.Fatalf("delivery %q is several words", term.Name)
		}
	}
}

func check(t *testing.T, got, want string) {
	t.Helper()
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestSlowAndFastAreWrittenAsSpeakingTags(t *testing.T) {
	a := fish.FishAudioS2ExpressionAdapter{}
	delivery := names(a.Capabilities().Delivery)
	if !slices.Contains(delivery, "slow") || !slices.Contains(delivery, "fast") {
		t.Fatalf("delivery = %q", delivery)
	}
	check(t, a.FormatText("Hi.", []string{"calm", "slow"}, nil), "[calm][speaking slowly] Hi.")
	check(t, a.FormatText("Hi.", []string{"fast"}, nil), "[speaking quickly] Hi.")
}

func TestAPaceIsSetBackWhereASpeedDeliveryEnds(t *testing.T) {
	a := fish.FishAudioS2ExpressionAdapter{}
	// Within a request, the words after the slow ones set the pace back...
	check(t, a.FormatText(", thanks.", []string{"calm"}, []string{"calm", "slow"}), "[normal pace][calm] , thanks.")
	// ...and a request that ends slow sets it back at its end.
	check(t, a.FormatEnd([]string{"calm", "slow"}), " [normal pace]")
	check(t, a.FormatEnd([]string{"calm"}), "")
}

func TestTagsWhereDeliveryChanges(t *testing.T) {
	a := fish.FishAudioS2ExpressionAdapter{}
	check(t, a.FormatText("Hi.", []string{"excited"}, nil), "[excited] Hi.")
	check(t, a.FormatText("Hi.", []string{"excited"}, []string{"excited"}), "Hi.")
	check(t, a.FormatText("Hi.", []string{"sad", "whispering"}, nil), "[sad][whispering] Hi.")
}

func TestReturnToBaselineWritesNoTag(t *testing.T) {
	// Fish reads a [neutral] tag as a delivery of its own.
	check(t, fish.FishAudioS2ExpressionAdapter{}.FormatText("Hi.", []string{}, []string{"excited"}), "Hi.")
}

func TestEmphasis(t *testing.T) {
	a := fish.FishAudioS2ExpressionAdapter{}
	check(t, a.FormatText("now", []string{"emphasis"}, nil), "[emphasis] now")
	check(t, a.FormatText("now", []string{"excited", "emphasis"}, []string{"excited"}), "[emphasis] now")
}

func TestSoundsAndPauses(t *testing.T) {
	a := fish.FishAudioS2ExpressionAdapter{}
	check(t, a.FormatSound("clear throat"), "[clear throat]")
	check(t, a.FormatPause(600), "[break]")
	check(t, a.FormatPause(1200), "[long-break]")
}
