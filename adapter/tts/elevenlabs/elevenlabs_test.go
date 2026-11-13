package elevenlabs_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/adapter/tts/elevenlabs"
)

func TestCapabilities(t *testing.T) {
	c := elevenlabs.ElevenLabsAudioTagExpressionAdapter{}.Capabilities()

	if !c.AllowsFreeTextSounds || !c.AllowsFreeTextDelivery || !c.Emphasis {
		t.Fatalf("capabilities = %+v", c)
	}
	if c.PauseRangeMS == nil || *c.PauseRangeMS != [2]int{100, 3000} {
		t.Fatalf("pause range = %v, want [100 3000]", c.PauseRangeMS)
	}
}

func TestDeliveryNamesAreSingleWords(t *testing.T) {
	// A delivery tag separates its names by spaces.
	for _, term := range (elevenlabs.ElevenLabsAudioTagExpressionAdapter{}).Capabilities().Delivery {
		if len(strings.Fields(term.Name)) != 1 {
			t.Errorf("delivery %q is not a single word", term.Name)
		}
	}
}

func TestTagsFormattedWhereDeliveryChanges(t *testing.T) {
	a := elevenlabs.ElevenLabsAudioTagExpressionAdapter{}
	for _, tt := range []struct {
		delivery, previous []string
		want               string
	}{
		{[]string{"excited"}, nil, "[excited] Hi."},
		{[]string{"excited"}, []string{"excited"}, "Hi."},
		{[]string{"sad", "whispers"}, nil, "[sad] [whispers] Hi."},
	} {
		if got := a.FormatText("Hi.", tt.delivery, tt.previous); got != tt.want {
			t.Errorf("FormatText(%q after %q) = %q, want %q", tt.delivery, tt.previous, got, tt.want)
		}
	}
}

func TestSlowAndFastAreWrittenAsPaceTags(t *testing.T) {
	a := elevenlabs.ElevenLabsAudioTagExpressionAdapter{}
	var delivery []string
	for _, term := range a.Capabilities().Delivery {
		delivery = append(delivery, term.Name)
	}
	if !slices.Contains(delivery, "slow") || !slices.Contains(delivery, "fast") {
		t.Fatalf("delivery = %q", delivery)
	}
	if got := a.FormatText("Hi.", []string{"calm", "slow"}, nil); got != "[calm] [slowly] Hi." {
		t.Errorf("calm slow = %q", got)
	}
	if got := a.FormatText("Hi.", []string{"fast"}, []string{"calm"}); got != "[quickly] Hi." {
		t.Errorf("fast after calm = %q", got)
	}
}

func TestReturnToBaselineFormatsNothing(t *testing.T) {
	if got := (elevenlabs.ElevenLabsAudioTagExpressionAdapter{}).FormatText(
		"Hi.", []string{}, []string{"excited"}); got != "Hi." {
		t.Errorf("baseline after excited = %q", got)
	}
}

func TestEmphasisIsCapitalized(t *testing.T) {
	a := elevenlabs.ElevenLabsAudioTagExpressionAdapter{}
	if got := a.FormatText("back up", []string{"emphasis"}, nil); got != "BACK UP" {
		t.Errorf("emphasis = %q", got)
	}
	if got := a.FormatText("back up", []string{"excited", "emphasis"}, []string{"excited"}); got != "BACK UP" {
		t.Errorf("excited emphasis after excited = %q", got)
	}
}

func TestSounds(t *testing.T) {
	if got := (elevenlabs.ElevenLabsAudioTagExpressionAdapter{}).FormatSound("laughs"); got != " [laughs] " {
		t.Errorf("sound = %q", got)
	}
}

func TestPausesAreShortOrLong(t *testing.T) {
	a := elevenlabs.ElevenLabsAudioTagExpressionAdapter{}
	if got := a.FormatPause(500); got != " [short pause] " {
		t.Errorf("500 ms = %q", got)
	}
	if got := a.FormatPause(1500); got != " [long pause] " {
		t.Errorf("1500 ms = %q", got)
	}
}
