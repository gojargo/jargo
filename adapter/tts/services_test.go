package tts_test

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/adapter/tts/azure"
	"github.com/gojargo/jargo/adapter/tts/gemini"
	"github.com/gojargo/jargo/adapter/tts/inworld"
	"github.com/gojargo/jargo/adapter/tts/mistral"
	"github.com/gojargo/jargo/adapter/tts/soniox"
	"github.com/gojargo/jargo/expressive"
)

func want(t *testing.T, got, expected string) {
	t.Helper()
	if got != expected {
		t.Errorf("got %q, want %q", got, expected)
	}
}

func named(t *testing.T, name func(string) (string, bool), universal, expected string) {
	t.Helper()
	got, ok := name(universal)
	if expected == "" {
		if ok {
			t.Errorf("%q performed as %q, want none", universal, got)
		}
		return
	}
	if !ok || got != expected {
		t.Errorf("%q performed as %q, want %q", universal, got, expected)
	}
}

func TestInworldPerformsEveryUniversalDelivery(t *testing.T) {
	c := inworld.InworldExpressionAdapter{}.Capabilities()
	if !c.AllowsFreeTextDelivery || c.AllowsFreeTextSounds || !c.Emphasis {
		t.Fatalf("capabilities = %+v", c)
	}
	named(t, c.DeliveryName, "warm", "warm")
	named(t, c.SoundName, "breath", "breathe")
	// Delivery is free text, so a name that isn't universal is performed as
	// written, while sounds are a fixed list.
	named(t, c.DeliveryName, "british accent", "british accent")
	named(t, c.SoundName, "beatbox", "")
}

func TestInworldInstructionsWhereDeliveryChanges(t *testing.T) {
	a := inworld.InworldExpressionAdapter{}
	want(t, a.FormatText("Hi.", []string{"calm"}, nil), "[calm] Hi.")
	want(t, a.FormatText("Hi.", []string{"calm"}, []string{"calm"}), "Hi.")
	want(t, a.FormatText("A B 7", []string{"calm", "slow"}, []string{"calm"}), "[calm and slow] A B 7")
}

func TestInworldAReturnToTheBaselineResets(t *testing.T) {
	a := inworld.InworldExpressionAdapter{}
	// An instruction lasts past its sentence, so one spoken in a delivery ends
	// by setting it back...
	want(t, a.FormatText("Bye.", []string{}, []string{"calm"}), "[reset] Bye.")
	want(t, a.FormatEnd([]string{"calm"}), " [reset]")
	// ...and a sentence in the baseline has nothing to set back.
	want(t, a.FormatText("Bye.", []string{}, nil), "Bye.")
	want(t, a.FormatEnd([]string{}), "")
	want(t, a.FormatEnd([]string{"emphasis"}), "")
}

func TestInworldEmphasisIsWrittenInCapitals(t *testing.T) {
	want(t, inworld.InworldExpressionAdapter{}.FormatText("now", []string{"calm", "emphasis"}, []string{"calm"}), "NOW")
}

func TestInworldSoundsAndPauses(t *testing.T) {
	a := inworld.InworldExpressionAdapter{}
	want(t, a.FormatSound("laugh"), " [laugh] ")
	want(t, a.FormatPause(600), ` <break time="600ms"/> `)
}

func TestInworldFlashPerformsSoundsAndPausesWithoutDelivery(t *testing.T) {
	c := inworld.InworldFlashExpressionAdapter{}.Capabilities()
	if len(c.Delivery) != 0 || c.Emphasis || !reflect.DeepEqual(c.PauseRangeMS, &[2]int{100, 10000}) {
		t.Fatalf("capabilities = %+v", c)
	}
	named(t, c.SoundName, "laugh", "laugh")
}

func TestSonioxCapabilities(t *testing.T) {
	c := soniox.SonioxExpressionAdapter{}.Capabilities()
	if c.AllowsFreeTextSounds || c.AllowsFreeTextDelivery || !c.Emphasis {
		t.Fatalf("capabilities = %+v", c)
	}
	named(t, c.DeliveryName, "sympathetic", "warm")
	named(t, c.DeliveryName, "soft", "softly")
	named(t, c.SoundName, "laugh", "laughs")
}

func TestSonioxTagsWhereDeliveryChanges(t *testing.T) {
	a := soniox.SonioxExpressionAdapter{}
	want(t, a.FormatText("Hi.", []string{"excited"}, nil), "[excited] Hi.")
	want(t, a.FormatText("Hi.", []string{"excited"}, []string{"excited"}), "Hi.")
	want(t, a.FormatText("A B 7", []string{"calm", "slow"}, []string{"calm"}), "[calm] [slowly] A B 7")
	want(t, a.FormatText("Hi.", []string{"fast"}, nil), "[quickly] Hi.")
}

func TestSonioxAReturnToTheBaselineWritesNeutral(t *testing.T) {
	a := soniox.SonioxExpressionAdapter{}
	// A tag lasts past its sentence, so one spoken in a delivery ends by setting
	// it back...
	want(t, a.FormatText("Bye.", []string{}, []string{"calm"}), "[neutral] Bye.")
	want(t, a.FormatEnd([]string{"calm"}), " [neutral]")
	// ...and a sentence in the baseline has nothing to set back.
	want(t, a.FormatText("Bye.", []string{}, nil), "Bye.")
	want(t, a.FormatEnd([]string{}), "")
}

func TestSonioxEmphasis(t *testing.T) {
	want(t, soniox.SonioxExpressionAdapter{}.FormatText("now", []string{"calm", "emphasis"}, []string{"calm"}), "*now*")
}

func TestSonioxSoundsAndPauses(t *testing.T) {
	a := soniox.SonioxExpressionAdapter{}
	want(t, a.FormatSound("clears throat"), " [clears throat] ")
	want(t, a.FormatPause(600), " [pause] ")
	want(t, a.FormatPause(1200), " [long pause] ")
}

func TestMistralCapabilities(t *testing.T) {
	c := mistral.MistralExpressionAdapter{}.Capabilities()
	if c.AllowsFreeTextSounds || c.AllowsFreeTextDelivery || c.Emphasis || c.PauseRangeMS != nil {
		t.Fatalf("capabilities = %+v", c)
	}
	named(t, c.DeliveryName, "sympathetic", "warm")
	named(t, c.SoundName, "clear throat", "throat clearing")
}

func TestMistralTagsWhereEmotionOrPaceChanges(t *testing.T) {
	a := mistral.MistralExpressionAdapter{}
	want(t, a.FormatText("Hi.", []string{"happy"}, nil), "[emotion=happy] Hi.")
	want(t, a.FormatText("Hi.", []string{"happy"}, []string{"happy"}), "Hi.")
	want(t, a.FormatText("A B 7", []string{"calm", "slow"}, []string{"calm"}), "[pace=slow] A B 7")
	want(t, a.FormatText("Hi.", []string{"calm", "fast"}, nil), "[emotion=calm][pace=fast] Hi.")
	// A return to the normal pace writes moderate.
	want(t, a.FormatText(", thanks.", []string{"calm"}, []string{"calm", "slow"}), "[pace=moderate] , thanks.")
	// A return to the baseline writes no emotion.
	want(t, a.FormatText("Bye.", []string{}, []string{"happy"}), "Bye.")
	// One emotion at a time.
	want(t, a.FormatText("Hi.", []string{"happy", "warm"}, nil), "[emotion=happy] Hi.")
	want(t, a.FormatSound("laugh hard"), "[laugh hard]")
}

func TestGeminiPerformsEveryUniversalDeliveryBetweenSentences(t *testing.T) {
	c := gemini.Gemini38ExpressionAdapter{}.Capabilities()
	if !c.AllowsFreeTextDelivery || c.AllowsFreeTextSounds || !c.Emphasis ||
		c.DeliveryScope != expressive.DeliveryScopeSentence {
		t.Fatalf("capabilities = %+v", c)
	}
	named(t, c.SoundName, "clear throat", "throat-clearing")
	// Delivery is free text, so a name that isn't universal is performed as
	// written, while sounds are a fixed list.
	named(t, c.DeliveryName, "british accent", "british accent")
	named(t, c.SoundName, "beatbox", "")
}

func TestGeminiDeliveryIsARequestStyle(t *testing.T) {
	a := gemini.Gemini38ExpressionAdapter{}
	got := a.RequestSettings([]string{"calm", "slow"})
	if !reflect.DeepEqual(got, map[string]any{"style": "calm, speaking slowly"}) {
		t.Errorf("settings = %v", got)
	}
	if got := a.RequestSettings([]string{"emphasis"}); len(got) != 0 {
		t.Errorf("settings = %v", got)
	}
	if got := a.RequestSettings([]string{}); len(got) != 0 {
		t.Errorf("settings = %v", got)
	}
	want(t, a.FormatText("Hi.", []string{"calm"}, nil), "Hi.")
	// Emphasis is written in capitals.
	want(t, a.FormatText("now", []string{"emphasis"}, nil), "NOW")
	want(t, a.FormatSound("laugh"), " <laugh> ")
	want(t, a.FormatPause(600), " <short pause> ")
	want(t, a.FormatPause(1500), " <long pause> ")
}

func TestAzureCapabilities(t *testing.T) {
	c := azure.AzureHDExpressionAdapter{}.Capabilities()
	if c.AllowsFreeTextSounds || c.AllowsFreeTextDelivery || c.Emphasis || c.PauseRangeMS != nil {
		t.Fatalf("capabilities = %+v", c)
	}
	named(t, c.DeliveryName, "apologetic", "remorseful")
	named(t, c.SoundName, "laugh", "laughter")
}

func TestAzureMarkersWhereTheStyleChanges(t *testing.T) {
	a := azure.AzureHDExpressionAdapter{}
	want(t, a.FormatText("Hi.", []string{"calm"}, nil), "[calm] Hi.")
	want(t, a.FormatText("Hi.", []string{"calm"}, []string{"calm"}), "Hi.")
	// One style at a time, the one added last.
	want(t, a.FormatText("A B 7", []string{"calm", "slow"}, []string{"calm"}), "[slow] A B 7")
	want(t, a.FormatText(", okay.", []string{"calm"}, []string{"calm", "slow"}), "[calm] , okay.")
	// A return to the baseline writes neutral.
	want(t, a.FormatText("Bye.", []string{}, []string{"calm"}), "[Neutral] Bye.")
	want(t, a.FormatSound("sighing"), " [sighing] ")
}
