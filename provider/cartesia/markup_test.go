package cartesia

import "testing"

// TestMarkupTags pins the markup each helper writes, since the tags reach
// Cartesia verbatim and the timing tokens it reports back are matched against
// text that carries them.
func TestMarkupTags(t *testing.T) {
	cases := []struct{ got, want string }{
		{Spell("A.B.C."), "<spell>A.B.C.</spell>"},
		{EmotionTag(EmotionExcited), `<emotion value="excited" />`},
		{EmotionTag(EmotionJokingComedic), `<emotion value="joking/comedic" />`},
		{PauseTag(0.5), `<break time="0.5s" />`},
		// A whole number keeps its decimal point, the way the markup spells one.
		{PauseTag(1), `<break time="1.0s" />`},
		{VolumeTag(1.5), `<volume ratio="1.5" />`},
		{VolumeTag(2), `<volume ratio="2.0" />`},
		{SpeedTag(0.6), `<speed ratio="0.6" />`},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("tag = %q, want %q", c.got, c.want)
		}
	}
}

// TestMarkupIsStrippedFromTheTimings checks a tag the helpers write comes back
// off the tokens Cartesia reports, so they still match the text that was sent.
func TestMarkupIsStrippedFromTheTimings(t *testing.T) {
	token := "to" + Spell("1234") + "."
	if got := stripCartesiaTags(token); got != "to 1234." {
		t.Errorf("stripCartesiaTags(%q) = %q, want %q", token, got, "to 1234.")
	}
	if got := stripCartesiaTags(EmotionTag(EmotionSad)); got != "" {
		t.Errorf("a token of nothing but markup came back as %q, want it dropped", got)
	}
}
