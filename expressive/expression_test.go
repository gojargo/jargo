package expressive_test

// Tests for the universal expression names, the names the TTS services perform
// them by, and each service's adapter.

import (
	"slices"
	"testing"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/provider/azure/speech"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/fish"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/inworld"
	"github.com/gojargo/jargo/provider/minimax"
	"github.com/gojargo/jargo/provider/mistral"
	"github.com/gojargo/jargo/provider/soniox"
)

func adapters() map[string]expressive.ExpressionAdapter {
	return map[string]expressive.ExpressionAdapter{
		"azure":         speech.HDExpressionAdapter{},
		"cartesia":      cartesia.ExpressionAdapter{},
		"elevenlabs":    elevenlabs.AudioTagExpressionAdapter{},
		"fish":          fish.S2ExpressionAdapter{},
		"gemini":        gemini.TTSExpressionAdapter{},
		"inworld":       inworld.ExpressionAdapter{},
		"inworld flash": inworld.FlashExpressionAdapter{},
		"minimax":       minimax.ExpressionAdapter{},
		"mistral":       mistral.ExpressionAdapter{},
		"soniox":        soniox.ExpressionAdapter{},
	}
}

// notAVoice are sounds a service declares that are not the voice's own.
func notAVoice() []string {
	return []string{
		"applause", "clapping", "gunshot", "explosion", "fart",
		"audience laughing", "background laughter", "crowd laughing",
	}
}

func TestANameIsEitherADeliveryOrASound(t *testing.T) {
	for _, d := range expressive.Deliveries() {
		if expressive.IsSound(string(d)) {
			t.Errorf("%q is both a delivery and a sound", d)
		}
	}
}

func TestTheCoreNamesAreUniversal(t *testing.T) {
	for _, d := range expressive.CoreDelivery() {
		if !expressive.IsDelivery(string(d)) {
			t.Errorf("core delivery %q is not universal", d)
		}
	}
	for _, s := range expressive.CoreSounds() {
		if !expressive.IsSound(string(s)) {
			t.Errorf("core sound %q is not universal", s)
		}
	}
}

func termNames(terms []expressive.ExpressionTerm) []string {
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		out = append(out, t.Name)
	}
	return out
}

func TestAServiceNamesUniversalNamesByNamesItPerforms(t *testing.T) {
	for name, a := range adapters() {
		c := a.Capabilities()
		for universal, names := range c.ServiceVocabulary {
			var (
				performed string
				freeText  bool
				declared  []string
			)
			switch u := universal.(type) {
			case expressive.ExpressionDelivery:
				performed, _ = c.DeliveryName(string(u))
				freeText, declared = c.AllowsFreeTextDelivery, termNames(c.Delivery)
			case expressive.ExpressionSound:
				performed, _ = c.SoundName(string(u))
				freeText, declared = c.AllowsFreeTextSounds, termNames(c.Sounds)
			}
			if !slices.Contains(names, performed) {
				t.Errorf("%s: %q is performed as %q, not one of %q", name, universal, performed, names)
			}
			if !freeText && !slices.ContainsFunc(names, func(n string) bool { return slices.Contains(declared, n) }) {
				t.Errorf("%s: none of %q, the names of %q, is declared", name, names, universal)
			}
		}
	}
}

func TestEveryVoiceAServiceDeclaresIsReachedByAUniversalName(t *testing.T) {
	for name, a := range adapters() {
		c := a.Capabilities()
		var reached []string
		for _, names := range c.ServiceVocabulary {
			reached = append(reached, names...)
		}
		for _, term := range append(append([]expressive.ExpressionTerm{}, c.Delivery...), c.Sounds...) {
			n := term.Name
			if expressive.IsDelivery(n) || expressive.IsSound(n) || slices.Contains(reached, n) ||
				slices.Contains(notAVoice(), n) {
				continue
			}
			t.Errorf("%s: %q is reached by no universal name", name, n)
		}
	}
}
