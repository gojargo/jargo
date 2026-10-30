package tts_test

// Tests for the universal expression names, the names the TTS services perform
// them by, and each service's adapter.

import (
	"slices"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/adapter/tts/azure"
	"github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/adapter/tts/elevenlabs"
	"github.com/gojargo/jargo/adapter/tts/fish"
	"github.com/gojargo/jargo/adapter/tts/gemini"
	"github.com/gojargo/jargo/adapter/tts/inworld"
	"github.com/gojargo/jargo/adapter/tts/minimax"
	"github.com/gojargo/jargo/adapter/tts/mistral"
	"github.com/gojargo/jargo/adapter/tts/soniox"
	"github.com/gojargo/jargo/expressive"
)

func adapters() map[string]ttsadapter.ExpressionAdapter {
	return map[string]ttsadapter.ExpressionAdapter{
		"azure":         azure.AzureHDExpressionAdapter{},
		"cartesia":      cartesia.CartesiaExpressionAdapter{},
		"elevenlabs":    elevenlabs.ElevenLabsAudioTagExpressionAdapter{},
		"fish":          fish.FishAudioS2ExpressionAdapter{},
		"gemini":        gemini.Gemini38ExpressionAdapter{},
		"inworld":       inworld.InworldExpressionAdapter{},
		"inworld flash": inworld.InworldFlashExpressionAdapter{},
		"minimax":       minimax.MiniMaxExpressionAdapter{},
		"mistral":       mistral.MistralExpressionAdapter{},
		"soniox":        soniox.SonioxExpressionAdapter{},
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
