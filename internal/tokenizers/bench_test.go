package tokenizers

import (
	"strings"
	"testing"
)

// BenchmarkTaggerPair encodes a typical pair of the tagger model, truncated
// to 160 tokens (the first sequence is long enough to be cut).
func BenchmarkTaggerPair(b *testing.B) {
	tok := load(b, "tagger")
	first := "caller (tone frustrated clear; sarcasm low): " +
		strings.Repeat("I have been waiting for my order for two weeks now and nobody calls me back. ", 4)
	second := "tts elevenlabs | can calm sympathetic apologetic warm excited surprised sad grateful curious" +
		" confident slow fast | sounds laugh chuckle sigh | stress yes | policy match style, slow for details" +
		" | caller sounded tone frustrated clear; sarcasm low | last reply none | state apologized before" +
		" | reply so far: I'm so sorry about the wait. Let me look into it right now." +
		" | sentence: Your order shipped yesterday."
	trunc := &Truncation{MaxLength: 160, Strategy: OnlyFirst, Direction: Left}
	enc, err := tok.EncodePair(first, second, trunc)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := tok.EncodePair(first, second, trunc); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(enc.IDs)), "tokens")
}

// BenchmarkReader encodes a typical sequence of the reader model.
func BenchmarkReader(b *testing.B) {
	tok := load(b, "reader")
	text := `"Oh great, another delay. Just what I needed today, thanks a lot." ` +
		`Before: "Your order is on its way and should arrive by Friday."`
	trunc := tok.Truncation()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := tok.Encode(text, trunc); err != nil {
			b.Fatal(err)
		}
	}
}
