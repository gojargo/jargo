package g711

import (
	"bytes"
	"testing"
)

// FuzzULawCodec exercises the μ-law codec on arbitrary bytes. It asserts the
// codec never panics, the one-byte-per-sample length contracts hold each way,
// and that the μ-law stream is stable under a further decode/encode round trip.
// Encode never emits negative zero (0x7F), so re-encoding a decoded stream must
// reproduce it exactly — a strong, always-true invariant that catches codec bugs
// on malformed or adversarial input (e.g. a hostile telephony peer).
func FuzzULawCodec(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x00})
	f.Add([]byte{0xFF, 0x7F, 0x80, 0x01, 0x00})
	f.Add(bytes.Repeat([]byte{0xAB, 0xCD}, 160)) // a 20 ms 8 kHz telephony frame

	f.Fuzz(func(t *testing.T, data []byte) {
		// Decoding yields two PCM bytes per μ-law byte and never panics.
		if pcm := DecodeULaw(data); len(pcm) != 2*len(data) {
			t.Fatalf("DecodeULaw length = %d, want %d", len(pcm), 2*len(data))
		}

		// Encoding treats data as 16-bit PCM: one μ-law byte per sample, a
		// trailing odd byte dropped; never panics.
		u := EncodeULaw(data)
		if len(u) != len(data)/2 {
			t.Fatalf("EncodeULaw length = %d, want %d", len(u), len(data)/2)
		}

		// Stability: decoding then re-encoding the μ-law stream reproduces it.
		if got := EncodeULaw(DecodeULaw(u)); !bytes.Equal(got, u) {
			t.Fatalf("μ-law not stable under round trip\n in:  %x\n out: %x", u, got)
		}
	})
}
