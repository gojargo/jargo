package g711

import (
	"encoding/binary"
	"testing"
)

// TestALawDomainIdempotent mirrors the μ-law idempotency check: decoding an
// A-law byte yields a canonical reconstruction level, so re-encoding it must
// return the same byte.
func TestALawDomainIdempotent(t *testing.T) {
	for a := range 256 {
		pcm := DecodeALaw([]byte{byte(a)})
		re := EncodeALaw(pcm)[0]
		if re != byte(a) {
			sample := int16(binary.LittleEndian.Uint16(pcm))
			t.Fatalf("re-encode A-law 0x%02X: decoded %d re-encoded 0x%02X", a, sample, re)
		}
	}
}

// TestALawLinearRoundTripRelative sweeps the 16-bit range and checks the
// round-trip error stays within A-law's companded resolution.
func TestALawLinearRoundTripRelative(t *testing.T) {
	for s := -32768; s <= 32767; s += 7 {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(int16(s)))
		out := int16(binary.LittleEndian.Uint16(DecodeALaw(EncodeALaw(b[:]))))
		bound := abs(s)/8 + 256
		if d := abs(s - int(out)); d > bound {
			t.Fatalf("sample %d round-tripped to %d (diff %d > %d)", s, out, d, bound)
		}
	}
}

// TestALawEncodeLengths verifies the byte ratios and that an odd trailing byte
// is ignored rather than panicking.
func TestALawEncodeLengths(t *testing.T) {
	if got := len(EncodeALaw(make([]byte, 320))); got != 160 {
		t.Fatalf("encode 320 PCM bytes: got %d A-law bytes want 160", got)
	}
	if got := len(DecodeALaw(make([]byte, 160))); got != 320 {
		t.Fatalf("decode 160 A-law bytes: got %d PCM bytes want 320", got)
	}
	if got := len(EncodeALaw(make([]byte, 5))); got != 2 {
		t.Fatalf("odd trailing byte: got %d want 2", got)
	}
}

// TestALawSilenceRoundTrip checks that digital silence stays quiet through the
// A-law round trip (A-law has no exact zero, so a small residual is expected).
func TestALawSilenceRoundTrip(t *testing.T) {
	decoded := DecodeALaw(EncodeALaw(make([]byte, 160)))
	for i := 0; i < len(decoded); i += 2 {
		if v := int16(binary.LittleEndian.Uint16(decoded[i:])); v < -8 || v > 8 {
			t.Fatalf("silence sample %d decoded to %d", i/2, v)
		}
	}
}
