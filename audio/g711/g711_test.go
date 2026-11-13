package g711

import (
	"encoding/binary"
	"errors"
	"testing"
)

// TestULawDomainIdempotent is the rigorous codec check: decoding a μ-law byte
// yields a canonical reconstruction level, so re-encoding it must return the
// same byte. Both zero codes (0x7F, 0xFF) decode to 0, and 0 re-encodes to 0xFF,
// which is the only allowed disagreement.
func TestULawDomainIdempotent(t *testing.T) {
	for u := range 256 {
		pcm := DecodeULaw([]byte{byte(u)})
		re := encode(t, EncodeULaw, pcm)[0]
		if re == byte(u) {
			continue
		}
		sample := int16(binary.LittleEndian.Uint16(pcm))
		if sample == 0 && re == 0xFF { // -0 (0x7F) re-encodes to +0 (0xFF)
			continue
		}
		t.Fatalf("re-encode μ-law 0x%02X: decoded %d re-encoded 0x%02X", u, sample, re)
	}
}

// TestLinearRoundTripRelative sweeps the 16-bit range and checks the round-trip
// error stays within μ-law's companded resolution (error grows with magnitude),
// catching gross encode/decode bugs without asserting bit-exactness.
func TestLinearRoundTripRelative(t *testing.T) {
	for s := -32768; s <= 32767; s += 7 {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(int16(s)))
		out := int16(binary.LittleEndian.Uint16(DecodeULaw(encode(t, EncodeULaw, b[:]))))
		bound := abs(s)/8 + 256
		if d := abs(s - int(out)); d > bound {
			t.Fatalf("sample %d round-tripped to %d (diff %d > %d)", s, out, d, bound)
		}
	}
}

// TestEncodeLengths verifies the byte ratios.
func TestEncodeLengths(t *testing.T) {
	if got := len(encode(t, EncodeULaw, make([]byte, 320))); got != 160 {
		t.Fatalf("encode 320 PCM bytes: got %d μ-law bytes want 160", got)
	}
	if got := len(DecodeULaw(make([]byte, 160))); got != 320 {
		t.Fatalf("decode 160 μ-law bytes: got %d PCM bytes want 320", got)
	}
}

// PCM that ends partway through a sample is rejected by both encoders.
func TestEncodingPCMThatIsNotWholeSamplesFails(t *testing.T) {
	for name, enc := range map[string]func([]byte) ([]byte, error){
		"EncodeULaw": EncodeULaw,
		"EncodeALaw": EncodeALaw,
	} {
		if out, err := enc([]byte{0x00, 0x00, 0x00}); !errors.Is(err, ErrPartialSample) {
			t.Errorf("%s(3 bytes) = % x, %v; want ErrPartialSample", name, out, err)
		}
	}
}

// encode runs enc over pcm, failing the test if it reports an error.
func encode(tb testing.TB, enc func([]byte) ([]byte, error), pcm []byte) []byte {
	tb.Helper()
	out, err := enc(pcm)
	if err != nil {
		tb.Fatalf("encode %d bytes: %v", len(pcm), err)
	}
	return out
}

// TestSilenceRoundTrip checks that digital silence stays silent.
func TestSilenceRoundTrip(t *testing.T) {
	decoded := DecodeULaw(encode(t, EncodeULaw, make([]byte, 160)))
	for i := 0; i < len(decoded); i += 2 {
		if v := int16(binary.LittleEndian.Uint16(decoded[i:])); v < -4 || v > 4 {
			t.Fatalf("silence sample %d decoded to %d", i/2, v)
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
