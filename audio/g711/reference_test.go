package g711

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"testing"
)

// The digests are of the Sun reference implementation's output for the same
// input, so a match means the codecs are bit-exact with it on every sample and
// every code, not only that they round-trip.

// pcm packs samples as 16-bit signed little-endian PCM.
func pcm(samples ...int16) []byte {
	out := make([]byte, 2*len(samples))
	for i, s := range samples {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(s))
	}
	return out
}

// everySample is every 16-bit sample, from -32768 to 32767, in order.
func everySample() []byte {
	samples := make([]int16, 0, 65536)
	for s := -32768; s <= 32767; s++ {
		samples = append(samples, int16(s))
	}
	return pcm(samples...)
}

// everyCode is every 8-bit code, from 0 to 255, in order.
func everyCode() []byte {
	codes := make([]byte, 256)
	for i := range codes {
		codes[i] = byte(i)
	}
	return codes
}

func assertBytes(t *testing.T, name string, got, want []byte) {
	t.Helper()
	if !bytes.Equal(got, want) {
		t.Errorf("%s = % x, want % x", name, got, want)
	}
}

func assertDigest(t *testing.T, name string, data []byte, want string) {
	t.Helper()
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != want {
		t.Errorf("%s: sha256 %s, want %s", name, got, want)
	}
}

func TestEncodeULawMatchesTheReferenceOutput(t *testing.T) {
	assertBytes(t, "EncodeULaw(0, 32767, -32768)", encode(t, EncodeULaw, pcm(0, 32767, -32768)), []byte{0xff, 0x80, 0x00})
	assertDigest(t, "EncodeULaw(every sample)", encode(t, EncodeULaw, everySample()),
		"81d633c9e6972a18c74a58720b96cb8ca0bdd096d4060b646dd708c3b846019a")
}

func TestDecodeULawMatchesTheReferenceOutput(t *testing.T) {
	assertBytes(t, "DecodeULaw(ff 80 00)", DecodeULaw([]byte{0xff, 0x80, 0x00}), pcm(0, 32124, -32124))
	assertDigest(t, "DecodeULaw(every code)", DecodeULaw(everyCode()),
		"3dab54339e520bb2c924826e3b72a917a2b612e9fd12fc867500f1d983a75827")
}

func TestEncodeALawMatchesTheReferenceOutput(t *testing.T) {
	assertBytes(t, "EncodeALaw(0, 32767, -32768)", encode(t, EncodeALaw, pcm(0, 32767, -32768)), []byte{0xd5, 0xaa, 0x2a})
	assertDigest(t, "EncodeALaw(every sample)", encode(t, EncodeALaw, everySample()),
		"38488f6fd710f4686360edc4d38639f96c491595ef93f8eb8d62d5e07ca6ce7b")
}

func TestDecodeALawMatchesTheReferenceOutput(t *testing.T) {
	assertBytes(t, "DecodeALaw(d5 55 80 00)", DecodeALaw([]byte{0xd5, 0x55, 0x80, 0x00}), pcm(8, -8, 5504, -5504))
	assertDigest(t, "DecodeALaw(every code)", DecodeALaw(everyCode()),
		"e04788d110e58ff8c70c93b8480190d973e3b67876b6119abbaec766cc75c174")
}

// Empty audio converts to empty audio in every direction.
func TestEmptyAudioConvertsToEmptyAudio(t *testing.T) {
	must := func(enc func([]byte) ([]byte, error)) func([]byte) []byte {
		return func(pcm []byte) []byte { return encode(t, enc, pcm) }
	}
	for name, convert := range map[string]func([]byte) []byte{
		"EncodeULaw": must(EncodeULaw),
		"DecodeULaw": DecodeULaw,
		"EncodeALaw": must(EncodeALaw),
		"DecodeALaw": DecodeALaw,
	} {
		for _, in := range [][]byte{nil, {}} {
			if got := convert(in); len(got) != 0 {
				t.Errorf("%s(%v) = % x, want empty", name, in, got)
			}
		}
	}
}
