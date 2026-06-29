// Package g711 implements the ITU-T G.711 μ-law companding codec used by
// telephony media streams (Twilio, Telnyx, Plivo and the wider PSTN). It
// converts between 16-bit signed little-endian PCM and 8-bit μ-law, one byte per
// sample, mono. The algorithm is the canonical Sun reference implementation, so
// it round-trips with audioop.lin2ulaw/ulaw2lin and the like.
package g711

const (
	// ulawBias is added to magnitudes before segment search (G.711 BIAS, 0x84).
	ulawBias = 0x84
	// ulawClip is the largest representable magnitude after the 16→14-bit scale.
	ulawClip = 8159
)

// ulawSegEnd holds the upper bound of each of the eight μ-law segments.
//
//nolint:gochecknoglobals // immutable codec table
var ulawSegEnd = [8]int16{0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF}

// ulawDecodeTable expands each μ-law byte to its 16-bit PCM sample.
//
//nolint:gochecknoglobals // immutable codec table, built once
var ulawDecodeTable = buildULawDecodeTable()

// EncodeULaw encodes 16-bit signed little-endian PCM to μ-law, returning one
// byte per sample. A trailing odd byte (an incomplete sample) is ignored.
func EncodeULaw(pcm []byte) []byte {
	out := make([]byte, len(pcm)/2)
	for i := 0; i+1 < len(pcm); i += 2 {
		sample := int16(uint16(pcm[i]) | uint16(pcm[i+1])<<8)
		out[i/2] = linearToULaw(sample)
	}
	return out
}

// DecodeULaw decodes μ-law bytes to 16-bit signed little-endian PCM, returning
// two bytes per input byte.
func DecodeULaw(ulaw []byte) []byte {
	out := make([]byte, len(ulaw)*2)
	for i, u := range ulaw {
		s := uint16(ulawDecodeTable[u])
		out[i*2] = byte(s)
		out[i*2+1] = byte(s >> 8)
	}
	return out
}

func linearToULaw(pcm int16) byte {
	v := pcm >> 2 // scale 16-bit to the codec's 14-bit range
	var mask int16
	if v < 0 {
		v = -v
		mask = 0x7F
	} else {
		mask = 0xFF
	}
	if v > ulawClip {
		v = ulawClip
	}
	v += ulawBias >> 2

	seg := ulawSegment(v)
	if seg >= 8 {
		return byte(0x7F ^ mask)
	}
	uval := int16(seg<<4) | ((v >> (uint(seg) + 1)) & 0x0F)
	return byte(uval ^ mask)
}

// ulawSegment returns the index of the segment v falls in, or 8 if it exceeds
// the last segment.
func ulawSegment(v int16) int {
	for i, end := range ulawSegEnd {
		if v <= end {
			return i
		}
	}
	return 8
}

func ulawToLinear(u byte) int16 {
	u = ^u
	t := int16(u&0x0F)<<3 + ulawBias
	t <<= uint(u&0x70) >> 4
	if u&0x80 != 0 {
		return ulawBias - t
	}
	return t - ulawBias
}

func buildULawDecodeTable() [256]int16 {
	var t [256]int16
	for i := range 256 {
		t[i] = ulawToLinear(byte(i))
	}
	return t
}
