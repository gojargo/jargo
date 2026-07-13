package g711

const (
	// alawQuantMask selects the quantization bits of an A-law byte.
	alawQuantMask = 0x0F
	// alawSegMask selects the segment bits; alawSegShift positions them.
	alawSegMask  = 0x70
	alawSegShift = 4
	// alawEvenBits toggles the even bits, the A-law even-bit inversion applied on
	// both encode and decode.
	alawEvenBits = 0x55
)

// alawSegEnd holds the upper bound of each of the eight A-law segments.
//
//nolint:gochecknoglobals // immutable codec table
var alawSegEnd = [8]int16{0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF}

// alawDecodeTable expands each A-law byte to its 16-bit PCM sample.
//
//nolint:gochecknoglobals // immutable codec table, built once
var alawDecodeTable = buildALawDecodeTable()

// EncodeALaw encodes 16-bit signed little-endian PCM to A-law (G.711 PCMA),
// returning one byte per sample. A trailing odd byte is ignored. A-law is the
// PSTN companding used outside North America (Telnyx's PCMA encoding).
func EncodeALaw(pcm []byte) []byte {
	out := make([]byte, len(pcm)/2)
	for i := 0; i+1 < len(pcm); i += 2 {
		sample := int16(uint16(pcm[i]) | uint16(pcm[i+1])<<8)
		out[i/2] = linearToALaw(sample)
	}
	return out
}

// DecodeALaw decodes A-law bytes to 16-bit signed little-endian PCM, returning
// two bytes per input byte.
func DecodeALaw(alaw []byte) []byte {
	out := make([]byte, len(alaw)*2)
	for i, a := range alaw {
		s := uint16(alawDecodeTable[a])
		out[i*2] = byte(s)
		out[i*2+1] = byte(s >> 8)
	}
	return out
}

func linearToALaw(pcm int16) byte {
	v := pcm >> 3 // scale 16-bit to the codec's 13-bit range
	var mask int16
	if v >= 0 {
		mask = 0xD5
	} else {
		v = -v - 1
		mask = alawEvenBits
	}

	seg := alawSegment(v)
	if seg >= 8 {
		return byte(0x7F ^ mask)
	}
	aval := int16(seg << alawSegShift)
	if seg < 2 {
		aval |= (v >> 1) & alawQuantMask
	} else {
		aval |= (v >> uint(seg)) & alawQuantMask
	}
	return byte(aval ^ mask)
}

// alawSegment returns the index of the segment v falls in, or 8 if it exceeds
// the last segment.
func alawSegment(v int16) int {
	for i, end := range alawSegEnd {
		if v <= end {
			return i
		}
	}
	return 8
}

func alawToLinear(a byte) int16 {
	a ^= alawEvenBits
	t := int16(a&alawQuantMask) << 4
	seg := (int(a) & alawSegMask) >> alawSegShift
	switch seg {
	case 0:
		t += 8
	case 1:
		t += 0x108
	default:
		t += 0x108
		t <<= uint(seg - 1)
	}
	if a&0x80 != 0 {
		return t
	}
	return -t
}

func buildALawDecodeTable() [256]int16 {
	var t [256]int16
	for i := range 256 {
		t[i] = alawToLinear(byte(i))
	}
	return t
}
