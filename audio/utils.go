package audio

import (
	"bytes"
	"encoding/binary"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/audio/resample"
)

// speakingThreshold is the largest absolute 16-bit sample amplitude a buffer can
// contain and still count as silence. Speech normally reaches amplitudes of
// roughly 500 to 5000 depending on loudness and microphone gain, so the
// threshold sits well below speech and distinguishes it from a silent stream.
const speakingThreshold = 20

// IsSilence reports whether a chunk of 16-bit signed PCM is silence, by
// comparing the largest absolute sample amplitude against speakingThreshold.
//
// It expects clean speech or true silence: a stream with audible background
// noise keeps the amplitude above the threshold and never reads as silent. An
// empty buffer carries no audible sample and so counts as silence.
func IsSilence(pcm []byte) bool {
	for i := 0; i+1 < len(pcm); i += 2 {
		// Widen to int before negating: the most negative 16-bit sample has no
		// positive counterpart, so negating it in place would wrap.
		sample := int(int16(binary.LittleEndian.Uint16(pcm[i:])))
		if sample < 0 {
			sample = -sample
		}
		if sample > speakingThreshold {
			return false
		}
	}
	return true
}

// MixAudio sums two streams of 16-bit signed PCM sample by sample, clipping the
// result to the 16-bit range. The streams need not be the same length: the
// shorter one is treated as though it were padded with silence, so the result is
// as long as the longer input.
func MixAudio(a, b []byte) []byte {
	n := max(len(a), len(b))
	n -= n % 2
	out := make([]byte, n)
	for i := 0; i+1 < n; i += 2 {
		var s int32
		if i+1 < len(a) {
			s += int32(int16(binary.LittleEndian.Uint16(a[i:])))
		}
		if i+1 < len(b) {
			s += int32(int16(binary.LittleEndian.Uint16(b[i:])))
		}
		binary.LittleEndian.PutUint16(out[i:], uint16(clampInt16(s)))
	}
	return out
}

// InterleaveStereo weaves two mono streams of 16-bit signed PCM into one stereo
// stream, left channel first (L, R, L, R and so on). The channels are truncated
// to the shorter of the two, since a frame is only complete when both channels
// have a sample in it.
func InterleaveStereo(left, right []byte) []byte {
	n := min(len(left), len(right))
	n -= n % 2
	out := make([]byte, n*2)
	for i := 0; i+1 < n; i += 2 {
		copy(out[i*2:], left[i:i+2])
		copy(out[i*2+2:], right[i:i+2])
	}
	return out
}

// clampInt16 saturates a widened sample to the 16-bit range.
func clampInt16(v int32) int16 {
	switch {
	case v > 32767:
		return 32767
	case v < -32768:
		return -32768
	default:
		return int16(v)
	}
}

// wavHeaderLen is the length of the canonical PCM WAV header that precedes the
// samples: the RIFF chunk's own header is 8 bytes on top of this.
const wavHeaderLen = 36

// PCMToWAV wraps raw PCM in a WAV container. The samples are expected to be
// signed 16-bit little-endian, which is what pipelines carry, interleaved by
// channel.
//
// Trailing bytes that do not complete a frame are dropped. Keeping them would
// leave the header reporting a frame count that excludes them while the data
// chunk still carries them, so readers would disagree about the length, and a
// stereo stream cut mid-frame would come back with its channels swapped.
func PCMToWAV(pcm []byte, sampleRate, numChannels int) []byte {
	if numChannels < 1 {
		numChannels = 1
	}
	blockAlign := 2 * numChannels
	if rem := len(pcm) % blockAlign; rem != 0 {
		pcm = pcm[:len(pcm)-rem]
	}

	var buf bytes.Buffer
	buf.Grow(wavHeaderLen + 8 + len(pcm))
	buf.WriteString("RIFF")
	writeLE(&buf, uint32(wavHeaderLen+len(pcm))) //nolint:gosec // payload length fits a session
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	writeLE(&buf, uint32(16))                    // PCM fmt chunk size.
	writeLE(&buf, uint16(1))                     // Audio format: PCM.
	writeLE(&buf, uint16(numChannels))           //nolint:gosec // channel count is small
	writeLE(&buf, uint32(sampleRate))            //nolint:gosec // sample rate is bounded
	writeLE(&buf, uint32(sampleRate*blockAlign)) //nolint:gosec // derived from bounded inputs
	writeLE(&buf, uint16(blockAlign))            //nolint:gosec // block align is small
	writeLE(&buf, uint16(16))                    // Bits per sample.
	buf.WriteString("data")
	writeLE(&buf, uint32(len(pcm))) //nolint:gosec // payload length fits a session
	buf.Write(pcm)
	return buf.Bytes()
}

func writeLE(buf *bytes.Buffer, v any) {
	_ = binary.Write(buf, binary.LittleEndian, v)
}

// ULawToPCM decodes G.711 μ-law to 16-bit PCM and converts it to the rate r
// runs at. A nil r leaves the rate alone.
//
// Decoding comes first: companded bytes are not samples, so resampling them
// would interpolate between values on a logarithmic scale as though they were
// linear.
func ULawToPCM(ulaw []byte, r *resample.Resampler) []byte {
	return resampleWith(g711.DecodeULaw(ulaw), r)
}

// PCMToULaw converts 16-bit PCM to the rate r runs at and encodes it as G.711
// μ-law. A nil r leaves the rate alone. Resampling comes first, for the reason
// given on ULawToPCM.
func PCMToULaw(pcm []byte, r *resample.Resampler) []byte {
	return g711.EncodeULaw(resampleWith(pcm, r))
}

// ALawToPCM decodes G.711 A-law to 16-bit PCM and converts it to the rate r
// runs at. A nil r leaves the rate alone. See ULawToPCM.
func ALawToPCM(alaw []byte, r *resample.Resampler) []byte {
	return resampleWith(g711.DecodeALaw(alaw), r)
}

// PCMToALaw converts 16-bit PCM to the rate r runs at and encodes it as G.711
// A-law. A nil r leaves the rate alone. See ULawToPCM.
func PCMToALaw(pcm []byte, r *resample.Resampler) []byte {
	return g711.EncodeALaw(resampleWith(pcm, r))
}

// ResamplePCM converts 16-bit PCM to the rate r runs at, leaving the samples
// otherwise untouched. A nil r returns the audio unchanged. It is the
// companding-free member of the family above, for a stream that is already
// linear PCM and only needs its rate changed.
func ResamplePCM(pcm []byte, r *resample.Resampler) []byte {
	return resampleWith(pcm, r)
}

// resampleWith runs pcm through r, or returns it unchanged when there is no
// resampler to run it through.
func resampleWith(pcm []byte, r *resample.Resampler) []byte {
	if r == nil {
		return pcm
	}
	return r.Process(pcm)
}
