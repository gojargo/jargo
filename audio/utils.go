package audio

import "encoding/binary"

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
