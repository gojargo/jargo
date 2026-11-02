package livekit

import (
	"github.com/gojargo/jargo/audio"
)

// participantAudioMixer mixes participants' audio, as it arrives, into a single
// stream.
//
// Each participant's audio is buffered and mixed in fixed-size chunks once
// every participant has a chunk buffered, so the output keeps the pace of the
// participants' streams. A participant who stops sending audio holds the mix
// back until another participant has maxWait buffered; then they are left out
// of the mix until their audio arrives again.
type participantAudioMixer struct {
	chunkBytes   int
	maxWaitBytes int
	// order is the participants in the order the mixer first heard from them,
	// which is the order their audio is summed in.
	order   []string
	buffers map[string][]byte
}

// newParticipantAudioMixer builds a mixer for 16-bit PCM at sampleRate with
// numChannels, mixing chunkMs chunks and waiting up to maxWaitMs for a
// participant with nothing buffered.
func newParticipantAudioMixer(sampleRate, numChannels, chunkMs, maxWaitMs int) *participantAudioMixer {
	bytesPerMs := sampleRate * numChannels * 2 / 1000
	return &participantAudioMixer{
		chunkBytes:   chunkMs * bytesPerMs,
		maxWaitBytes: maxWaitMs * bytesPerMs,
		buffers:      map[string][]byte{},
	}
}

// add adds a participant's 16-bit PCM audio and returns the mixed chunks it
// completes, oldest first.
func (m *participantAudioMixer) add(participant string, pcm []byte) [][]byte {
	if _, ok := m.buffers[participant]; !ok {
		m.order = append(m.order, participant)
	}
	m.buffers[participant] = append(m.buffers[participant], pcm...)

	var chunks [][]byte
	for {
		if m.anyBelow(m.chunkBytes) {
			if !m.anyAtLeast(m.maxWaitBytes) {
				break
			}
			m.dropSilent()
		}
		chunks = append(chunks, m.mixChunk())
	}
	return chunks
}

// anyBelow reports whether some participant has fewer than n bytes buffered.
func (m *participantAudioMixer) anyBelow(n int) bool {
	for _, b := range m.buffers {
		if len(b) < n {
			return true
		}
	}
	return false
}

// anyAtLeast reports whether some participant has n bytes or more buffered.
func (m *participantAudioMixer) anyAtLeast(n int) bool {
	for _, b := range m.buffers {
		if len(b) >= n {
			return true
		}
	}
	return false
}

// dropSilent leaves out of the mix every participant with nothing buffered.
func (m *participantAudioMixer) dropSilent() {
	kept := m.order[:0]
	for _, p := range m.order {
		if len(m.buffers[p]) == 0 {
			delete(m.buffers, p)
			continue
		}
		kept = append(kept, p)
	}
	m.order = kept
}

// mixChunk takes a chunk from every participant and sums them.
func (m *participantAudioMixer) mixChunk() []byte {
	var mixed []byte
	for i, p := range m.order {
		b := m.buffers[p]
		n := min(m.chunkBytes, len(b))
		part := b[:n:n]
		m.buffers[p] = b[n:]
		if i == 0 {
			mixed = append([]byte(nil), part...)
			continue
		}
		mixed = audio.MixAudio(mixed, part)
	}
	return mixed
}
