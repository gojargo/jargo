package audio

import "time"

// DefaultPreSpeech is how much audio a SpeechSegmentBuffer keeps from before
// speech starts by default. The VAD can report speech most of a second after it
// starts.
const DefaultPreSpeech = time.Second

// SpeechSegmentBuffer keeps the audio of one stretch of speech, with the audio
// just before it.
//
// Between stretches of speech it keeps only the last PreSpeech of audio, which
// holds the start of the speech the VAD reports late. From Start to Stop it
// keeps everything, or the last Max of it, and Stop returns the segment. The
// audio is 16-bit mono PCM.
//
//	buffer := audio.NewSpeechSegmentBuffer(audio.DefaultPreSpeech, 0)
//	buffer.SetSampleRate(16000)
//
//	buffer.Append(pcm)        // every input audio chunk
//	buffer.Start()            // when the VAD reports speech
//	segment := buffer.Stop()  // when the VAD reports silence
//
// It is not safe for concurrent use.
type SpeechSegmentBuffer struct {
	preSpeech time.Duration
	max       time.Duration

	preSpeechSize int
	// maxSize is the most bytes a segment keeps, or 0 for no limit.
	maxSize  int
	buf      []byte
	speaking bool
}

// NewSpeechSegmentBuffer builds a buffer keeping preSpeech of audio from before
// speech starts, and at most the last max of a segment; a max of zero keeps it
// all. Set the sample rate before appending audio.
func NewSpeechSegmentBuffer(preSpeech, maxDuration time.Duration) *SpeechSegmentBuffer {
	return &SpeechSegmentBuffer{preSpeech: max(preSpeech, 0), max: max(maxDuration, 0)}
}

// SetSampleRate sets the sample rate of the audio to come, in Hz.
func (b *SpeechSegmentBuffer) SetSampleRate(sampleRate int) {
	b.preSpeechSize = durationBytes(sampleRate, b.preSpeech)
	b.maxSize = durationBytes(sampleRate, b.max)
}

// durationBytes is the size of d of 16-bit mono audio at sampleRate.
func durationBytes(sampleRate int, d time.Duration) int {
	return int(float64(sampleRate)*d.Seconds()) * 2
}

// Append adds audio, dropping the oldest beyond what the buffer keeps.
func (b *SpeechSegmentBuffer) Append(pcm []byte) {
	b.buf = append(b.buf, pcm...)
	limit, limited := b.preSpeechSize, true
	if b.speaking {
		limit, limited = b.maxSize, b.max > 0
	}
	if limited && len(b.buf) > limit {
		b.buf = append([]byte(nil), b.buf[len(b.buf)-limit:]...)
	}
}

// Start starts a segment, with the audio kept from before it.
func (b *SpeechSegmentBuffer) Start() { b.speaking = true }

// Stop ends the segment and returns its audio, emptying the buffer.
func (b *SpeechSegmentBuffer) Stop() []byte {
	b.speaking = false
	segment := b.buf
	b.buf = nil
	return segment
}
