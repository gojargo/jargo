package stt

import "github.com/gojargo/jargo/audio"

// WAV wraps 16-bit PCM in a minimal RIFF/WAVE container. Batch transcription
// APIs accept an audio file rather than raw PCM, so segmented providers wrap the
// buffered samples before uploading.
//
// It is a thin alias for audio.PCMToWAV, kept so a provider does not have to
// reach past the service layer for it.
func WAV(pcm []byte, sampleRate, channels int) []byte {
	return audio.PCMToWAV(pcm, sampleRate, channels)
}
