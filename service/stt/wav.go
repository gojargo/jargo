package stt

import (
	"bytes"
	"encoding/binary"
)

// wavHeaderSize is the byte length of the canonical 44-byte PCM WAV header.
const wavHeaderSize = 36

// WAV wraps 16-bit PCM in a minimal RIFF/WAVE container. Batch transcription
// APIs accept an audio file rather than raw PCM, so segmented providers wrap the
// buffered samples before uploading.
func WAV(pcm []byte, sampleRate, channels int) []byte {
	dataLen := len(pcm)
	byteRate := sampleRate * channels * 2
	blockAlign := channels * 2

	var buf bytes.Buffer
	buf.Grow(wavHeaderSize + 8 + dataLen)
	buf.WriteString("RIFF")
	writeLE(&buf, uint32(wavHeaderSize+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	writeLE(&buf, uint32(16))         // PCM fmt chunk size.
	writeLE(&buf, uint16(1))          // Audio format: PCM.
	writeLE(&buf, uint16(channels))   //nolint:gosec // channel count is small.
	writeLE(&buf, uint32(sampleRate)) //nolint:gosec // sample rate is bounded.
	writeLE(&buf, uint32(byteRate))   //nolint:gosec // derived from bounded inputs.
	writeLE(&buf, uint16(blockAlign)) //nolint:gosec // block align is small.
	writeLE(&buf, uint16(16))         // Bits per sample.
	buf.WriteString("data")
	writeLE(&buf, uint32(dataLen)) //nolint:gosec // payload length fits a session.
	buf.Write(pcm)
	return buf.Bytes()
}

func writeLE(buf *bytes.Buffer, v any) {
	_ = binary.Write(buf, binary.LittleEndian, v)
}
