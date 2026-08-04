package moshi

import "encoding/binary"

// audioMsg decodes a moshi {"type":"Audio","pcm":[...]} message: TTS returns it
// carrying 24 kHz float32 synthesized PCM.
type audioMsg struct {
	Type string    `msgpack:"type"`
	PCM  []float32 `msgpack:"pcm"`
}

// sttMsg decodes the subset of moshi's STT result messages we act on. A "Word"
// message carries a transcribed word in Text; a "Step" message carries the
// semantic-VAD pause predictions in Prs (one float per look-ahead window).
type sttMsg struct {
	Type string    `msgpack:"type"`
	Text string    `msgpack:"text"`
	Prs  []float64 `msgpack:"prs"`
}

// int16BytesToFloat32 appends the S16LE PCM samples in b to dst as float32 in
// [-1, 1).
func int16BytesToFloat32(dst []float32, b []byte) []float32 {
	for i := 0; i+1 < len(b); i += 2 {
		s := int16(binary.LittleEndian.Uint16(b[i:]))
		dst = append(dst, float32(s)/32768.0)
	}
	return dst
}

// float32ToInt16Bytes converts float32 PCM samples to S16LE PCM bytes, clamping
// to the int16 range.
func float32ToInt16Bytes(pcm []float32) []byte {
	out := make([]byte, len(pcm)*2)
	for i, f := range pcm {
		v := f * 32767.0
		switch {
		case v > 32767:
			v = 32767
		case v < -32768:
			v = -32768
		}
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(v)))
	}
	return out
}
