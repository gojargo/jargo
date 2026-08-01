package audio_test

import (
	"encoding/binary"
	"testing"

	"github.com/gojargo/jargo/audio"
)

// pcm builds a little-endian S16LE buffer from samples.
func pcm(samples ...int16) []byte {
	b := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(s))
	}
	return b
}

func TestIsSilence(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		want bool
	}{
		{"empty buffer", nil, true},
		{"all zero samples", pcm(0, 0, 0, 0), true},
		{"at the threshold", pcm(20, -20), true},
		{"just past the threshold", pcm(0, 21), false},
		{"just past it going negative", pcm(0, -21), false},
		{"speech level", pcm(0, 1200, -3000), false},
		{"the most negative sample", pcm(-32768), false},
		{"one loud sample among quiet ones", pcm(1, -2, 5000, 3), false},
		{"odd trailing byte is ignored", append(pcm(0, 0), 0xFF), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := audio.IsSilence(tt.in); got != tt.want {
				t.Errorf("IsSilence() = %v, want %v", got, tt.want)
			}
		})
	}
}
