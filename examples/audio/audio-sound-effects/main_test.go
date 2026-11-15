package main

import "testing"

// TestSoundsDecode checks both embedded sound effects decode to 24 kHz mono
// 16-bit audio.
func TestSoundsDecode(t *testing.T) {
	for name, data := range map[string][]byte{"ding1": ding1WAV, "ding2": ding2WAV} {
		s, err := decodeWAV(data)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if s.sampleRate != 24000 || s.numChannels != 1 || len(s.pcm) == 0 || len(s.pcm)%2 != 0 {
			t.Fatalf("%s: rate %d, channels %d, %d bytes", name, s.sampleRate, s.numChannels, len(s.pcm))
		}
	}
}
