package audio_test

import (
	"bytes"
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

// samples decodes S16LE PCM back to signed samples.
func samples(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

func equal(got []int16, want ...int16) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestMixAudio(t *testing.T) {
	tests := []struct {
		name string
		a, b []byte
		want []int16
	}{
		{"same length sums sample by sample", pcm(100, 200), pcm(10, 20), []int16{110, 220}},
		{"the shorter side is padded with silence", pcm(100, 200, 300), pcm(10), []int16{110, 200, 300}},
		{"the other way round", pcm(10), pcm(100, 200, 300), []int16{110, 200, 300}},
		{"positive overflow clips", pcm(32000), pcm(32000), []int16{32767}},
		{"negative overflow clips", pcm(-32000), pcm(-32000), []int16{-32768}},
		{"both empty", nil, nil, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := samples(audio.MixAudio(tt.a, tt.b)); !equal(got, tt.want...) {
				t.Errorf("MixAudio() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInterleaveStereo(t *testing.T) {
	tests := []struct {
		name        string
		left, right []byte
		want        []int16
	}{
		{"equal channels weave together", pcm(1, 2), pcm(-1, -2), []int16{1, -1, 2, -2}},
		{"truncated to the shorter channel", pcm(1, 2, 3), pcm(-1), []int16{1, -1}},
		{"the other way round", pcm(1), pcm(-1, -2, -3), []int16{1, -1}},
		{"an empty channel yields nothing", nil, pcm(1, 2), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := samples(audio.InterleaveStereo(tt.left, tt.right)); !equal(got, tt.want...) {
				t.Errorf("InterleaveStereo() = %v, want %v", got, tt.want)
			}
		})
	}
}

// readWAV pulls the format fields and the data payload back out of a WAV file,
// so a test can check what a reader would actually see.
func readWAV(t *testing.T, wav []byte) (numChannels, sampleWidth, sampleRate int, data []byte) {
	t.Helper()
	if len(wav) < 44 {
		t.Fatalf("WAV is %d bytes, too short to hold a header", len(wav))
	}
	if string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		t.Fatalf("not a RIFF/WAVE file: %q", wav[:12])
	}
	if got := binary.LittleEndian.Uint32(wav[4:]); int(got) != len(wav)-8 {
		t.Errorf("RIFF size = %d, want %d", got, len(wav)-8)
	}
	numChannels = int(binary.LittleEndian.Uint16(wav[22:]))
	sampleRate = int(binary.LittleEndian.Uint32(wav[24:]))
	sampleWidth = int(binary.LittleEndian.Uint16(wav[34:])) / 8
	if string(wav[36:40]) != "data" {
		t.Fatalf("expected a data chunk at offset 36, got %q", wav[36:40])
	}
	dataLen := int(binary.LittleEndian.Uint32(wav[40:]))
	if dataLen != len(wav)-44 {
		t.Errorf("data chunk size = %d, but %d bytes follow it", dataLen, len(wav)-44)
	}
	return numChannels, sampleWidth, sampleRate, wav[44:]
}

func TestPCMToWAV(t *testing.T) {
	t.Run("mono", func(t *testing.T) {
		in := bytes.Repeat([]byte{0x01, 0x00}, 1600) // 0.1s at 16kHz
		ch, width, rate, data := readWAV(t, audio.PCMToWAV(in, 16000, 1))
		if ch != 1 || width != 2 || rate != 16000 {
			t.Errorf("format = %d channels, %d bytes/sample, %d Hz", ch, width, rate)
		}
		if !bytes.Equal(data, in) {
			t.Error("the samples did not survive the round trip")
		}
	})

	t.Run("stereo", func(t *testing.T) {
		in := bytes.Repeat([]byte{0x01, 0x00, 0x02, 0x00}, 2400) // 0.1s at 24kHz
		ch, width, rate, data := readWAV(t, audio.PCMToWAV(in, 24000, 2))
		if ch != 2 || width != 2 || rate != 24000 {
			t.Errorf("format = %d channels, %d bytes/sample, %d Hz", ch, width, rate)
		}
		if !bytes.Equal(data, in) {
			t.Error("the samples did not survive the round trip")
		}
	})

	t.Run("empty", func(t *testing.T) {
		_, _, rate, data := readWAV(t, audio.PCMToWAV(nil, 16000, 1))
		if rate != 16000 {
			t.Errorf("sample rate = %d, want 16000", rate)
		}
		if len(data) != 0 {
			t.Errorf("data = %d bytes, want none", len(data))
		}
	})

	// A frame is only complete when every channel has a sample in it. Keeping a
	// half frame would leave the header and the data chunk disagreeing about the
	// length, and a reader would take the stray sample for the start of the next
	// frame, swapping the channels from there on.
	t.Run("drops a partial trailing frame", func(t *testing.T) {
		in := append(bytes.Repeat([]byte{0x01, 0x00, 0x02, 0x00}, 100), 0x03, 0x00)
		ch, _, _, data := readWAV(t, audio.PCMToWAV(in, 24000, 2))
		if ch != 2 {
			t.Errorf("channels = %d, want 2", ch)
		}
		if !bytes.Equal(data, in[:len(in)-2]) {
			t.Error("the lone trailing sample should have been dropped")
		}
	})
}
