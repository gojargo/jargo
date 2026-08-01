package localaudio

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport"
)

func TestInt16ToPCMRoundTrip(t *testing.T) {
	samples := []int16{0, 1, -1, 32767, -32768, 1234}
	pcm := int16ToPCM(samples)
	if len(pcm) != len(samples)*2 {
		t.Fatalf("pcm len = %d, want %d", len(pcm), len(samples)*2)
	}
	for i, s := range samples {
		if got := int16(binary.LittleEndian.Uint16(pcm[2*i:])); got != s {
			t.Fatalf("sample %d = %d, want %d", i, got, s)
		}
	}
}

// TestFillDrainsThenPads checks that fill drains queued PCM in order and pads the
// remainder with silence, always filling the whole request so the stream keeps
// running.
func TestFillDrainsThenPads(t *testing.T) {
	out := newOutput(nil, transport.DefaultParams())
	chunk := frames.NewOutputAudioRawFrame(int16ToPCM([]int16{100, -200}), 24000, 1)
	if err := out.WriteAudio(context.Background(), chunk); err != nil {
		t.Fatalf("WriteAudio: %v", err)
	}

	got := make([]int16, 4) // ask for more than is queued
	n, err := out.fill(got)
	if err != nil {
		t.Fatalf("fill: %v", err)
	}
	if n != 4 {
		t.Fatalf("fill returned %d, want the full 4 (silence-padded)", n)
	}
	want := []int16{100, -200, 0, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sample %d = %d, want %d", i, got[i], want[i])
		}
	}

	// The queue is now empty: the next fill is pure silence.
	next := make([]int16, 3)
	if _, err := out.fill(next); err != nil {
		t.Fatalf("fill: %v", err)
	}
	for i, s := range next {
		if s != 0 {
			t.Fatalf("silence sample %d = %d, want 0", i, s)
		}
	}
}

// TestWriteAudioCapsBacklog checks the playback backlog is bounded so a stalled
// speaker cannot grow the buffer without limit.
func TestWriteAudioCapsBacklog(t *testing.T) {
	out := newOutput(nil, transport.DefaultParams())
	capacity := maxBufferBytes(0)
	chunk := frames.NewOutputAudioRawFrame(make([]byte, capacity+4096), 24000, 1)
	if err := out.WriteAudio(context.Background(), chunk); err != nil {
		t.Fatalf("WriteAudio: %v", err)
	}
	if len(out.buf) > capacity {
		t.Fatalf("buffer len = %d, want <= %d", len(out.buf), capacity)
	}
}
