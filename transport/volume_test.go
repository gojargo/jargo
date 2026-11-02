package transport_test

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport"
)

// The output volume, and the gain on it, scale the audio that follows them.

// volumeAudio is one chunk of audio, every sample set to sample.
func volumeAudio(sample int16) *frames.OutputAudioRawFrame {
	pcm := make([]byte, filterChunk)
	for i := 0; i < len(pcm); i += 2 {
		binary.LittleEndian.PutUint16(pcm[i:], uint16(sample))
	}
	return frames.NewOutputAudioRawFrame(pcm, filterTestRate, 1)
}

func volumeParams(volume *float64) transport.Params {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = filterTestRate
	params.AudioOutVolume = volume
	return params
}

// firstSample is a chunk's first sample.
func firstSample(chunk []byte) int16 {
	return int16(binary.LittleEndian.Uint16(chunk))
}

// recvFirstSamples takes n writes and returns the first sample of each.
func recvFirstSamples(t *testing.T, o *fakeOutput, n int) []int16 {
	t.Helper()
	out := make([]int16, n)
	for i := range out {
		out[i] = firstSample(recvWrite(t, o))
	}
	return out
}

func assertSamples(t *testing.T, got, want []int16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("first samples = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("first samples = %v, want %v", got, want)
		}
	}
}

func TestBaseOutputWritesAudioAtItsVolume(t *testing.T) {
	volume := 0.5
	o := newFakeOutput(volumeParams(&volume))
	task, stop := runOutput(t, o)
	defer stop()

	task.QueueFrame(volumeAudio(1000))
	assertSamples(t, recvFirstSamples(t, o, 1), []int16{500})
}

func TestBaseOutputVolumeAndGainScaleTheAudioAfterThem(t *testing.T) {
	o := newFakeOutput(volumeParams(nil))
	task, stop := runOutput(t, o)
	defer stop()

	task.QueueFrames([]frames.Frame{
		volumeAudio(1000),
		frames.NewVolumeFrame(0.5),
		volumeAudio(1000),
		frames.NewVolumeGainFrame(0.5),
		volumeAudio(1000),
		frames.NewVolumeGainFrame(1.0),
		volumeAudio(1000),
	})
	assertSamples(t, recvFirstSamples(t, o, 4), []int16{1000, 500, 250, 500})
}

func TestBaseOutputInterruptionKeepsTheGainFrames(t *testing.T) {
	o := newGatedOutput(volumeParams(nil))
	task, _, stop := runLogged(t, o)
	defer stop()

	// The first chunk holds the write, so the gain frames and the quiet chunk
	// between them are still queued when the user interrupts.
	task.QueueFrame(volumeAudio(1000))
	waitEntered(t, o.entered)
	task.QueueFrames([]frames.Frame{
		frames.NewVolumeGainFrame(0.5),
		volumeAudio(1000),
		frames.NewVolumeGainFrame(1.0),
	})
	time.Sleep(50 * time.Millisecond)
	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(50 * time.Millisecond)
	close(o.gate)
	task.QueueFrame(volumeAudio(1000))

	// The quiet chunk is dropped, and the audio after it plays at full volume
	// again.
	deadline := time.Now().Add(3 * time.Second)
	for len(o.writtenAudio()) < 2*filterChunk {
		if time.Now().After(deadline) {
			t.Fatalf("%d bytes written, want two chunks", len(o.writtenAudio()))
		}
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(100 * time.Millisecond)
	written := o.writtenAudio()
	var got []int16
	for i := 0; i+filterChunk <= len(written); i += filterChunk {
		got = append(got, firstSample(written[i:]))
	}
	assertSamples(t, got, []int16{1000, 1000})
}

func TestBaseOutputALoudGainClipsToTheSampleRange(t *testing.T) {
	o := newFakeOutput(volumeParams(nil))
	task, stop := runOutput(t, o)
	defer stop()

	task.QueueFrames([]frames.Frame{frames.NewVolumeGainFrame(2.0), volumeAudio(30000)})
	assertSamples(t, recvFirstSamples(t, o, 1), []int16{32767})
}
