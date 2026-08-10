package transport_test

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport"
)

// The output transport converts incoming audio to the rate it was started at,
// so a service that speaks at its own rate (24 kHz TTS into a 48 kHz WebRTC
// output, say) is heard at the right pitch. The tests below drive that
// conversion through the pipeline rather than calling the resampler directly:
// what matters is the wiring in the media sender, which decides when a
// resampler is created, reused, and rebuilt.

// dcPCM returns n bytes of constant-amplitude S16LE mono audio. A steady level
// survives rate conversion, so a test can tell converted audio apart from
// silence without depending on the converter's exact output.
func dcPCM(n int) []byte {
	pcm := make([]byte, n)
	for i := 0; i+1 < n; i += 2 {
		pcm[i], pcm[i+1] = 0x00, 0x40 // 16384
	}
	return pcm
}

// drainWrites collects everything written until nothing more arrives for quiet,
// which is how a test reads off one stream's worth of output. The send loop
// writes as fast as the transport accepts, so a short lull means the queued
// audio has all been paced out.
func drainWrites(t *testing.T, o *fakeOutput, quiet time.Duration) []byte {
	t.Helper()
	var pcm []byte
	deadline := time.After(5 * time.Second)
	for {
		select {
		case chunk := <-o.writes:
			pcm = append(pcm, chunk...)
		case <-time.After(quiet):
			return pcm
		case <-deadline:
			t.Fatalf("the writes never went quiet: %d bytes so far", len(pcm))
			return pcm
		}
	}
}

// TestBaseOutputResamplesAudioToTheTransportRate covers audio arriving at a
// different rate from the one the transport started at. Half-rate audio has to
// come out about twice as long, because the samples are being spread over the
// transport's faster clock; forwarding it unconverted would play the turn at
// double speed.
func TestBaseOutputResamplesAudioToTheTransportRate(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks
	// The closing silence would land in the same drain and blur the count.
	params.AudioOutEndSilenceSecs = 0

	o := newFakeOutput(params)
	task, stop := startFakeOutput(t, o)
	defer stop()

	// 9600 bytes at 24 kHz mono is 200ms of audio. At 48 kHz that is 19200
	// bytes, of which the sender writes whole 1920-byte chunks and buffers the
	// rest, so 9 chunks (17280 bytes) reach the transport.
	const in = 9600
	task.QueueFrame(frames.NewOutputAudioRawFrame(dcPCM(in), 24000, 1))

	got := drainWrites(t, o, 200*time.Millisecond)

	// Unconverted, the same audio would fill 5 chunks. Anything at or below
	// that means the frame went out at its own rate.
	if len(got) <= in {
		t.Errorf("wrote %d bytes for %d bytes of half-rate audio, want about twice the input: "+
			"the audio was not resampled to 48 kHz", len(got), in)
	}
	if len(got) > 2*in {
		t.Errorf("wrote %d bytes, want at most %d: more audio came out than the conversion accounts for", len(got), 2*in)
	}
	if allZero(got) {
		t.Error("the resampled audio is silent: the conversion dropped the signal")
	}
}

// TestBaseOutputRebuildsTheResamplerWhenTheInputRateChanges covers a sender
// that has already converted one stream and is then handed audio at a new rate,
// which happens whenever a service is swapped mid-call or a second one speaks
// at a rate of its own. The resampler is created lazily and held for reuse, so
// the rate it was built for has to be checked on every frame: one kept from the
// previous stream would convert by the wrong ratio and pitch-shift the turn.
func TestBaseOutputRebuildsTheResamplerWhenTheInputRateChanges(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks
	params.AudioOutEndSilenceSecs = 0

	o := newFakeOutput(params)
	task, stop := startFakeOutput(t, o)
	defer stop()

	const in = 9600

	// First stream at 24 kHz builds the resampler.
	task.QueueFrame(frames.NewOutputAudioRawFrame(dcPCM(in), 24000, 1))
	first := drainWrites(t, o, 200*time.Millisecond)
	if len(first) <= in {
		t.Fatalf("first stream wrote %d bytes for %d bytes in, want about twice: it was not resampled", len(first), in)
	}

	// Second stream at 16 kHz on the same sender. 9600 bytes is 300ms, which is
	// 28800 bytes at 48 kHz: three times the input, not the two the resampler
	// was built for.
	task.QueueFrame(frames.NewOutputAudioRawFrame(dcPCM(in), 16000, 1))
	second := drainWrites(t, o, 200*time.Millisecond)

	// A resampler carried over from the 24 kHz stream would double the audio
	// instead of tripling it, landing near 19200 bytes. The bound sits between
	// the two so it separates them with room for the converter's filter delay.
	const doubled = 2 * in
	if len(second) <= doubled+2400 {
		t.Errorf("second stream wrote %d bytes for %d bytes of third-rate audio, want about %d: "+
			"the resampler was not rebuilt for 16 kHz", len(second), in, 3*in)
	}
	if len(second) > 3*in+1920 {
		t.Errorf("second stream wrote %d bytes, want at most %d: "+
			"more audio came out than the conversion accounts for", len(second), 3*in+1920)
	}
	if allZero(second) {
		t.Error("the resampled audio is silent: the conversion dropped the signal")
	}
}

// allZero reports whether pcm is entirely silence.
func allZero(pcm []byte) bool {
	for _, b := range pcm {
		if b != 0 {
			return false
		}
	}
	return true
}
