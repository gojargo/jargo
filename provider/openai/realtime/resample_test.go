package realtime

import (
	"bytes"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// sendOneSecond converts one second of audio at rate, in 20 ms frames the way
// an input transport delivers it, and returns how many bytes reach the session.
func sendOneSecond(s *LLMService, rate int) int {
	frameBytes := rate * 2 / 50
	sent := 0
	for range 50 {
		pcm := bytes.Repeat([]byte{0x01, 0x02}, frameBytes/2)
		sent += len(s.toSessionRate(frames.AudioRawData{Audio: pcm, SampleRate: rate, NumChannels: 1}))
	}
	return sent
}

// TestPipelineAudioIsResampledTo24kHz checks that 16 kHz pipeline audio reaches
// the input buffer as 24 kHz audio, which is the session's fixed input rate.
func TestPipelineAudioIsResampledTo24kHz(t *testing.T) {
	s := New(Config{APIKey: "k"})
	t.Cleanup(s.closeResampler)

	// One second at 24 kHz is 48000 bytes, less what the streaming resampler
	// still holds.
	if sent := sendOneSecond(s, 16000); sent < 46000 || sent > 48000 {
		t.Errorf("sent %d bytes, want one second at 24 kHz (46000 to 48000)", sent)
	}
}

// TestAudioAtTheSessionRateIsSentAsItIs checks that 24 kHz audio needs no
// conversion.
func TestAudioAtTheSessionRateIsSentAsItIs(t *testing.T) {
	s := New(Config{APIKey: "k"})
	pcm := bytes.Repeat([]byte{0x01, 0x02}, 480)

	got := s.toSessionRate(frames.AudioRawData{Audio: pcm, SampleRate: sampleRate, NumChannels: 1})

	if !bytes.Equal(got, pcm) {
		t.Errorf("24 kHz audio was changed: %d bytes in, %d out", len(pcm), len(got))
	}
}
