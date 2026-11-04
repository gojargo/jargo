package audio_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio"
)

// segmentRate is a sample rate where one second of 16-bit mono audio is 20
// bytes.
const segmentRate = 10

// seconds is secs of audio whose every byte is value.
func seconds(value byte, secs float64) []byte {
	return bytes.Repeat([]byte{value}, int(segmentRate*secs*2))
}

func newSegmentBuffer(preSpeech, maxDuration time.Duration) *audio.SpeechSegmentBuffer {
	b := audio.NewSpeechSegmentBuffer(preSpeech, maxDuration)
	b.SetSampleRate(segmentRate)
	return b
}

func join(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

func TestASegmentStartsWithTheLastPreSpeechBeforeIt(t *testing.T) {
	b := newSegmentBuffer(time.Second, 0)

	b.Append(seconds(1, 2))
	b.Append(seconds(2, 0.5))
	b.Start()
	b.Append(seconds(3, 3))

	if got, want := b.Stop(), join(seconds(1, 0.5), seconds(2, 0.5), seconds(3, 3)); !bytes.Equal(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

func TestStoppingEmptiesTheBufferForTheNextSegment(t *testing.T) {
	b := newSegmentBuffer(audio.DefaultPreSpeech, 0)

	b.Start()
	b.Append(seconds(1, 1))
	b.Stop()
	b.Append(seconds(2, 0.5))
	b.Start()
	b.Append(seconds(3, 1))

	if got, want := b.Stop(), join(seconds(2, 0.5), seconds(3, 1)); !bytes.Equal(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

func TestASegmentKeepsItsLastMax(t *testing.T) {
	b := newSegmentBuffer(time.Second, 2*time.Second)

	b.Append(seconds(1, 1))
	b.Start()
	b.Append(seconds(2, 3))

	if got, want := b.Stop(), seconds(2, 2); !bytes.Equal(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}

func TestASegmentWithoutMaxKeepsEverything(t *testing.T) {
	b := newSegmentBuffer(500*time.Millisecond, 0)

	b.Append(seconds(1, 1))
	b.Start()
	b.Append(seconds(2, 60))

	if got, want := b.Stop(), join(seconds(1, 0.5), seconds(2, 60)); !bytes.Equal(got, want) {
		t.Errorf("segment = %v, want %v", got, want)
	}
}
