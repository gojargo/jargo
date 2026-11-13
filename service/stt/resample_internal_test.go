package stt

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
)

// Input audio reaches the provider at the rate it is sent audio at, resampled
// when it arrives at another one.

// tone is a 220 Hz sine of the given length, as 16-bit mono PCM at rate.
func tone(rate int, d time.Duration) []byte {
	n := int(float64(rate) * d.Seconds())
	pcm := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/float64(rate)) * 8000)
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(v))
	}
	return pcm
}

// resampled is pcm at inRate converted to outRate as the start of a fresh
// stream.
func resampled(t *testing.T, pcm []byte, inRate, outRate int) []byte {
	t.Helper()
	r, err := resample.New(inRate, outRate, 1)
	if err != nil {
		t.Fatalf("resampler: %v", err)
	}
	return r.Process(pcm)
}

// rateConnector hands out one recording stream, and sends audio at sendRate
// when it is set, recording the rate each session was opened for.
type rateConnector struct {
	stream   *kaStream
	sendRate int

	mu    sync.Mutex
	rates []int
}

func (c *rateConnector) Connect(ctx context.Context, sampleRate int) (Stream, error) {
	c.mu.Lock()
	c.rates = append(c.rates, sampleRate)
	c.mu.Unlock()
	c.stream.ctx = ctx
	return c.stream, nil
}

func (c *rateConnector) opened() []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]int(nil), c.rates...)
}

// sendRateConnector is a rateConnector whose provider takes audio at sendRate
// only.
type sendRateConnector struct{ rateConnector }

func (c *sendRateConnector) SendSampleRate(int) int { return c.sendRate }

// A provider is sent audio at the rate the service runs at, whatever rate the
// transport delivered it at.
func TestInputAudioAtAnotherRateIsResampledToTheServiceRate(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	svc := NewStream("FakeSTT", &rateConnector{stream: st}, 16000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	clip := tone(24000, 100*time.Millisecond)
	task.QueueFrame(frames.NewInputAudioRawFrame(clip, 24000, 1))
	waitFor(t, "the audio to be sent", func() bool { return len(st.sends()) == 1 })

	if got, want := st.sends()[0], resampled(t, clip, 24000, 16000); !bytes.Equal(got, want) {
		t.Errorf("sent %d bytes, want the clip resampled to 16 kHz (%d bytes)", len(got), len(want))
	}
}

// A provider that takes audio at only some rates is sent it at the one it asks
// for, and its session is opened for that rate.
func TestInputAudioIsResampledToTheRateTheProviderTakes(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	conn := &sendRateConnector{rateConnector{stream: st, sendRate: 8000}}
	svc := NewStream("FakeSTT", conn, 16000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	clip := tone(16000, 100*time.Millisecond)
	task.QueueFrame(frames.NewInputAudioRawFrame(clip, 16000, 1))
	waitFor(t, "the audio to be sent", func() bool { return len(st.sends()) == 1 })

	if got, want := st.sends()[0], resampled(t, clip, 16000, 8000); !bytes.Equal(got, want) {
		t.Errorf("sent %d bytes, want the clip resampled to 8 kHz (%d bytes)", len(got), len(want))
	}
	if got := conn.opened(); len(got) != 1 || got[0] != 8000 {
		t.Errorf("sessions opened for %v Hz, want one for the 8000 Hz the provider takes", got)
	}
}

// Usage measures the audio the provider received, which is at the rate it is
// sent audio at: a second of speech is a second, however many bytes it took to
// arrive.
func TestUsageIsCountedAtTheRateTheProviderTakes(t *testing.T) {
	t.Parallel()

	st := &kaStream{}
	svc := NewStream("FakeSTT", &sendRateConnector{rateConnector{stream: st, sendRate: 8000}}, 48000)
	task, stop := runKeepaliveService(t, svc)
	defer stop()
	waitFor(t, "the session to open", func() bool { return svc.Connected() })

	task.QueueFrame(frames.NewInputAudioRawFrame(tone(48000, time.Second), 48000, 1))
	waitFor(t, "the audio to be sent", func() bool { return len(st.sends()) == 1 })

	if got := len(st.sends()[0]); got != 8000*2 {
		t.Errorf("sent %d bytes, want one second at 8 kHz (16000 bytes)", got)
	}
	svc.mu.Lock()
	pending := pcmDuration(svc.audioBytes, svc.sendRate)
	svc.mu.Unlock()
	if pending != time.Second {
		t.Errorf("usage pending = %s, want the 1s submitted", pending)
	}
}
