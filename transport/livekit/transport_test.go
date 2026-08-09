package livekit

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/transport"
)

// TestChannels checks an unset channel count means mono. Zero channels is not a
// stream anything can be decoded from, so it stands for "not configured".
func TestChannels(t *testing.T) {
	if got := channels(0); got != 1 {
		t.Errorf("channels(0) = %d, want mono", got)
	}
	for _, n := range []int{1, 2} {
		if got := channels(n); got != n {
			t.Errorf("channels(%d) = %d, want it left alone", n, got)
		}
	}
}

// TestNewTransportProcessors checks the transport hands out the two processors a
// pipeline puts at its ends, each under the label that identifies it in logs,
// metrics and traces.
func TestNewTransportProcessors(t *testing.T) {
	tr := NewTransport(&Connection{}, transport.Params{})

	in := tr.Input()
	if in == nil {
		t.Fatal("Input() = nil, want the input processor")
	}
	if got := in.Name(); !strings.HasPrefix(got, "LiveKitInput#") {
		t.Errorf("Input().Name() = %q, want the LiveKitInput label", got)
	}

	out := tr.Output()
	if out == nil {
		t.Fatal("Output() = nil, want the output processor")
	}
	if got := out.Name(); !strings.HasPrefix(got, "LiveKitOutput#") {
		t.Errorf("Output().Name() = %q, want the LiveKitOutput label", got)
	}
}

// TestStartAndStopReading checks the read goroutine starts and is waited for on
// the way out. StopReading returning is what says the goroutine is gone, so a
// session that ends cannot leave one decoding into a closed pipeline.
func TestStartAndStopReading(t *testing.T) {
	in := newInput(&Connection{}, transport.Params{})

	if err := in.StartReading(t.Context()); err != nil {
		t.Fatalf("StartReading: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- in.StopReading(t.Context()) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("StopReading: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("StopReading did not return, so the read goroutine is still running")
	}
}

// TestRemoteAudioTrackWaitsForTheContext checks a caller waiting for a track
// that never arrives is released when its context ends rather than blocking for
// the life of the session.
func TestRemoteAudioTrackWaitsForTheContext(t *testing.T) {
	c := &Connection{closed: make(chan struct{})}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if _, err := c.RemoteAudioTrack(ctx); err == nil {
		t.Fatal("RemoteAudioTrack on a canceled context = nil, want an error")
	}
}

// TestRemoteAudioTrackStopsWhenTheRoomGoes checks a caller waiting for a track
// is released when the room disconnects, with the reason it was released.
func TestRemoteAudioTrackStopsWhenTheRoomGoes(t *testing.T) {
	c := &Connection{closed: make(chan struct{})}
	close(c.closed)

	_, err := c.RemoteAudioTrack(t.Context())
	if err == nil {
		t.Fatal("RemoteAudioTrack on a closed room = nil, want an error")
	}
	if !strings.Contains(err.Error(), "closed") {
		t.Errorf("error = %v, want it to say the connection closed", err)
	}
}

// TestDone reports the channel that closes with the room, which is what a
// session watches to know the call ended.
func TestDone(t *testing.T) {
	c := &Connection{closed: make(chan struct{})}
	select {
	case <-c.Done():
		t.Fatal("Done() fired while the room was still up")
	default:
	}

	close(c.closed)
	select {
	case <-c.Done():
	case <-time.After(time.Second):
		t.Fatal("Done() did not fire once the room went")
	}
}

// TestOnMessageDeliversWhatArrivedFirst checks a data message that arrived
// before the input transport registered its handler is held rather than dropped.
// The room is joined before the pipeline runs, so the first messages routinely
// arrive with nobody listening.
func TestOnMessageDeliversWhatArrivedFirst(t *testing.T) {
	c := &Connection{}

	c.deliver([]byte("first"))
	c.deliver([]byte("second"))

	var got []string
	c.OnMessage(func(b []byte) { got = append(got, string(b)) })
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Fatalf("messages held before a handler = %v, want both in order", got)
	}

	c.deliver([]byte("third"))
	if len(got) != 3 || got[2] != "third" {
		t.Errorf("messages = %v, want the third delivered straight through", got)
	}
}

// TestPaceHoldsOutputAtRealTime checks the output is paced one frame at a time.
// LiveKit sends a sample the instant it is handed one, so an utterance written
// back-to-back would arrive far faster than it plays and flood the client's
// jitter buffer.
func TestPaceHoldsOutputAtRealTime(t *testing.T) {
	out := newOutput(&Connection{}, transport.Params{})
	ctx := t.Context()

	// The first send goes immediately: there is nothing to be behind.
	start := time.Now()
	out.pace(ctx)
	if elapsed := time.Since(start); elapsed > opus.FrameDuration {
		t.Errorf("the first frame waited %v, want it sent straight away", elapsed)
	}

	// The next three are held to the frame duration each.
	start = time.Now()
	for range 3 {
		out.pace(ctx)
	}
	if elapsed := time.Since(start); elapsed < 2*opus.FrameDuration {
		t.Errorf("three frames took %v, want them paced at %v each", elapsed, opus.FrameDuration)
	}
}

// TestPaceResetsAfterAGap checks a gap longer than one frame restarts the clock
// rather than sending a burst to catch up. The gap is the bot not talking, and
// what follows is the start of a new utterance, not a late part of the last one.
func TestPaceResetsAfterAGap(t *testing.T) {
	out := newOutput(&Connection{}, transport.Params{})
	ctx := t.Context()

	out.pace(ctx)
	// Stand where a long silence would have left the clock.
	out.nextSend = time.Now().Add(-time.Second)

	start := time.Now()
	out.pace(ctx)
	if elapsed := time.Since(start); elapsed > opus.FrameDuration {
		t.Errorf("the frame after a gap waited %v, want it sent straight away", elapsed)
	}
}

// TestPaceStopsOnCancel checks a paced send gives up when the session ends, so a
// pipeline tearing down is not held for the frame it was waiting to place.
func TestPaceStopsOnCancel(t *testing.T) {
	out := newOutput(&Connection{}, transport.Params{})
	out.pace(t.Context())

	ctx, cancel := context.WithCancel(t.Context())
	var wg sync.WaitGroup
	wg.Go(func() {
		time.Sleep(2 * time.Millisecond)
		cancel()
	})

	start := time.Now()
	out.pace(ctx)
	wg.Wait()
	if elapsed := time.Since(start); elapsed >= opus.FrameDuration {
		t.Errorf("the paced send waited %v, want it cut short by the cancel", elapsed)
	}
}
