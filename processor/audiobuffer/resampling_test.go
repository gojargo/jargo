package audiobuffer_test

import (
	"context"
	"encoding/binary"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/audiobuffer"
)

// tracks are the user and bot tracks a recording delivered.
type tracks struct {
	mu        sync.Mutex
	user, bot []byte
}

// recordAt24k records, at 16 kHz, audio sent at 24 kHz, so every chunk goes
// through the resamplers.
func recordAt24k(t *testing.T) (*tracks, *pipeline.Worker, chan error) {
	t.Helper()
	got := &tracks{}
	p := audiobuffer.New(audiobuffer.Config{
		SampleRate:  16000,
		NumChannels: 2,
		OnTrackAudioData: func(u, b []byte, _, _ int) {
			got.mu.Lock()
			got.user, got.bot = u, b
			got.mu.Unlock()
		},
	})
	w := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	time.Sleep(100 * time.Millisecond)
	p.StartRecording()
	return got, w, done
}

// constant is n samples of one value.
func constant(sample int16, n int) []byte {
	b := make([]byte, 2*n)
	for i := range n {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(sample))
	}
	return b
}

// queuePaced queues a frame and lets it through before the next: a system
// frame queued with data frames would otherwise jump ahead of them, where in a
// call it arrives in its own time.
func queuePaced(w *pipeline.Worker, f frames.Frame) {
	w.QueueFrame(f)
	time.Sleep(5 * time.Millisecond)
}

func sendUserAudio(w *pipeline.Worker, sample int16, ms int) {
	for range ms / 20 {
		queuePaced(w, frames.NewInputAudioRawFrame(constant(sample, 480), 24000, 1))
	}
}

func sendBotReply(w *pipeline.Worker, sample int16, ms int) {
	queuePaced(w, frames.NewBotStartedSpeakingFrame())
	for range ms / 20 {
		queuePaced(w, frames.NewOutputAudioRawFrame(constant(sample, 480), 24000, 1))
	}
	queuePaced(w, frames.NewBotStoppedSpeakingFrame())
}

func finishRecording(t *testing.T, w *pipeline.Worker, done chan error) {
	t.Helper()
	time.Sleep(200 * time.Millisecond)
	w.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not end")
	}
}

// nonZeroAfter counts the non-zero bytes of b from off on.
func nonZeroAfter(b []byte, off int) int {
	n := 0
	for _, v := range b[min(off, len(b)):] {
		if v != 0 {
			n++
		}
	}
	return n
}

// Ported from upstream
// (TestResamplingAcrossPauses::test_bot_reply_doesnt_start_with_the_end_of_the_previous_one).
// No bot audio arrives between replies, so the second reply, silent, starts
// clean rather than with the tail of the first, loud one.
func TestBotReplyDoesNotStartWithTheEndOfThePreviousOne(t *testing.T) {
	got, w, done := recordAt24k(t)
	sendBotReply(w, 1000, 100)
	sendUserAudio(w, 0, 200)
	sendBotReply(w, 0, 100)
	finishRecording(t, w, done)

	got.mu.Lock()
	defer got.mu.Unlock()
	// 100 ms at 16 kHz is 3200 bytes: everything after it is the silent reply.
	if len(got.bot) <= 3200 {
		t.Fatalf("bot track is %d bytes, want more than the first reply", len(got.bot))
	}
	if n := nonZeroAfter(got.bot, 3200); n != 0 {
		t.Errorf("the bot track after the first reply holds %d non-zero bytes, want none", n)
	}
}

// Ported from upstream
// (TestResamplingAcrossPauses::test_unmuted_user_audio_doesnt_start_with_audio_from_before_the_mute).
func TestUnmutedUserAudioDoesNotStartWithAudioFromBeforeTheMute(t *testing.T) {
	got, w, done := recordAt24k(t)
	sendUserAudio(w, 1000, 100)
	queuePaced(w, frames.NewUserMuteStartedFrame())
	sendBotReply(w, 0, 200)
	queuePaced(w, frames.NewUserMuteStoppedFrame())
	sendUserAudio(w, 0, 100)
	finishRecording(t, w, done)

	got.mu.Lock()
	defer got.mu.Unlock()
	if len(got.user) <= 3200 {
		t.Fatalf("user track is %d bytes, want more than the first 100 ms", len(got.user))
	}
	if n := nonZeroAfter(got.user, 3200); n != 0 {
		t.Errorf("the user track after the mute holds %d non-zero bytes, want none", n)
	}
}
