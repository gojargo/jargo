package rtvi_test

import (
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// Tests for the audio levels a client draws a speaking meter from. They are off
// by default, since they are a message every period for as long as the call
// lasts, and they are measured over a rolling window rather than per frame.

// levelAudio is one chunk of a 440 Hz tone, long enough that four of them fill
// the 400 ms window loudness is measured over.
func levelAudio(ms, rate int, amp float64) []byte {
	n := ms * rate / 1000
	pcm := make([]byte, n*2)
	for i := range n {
		s := int16(amp * math.Sin(2*math.Pi*440*float64(i)/float64(rate)))
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(s))
	}
	return pcm
}

// userAudio and botAudio are the frames each side's level is measured from.
func userAudio(pcm []byte, rate int) frames.Frame {
	return frames.NewInputAudioRawFrame(pcm, rate, 1)
}

func botAudio(pcm []byte, rate int) frames.Frame {
	return frames.NewTTSAudioRawFrame(pcm, rate, 1)
}

// levelOf returns the reported level of the first message of msgType.
func levelOf(t *testing.T, msgs []rtvi.Message, msgType string) (float64, bool) {
	t.Helper()
	for _, m := range msgs {
		if m.Type != msgType {
			continue
		}
		d, ok := m.Data.(rtvi.AudioLevelData)
		if !ok {
			t.Fatalf("message %s carries %T, want AudioLevelData", msgType, m.Data)
		}
		return d.Value, true
	}
	return 0, false
}

// TestAudioLevelsAreOffByDefault checks a client that never asked for levels is
// not sent one per period for the length of the call.
func TestAudioLevelsAreOffByDefault(t *testing.T) {
	const rate = 16000
	chunk := levelAudio(100, rate, 8000)

	msgs := observerHarness(t, rtvi.DefaultObserverParams(),
		userAudio(chunk, rate), botAudio(chunk, rate))

	if _, ok := levelOf(t, msgs, rtvi.TypeUserAudioLevel); ok {
		t.Error("a user audio level was sent with the levels off")
	}
	if _, ok := levelOf(t, msgs, rtvi.TypeBotAudioLevel); ok {
		t.Error("a bot audio level was sent with the levels off")
	}
}

// TestUserAudioLevelReported checks the user's level is measured over the
// rolling window and reported once enabled. The window is 400 ms, so the first
// report of a fresh stream reads zero until enough audio has arrived to measure.
func TestUserAudioLevelReported(t *testing.T) {
	const rate = 16000
	params := rtvi.DefaultObserverParams()
	params.UserAudioLevelEnabled = true
	// Report every frame, so the levels follow the window filling.
	params.AudioLevelPeriod = time.Nanosecond

	// Five 100 ms chunks: the window fills on the fourth.
	chunk := levelAudio(100, rate, 8000)
	queue := make([]frames.Frame, 0, 5)
	for range 5 {
		queue = append(queue, userAudio(chunk, rate))
	}

	msgs := observerHarness(t, params, queue...)

	var levels []float64
	for _, m := range msgs {
		if m.Type == rtvi.TypeUserAudioLevel {
			d, ok := m.Data.(rtvi.AudioLevelData)
			if !ok {
				t.Fatalf("the level message carries %T, want AudioLevelData", m.Data)
			}
			levels = append(levels, d.Value)
		}
	}

	if len(levels) != 5 {
		t.Fatalf("got %d level messages, want one per frame: %v", len(levels), levels)
	}
	for i, l := range levels[:3] {
		if l != 0 {
			t.Errorf("level %d = %v, want 0 while the window is still filling", i, l)
		}
	}
	if levels[3] <= 0 {
		t.Errorf("level 3 = %v, want a positive level once the window is full", levels[3])
	}
	if levels[3] > 1 {
		t.Errorf("level 3 = %v, want a level on the 0..1 scale", levels[3])
	}
}

// TestBotAudioLevelReported checks the bot's level comes off its synthesized
// audio, and that the two sides are measured separately.
func TestBotAudioLevelReported(t *testing.T) {
	const rate = 16000
	params := rtvi.DefaultObserverParams()
	params.BotAudioLevelEnabled = true
	params.AudioLevelPeriod = time.Nanosecond

	// A full window in one chunk, so the level is measurable straight away.
	full := levelAudio(400, rate, 8000)
	msgs := observerHarness(t, params, botAudio(full, rate), botAudio(full, rate))

	level, ok := levelOf(t, msgs, rtvi.TypeBotAudioLevel)
	if !ok {
		t.Fatal("no bot audio level was reported")
	}
	if level < 0 || level > 1 {
		t.Errorf("level = %v, want it on the 0..1 scale", level)
	}

	// The user's side was never fed, so it reports nothing.
	if _, ok := levelOf(t, msgs, rtvi.TypeUserAudioLevel); ok {
		t.Error("a user level was reported from the bot's audio")
	}
}

// TestAudioLevelPeriodRateLimits checks the level is reported on a period rather
// than per frame: audio arrives far more often than a meter can be redrawn.
func TestAudioLevelPeriodRateLimits(t *testing.T) {
	const rate = 16000
	params := rtvi.DefaultObserverParams()
	params.UserAudioLevelEnabled = true
	params.AudioLevelPeriod = time.Hour // nothing after the first is ever due

	chunk := levelAudio(100, rate, 8000)
	queue := make([]frames.Frame, 0, 6)
	for range 6 {
		queue = append(queue, userAudio(chunk, rate))
	}

	msgs := observerHarness(t, params, queue...)

	var n int
	for _, m := range msgs {
		if m.Type == rtvi.TypeUserAudioLevel {
			n++
		}
	}
	if n != 1 {
		t.Errorf("got %d level messages from 6 frames, want the period to allow 1", n)
	}
}
