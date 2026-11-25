package realtime

import (
	"context"
	"fmt"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Stops of the bot's speech must keep the interruption reports right for the
// audio that follows them.

// observeStops counts the stops of the bot's speech the service has handled.
func observeStops(s *Service) *atomic.Int64 {
	var stops atomic.Int64
	event.On(s.Events(), processor.EventAfterProcessFrame, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.BotStoppedSpeakingFrame); ok {
			stops.Add(1)
		}
	})
	return &stops
}

// lastTruncation is the last playback position reported.
func (h *harness) lastTruncation() object {
	h.t.Helper()
	until(h.t, "a truncation", func() bool { return len(h.peer.sent(kindItemTruncate)) > 0 })
	sent := h.peer.sent(kindItemTruncate)
	return sent[len(sent)-1]
}

func TestStopOfAnEarlierOutputKeepsTheNextOneTruncatable(t *testing.T) {
	for _, generationEnded := range []bool{false, true} {
		t.Run(fmt.Sprint(generationEnded), func(t *testing.T) {
			var stops *atomic.Int64
			h := converse(t, harnessOptions{setup: func(s *Service) { stops = observeStops(s) }})
			base := stops.Load()
			writing, release := h.output.gateWrites(t, 2)
			first := h.peer.start(nil, nil)
			h.peer.message(first, messageOptions{text: "First output", samples: 480})
			h.peer.complete(first)
			waitFor(t, "the first write", writing[0])

			second := h.peer.start(nil, nil)
			h.peer.message(second, messageOptions{text: "Second output", samples: 2400, open: !generationEnded})
			if generationEnded {
				h.peer.complete(second)
			}
			until(t, "both outputs playing", func() bool {
				return slices.Equal(h.playingResponses(), []string{first, second})
			})
			release[0]()
			waitFor(t, "the second write", writing[1])
			until(t, "one stop", func() bool { return stops.Load() == base+1 })
			if got := h.playingResponses(); !slices.Equal(got, []string{second}) {
				t.Errorf("playing = %v, want %v", got, second)
			}

			h.queue(frames.NewInterruptionFrame())
			report := h.lastTruncation()
			if report["item_id"] != "message-"+second {
				t.Errorf("truncated %v, want the second output", report["item_id"])
			}
			if position := number(report["audio_end_ms"]); position < 0 || position > 100 {
				t.Errorf("position = %v", position)
			}
			until(t, "the requests to settle", func() bool { return h.requestsPending() == 0 })
			if !h.usable() || h.failed() {
				t.Errorf("usable = %v, errors = %v", h.usable(), h.errors())
			}
		})
	}
}

func TestQueuedOutputPlaybackIsTimedFromTheEarlierStop(t *testing.T) {
	var stops *atomic.Int64
	h := converse(t, harnessOptions{setup: func(s *Service) { stops = observeStops(s) }})
	base := stops.Load()
	writing, release := h.output.gateWrites(t, 2)
	first := h.peer.start(nil, nil)
	h.peer.message(first, messageOptions{text: "Filler", samples: 480})
	h.peer.complete(first)
	waitFor(t, "the first write", writing[0])

	// One second of audio, generated while the filler still plays.
	second := h.peer.start(nil, nil)
	h.peer.message(second, messageOptions{text: "Answer", samples: 24000})
	h.peer.complete(second)
	until(t, "both outputs playing", func() bool { return slices.Equal(h.playingResponses(), []string{first, second}) })
	time.Sleep(400 * time.Millisecond)
	release[0]()
	waitFor(t, "the second write", writing[1])
	until(t, "one stop", func() bool { return stops.Load() == base+1 })

	h.queue(frames.NewInterruptionFrame())
	report := h.lastTruncation()
	if report["item_id"] != "message-"+second {
		t.Errorf("truncated %v, want the second output", report["item_id"])
	}
	if position := number(report["audio_end_ms"]); position >= 300 {
		t.Errorf("position = %v, want it timed from the stop", position)
	}
	if h.failed() {
		t.Errorf("errors = %v", h.errors())
	}
}

func TestInterruptionDuringAnEarlierOutputTruncatesIt(t *testing.T) {
	h := converse(t, harnessOptions{})
	writing, _ := h.output.gateWrites(t, 1)
	first := h.peer.start(nil, nil)
	h.peer.message(first, messageOptions{text: "Filler", samples: 2400})
	h.peer.complete(first)
	waitFor(t, "the first write", writing[0])
	second := h.peer.start(nil, nil)
	h.peer.message(second, messageOptions{text: "Answer", samples: 2400})
	h.peer.complete(second)
	until(t, "both outputs playing", func() bool { return slices.Equal(h.playingResponses(), []string{first, second}) })

	h.queue(frames.NewInterruptionFrame())
	if report := h.lastTruncation(); report["item_id"] != "message-"+first {
		t.Errorf("truncated %v, want the first output", report["item_id"])
	}
	if got := h.playingResponses(); len(got) != 0 {
		t.Errorf("playing = %v, want nothing", got)
	}
	if h.failed() {
		t.Errorf("errors = %v", h.errors())
	}
}

func TestCompletedOutputIsNotTruncatedByALaterInterruption(t *testing.T) {
	h := converse(t, harnessOptions{})
	ref := h.peer.start(nil, nil)
	h.peer.say(ref, "Hello")
	h.peer.complete(ref)
	until(t, "audio written", func() bool { return h.output.written() > 0 })
	until(t, "the output to finish", func() bool { return len(h.playingResponses()) == 0 })
	h.queue(frames.NewInterruptionFrame())
	until(t, "the interruption handled", func() bool {
		h.svc.mu.Lock()
		defer h.svc.mu.Unlock()
		return h.svc.interruptEpoch == 1
	})
	until(t, "the interruption's requests", func() bool { return h.requestsPending() == 0 })
	if len(h.peer.sent(kindItemTruncate)) > 0 || len(h.peer.sent(kindResponseStop)) > 0 {
		t.Error("a completed output was truncated or canceled")
	}
	if h.failed() {
		t.Errorf("errors = %v", h.errors())
	}
}

func TestPlaybackStopWaitsForTheTransportWrite(t *testing.T) {
	var stops *atomic.Int64
	h := converse(t, harnessOptions{setup: func(s *Service) { stops = observeStops(s) }})
	base := stops.Load()
	writing, release := h.output.gateWrites(t, 1)
	ref := h.peer.start(nil, nil)
	h.peer.say(ref, "Hello")
	h.peer.complete(ref)
	waitFor(t, "the write", writing[0])
	until(t, "the reply's end", func() bool {
		value, ok := h.responseOf(ref)
		h.svc.mu.Lock()
		defer h.svc.mu.Unlock()
		return ok && value.ended
	})
	if stops.Load() != base {
		t.Error("the stop came before the write")
	}
	if got := h.playingResponses(); !slices.Equal(got, []string{ref}) {
		t.Errorf("playing = %v", got)
	}
	release[0]()
	until(t, "the stop", func() bool { return stops.Load() == base+1 })
	if got := h.playingResponses(); len(got) != 0 {
		t.Errorf("playing = %v, want nothing", got)
	}
}
