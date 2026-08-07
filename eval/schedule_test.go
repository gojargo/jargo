package eval

import (
	"testing"
	"time"
)

// newScheduleSession builds a session with only what waitSendAfter reads.
func newScheduleSession() *session {
	return &session{seenAt: make(map[string]time.Time)}
}

// TestWaitSendAfterAnchorsOnAnEarlierSighting checks the delay is measured from
// when the event was seen, not from when the turn asked. An event seen longer
// ago than the delay means the schedule has already fired.
func TestWaitSendAfterAnchorsOnAnEarlierSighting(t *testing.T) {
	s := newScheduleSession()
	s.seenAt[EventLLMStarted] = time.Now().Add(-time.Second)

	start := time.Now()
	if err := s.waitSendAfter(t.Context(), SendAfter{Event: EventLLMStarted, DelayMS: 200}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("the delay had already elapsed, should not have waited: took %s", elapsed)
	}
}

// TestWaitSendAfterWaitsForAnUnseenEvent checks the schedule polls until the
// event it anchors on arrives, and only then waits out its delay.
func TestWaitSendAfterWaitsForAnUnseenEvent(t *testing.T) {
	s := newScheduleSession()
	const arrivesAfter = 200 * time.Millisecond

	go func() {
		time.Sleep(arrivesAfter)
		s.seenMu.Lock()
		s.seenAt[EventLLMStarted] = time.Now()
		s.seenMu.Unlock()
	}()

	start := time.Now()
	if err := s.waitSendAfter(t.Context(), SendAfter{Event: EventLLMStarted, DelayMS: 100}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < arrivesAfter+100*time.Millisecond {
		t.Fatalf("should have waited for the event and then its delay: took %s", elapsed)
	}
}

// TestWaitSendAfterPureDelay checks a schedule with no event waits out its delay
// from now, the previous turn's send.
func TestWaitSendAfterPureDelay(t *testing.T) {
	s := newScheduleSession()

	start := time.Now()
	if err := s.waitSendAfter(t.Context(), SendAfter{DelayMS: 150}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 150*time.Millisecond {
		t.Fatalf("should have waited out the delay: took %s", elapsed)
	}
}
