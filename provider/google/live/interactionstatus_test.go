package live

import (
	"context"
	"reflect"
	"slices"
	"testing"
	"time"
)

// The thinking models reason in the background between chunks of output, so a
// completed generation does not mean the turn is over. They say which it is with
// an interaction status, and the turn is held open until they report themselves
// finished. A reply split across several assistant turns is what this prevents.

// speakingThen drives a bot turn and then the given messages, returning the
// frames the service emitted.
func speakingThen(t *testing.T, messages ...string) []string {
	t.Helper()
	s, sink := newTestService(t)
	ctx := context.Background()
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2})))
	for _, m := range messages {
		s.handle(ctx, decode(t, m))
	}
	return sink.names()
}

// speaking is the frames of a bot turn that has begun and not ended.
//
//nolint:gochecknoglobals // fixed expectation
var speaking = []string{"BotStartedSpeakingFrame", "TTSAudioRawFrame"}

// stopped is speaking with the turn closed behind it.
//
//nolint:gochecknoglobals // fixed expectation
var stopped = []string{"BotStartedSpeakingFrame", "TTSAudioRawFrame", "BotStoppedSpeakingFrame"}

// A generation that completes while the model is still working closes a chunk of
// the reply, not the turn.
func TestAGenerationCompletedWhileWorkingHoldsTheTurnOpen(t *testing.T) {
	got := speakingThen(t, `{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`)
	if !reflect.DeepEqual(got, speaking) {
		t.Errorf("frames = %v, want the turn still open (%v)", got, speaking)
	}
}

// Either spelling of the finished state releases the turn that was held.
func TestReportingItselfFinishedClosesAHeldTurn(t *testing.T) {
	for _, status := range []string{"IDLE", "REQUIRES_ACTION"} {
		got := speakingThen(t,
			`{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`,
			`{"serverContent":{"interactionStatus":"`+status+`"}}`)
		if !reflect.DeepEqual(got, stopped) {
			t.Errorf("%s: frames = %v, want the turn closed (%v)", status, got, stopped)
		}
	}
}

// A generation that completes with the model already finished needs no holding.
func TestAGenerationCompletedWhileIdleClosesTheTurnAtOnce(t *testing.T) {
	got := speakingThen(t, `{"serverContent":{"generationComplete":true,"interactionStatus":"IDLE"}}`)
	if !reflect.DeepEqual(got, stopped) {
		t.Errorf("frames = %v, want the turn closed at once (%v)", got, stopped)
	}
}

// The models that report no status keep the behavior they had, which is the
// only behavior there was.
func TestAModelThatReportsNoStatusClosesTheTurnAsBefore(t *testing.T) {
	got := speakingThen(t, `{"serverContent":{"generationComplete":true}}`)
	if !reflect.DeepEqual(got, stopped) {
		t.Errorf("frames = %v, want the turn closed (%v)", got, stopped)
	}
}

// A status this does not know says nothing, so it cannot be read as the model
// still working.
func TestAnUnknownStatusDoesNotHoldTheTurn(t *testing.T) {
	got := speakingThen(t,
		`{"serverContent":{"generationComplete":true,"interactionStatus":"INTERACTION_STATUS_UNSPECIFIED"}}`)
	if !reflect.DeepEqual(got, stopped) {
		t.Errorf("frames = %v, want the turn closed (%v)", got, stopped)
	}
}

// Reporting itself finished with no turn held closes nothing: there is no turn
// to close, and inventing one would tell everything downstream the bot stopped
// speaking twice.
func TestReportingItselfFinishedWithNoHeldTurnClosesNothing(t *testing.T) {
	got := speakingThen(t,
		`{"serverContent":{"generationComplete":true,"interactionStatus":"IDLE"}}`,
		`{"serverContent":{"interactionStatus":"IDLE"}}`)
	if !reflect.DeepEqual(got, stopped) {
		t.Errorf("frames = %v, want the turn closed once (%v)", got, stopped)
	}
}

// A generation that ends the turn stands in for one still held, and the turn is
// closed once between them.
func TestAFinishedGenerationSupersedesAHeldTurn(t *testing.T) {
	got := speakingThen(t,
		`{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`,
		`{"serverContent":{"generationComplete":true,"interactionStatus":"REQUIRES_ACTION"}}`)
	if !reflect.DeepEqual(got, stopped) {
		t.Errorf("frames = %v, want the turn closed exactly once (%v)", got, stopped)
	}
}

// An interruption has ended the turn already, so a held one is moot: releasing
// it afterwards would report the bot stopping a second time.
func TestAnInterruptionDropsAHeldTurn(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2})))
	s.handle(ctx, decode(t, `{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`))
	s.handle(ctx, decode(t, `{"serverContent":{"interrupted":true}}`))
	before := len(sink.names())

	s.handle(ctx, decode(t, `{"serverContent":{"interactionStatus":"IDLE"}}`))

	if got := sink.names(); len(got) != before {
		t.Errorf("frames = %v, want nothing released: the interruption ended the turn", got)
	}
}

// A session that goes quiet without ever reporting itself finished would leave
// everything downstream waiting on a turn that never ends, so the turn is closed
// anyway. The watch measures silence, so a reply still streaming keeps it at bay.
func TestAQuietSessionEndsAHeldTurnAnyway(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2})))
	s.handle(ctx, decode(t, `{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`))

	// Bring the watch forward rather than waiting out its own timeout.
	s.mu.Lock()
	watchdog := s.heldTurnWatchdog
	s.mu.Unlock()
	if watchdog == nil {
		t.Fatal("a held turn is not being watched, so a quiet session would never end it")
	}
	if !watchdog.Reset(time.Millisecond) {
		t.Fatal("the watch had already fired")
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if reflect.DeepEqual(sink.names(), stopped) {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Errorf("frames = %v, want the held turn ended (%v)", sink.names(), stopped)
}

// The watch is on silence, so a server still speaking keeps the turn held.
func TestAStreamingReplyKeepsAHeldTurnOpen(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2})))
	s.handle(ctx, decode(t, `{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`))

	s.mu.Lock()
	first := s.heldTurnWatchdog
	s.mu.Unlock()

	s.handle(ctx, decode(t, audioMessage([]byte{3, 4})))

	s.mu.Lock()
	second := s.heldTurnWatchdog
	s.mu.Unlock()
	if second == nil || second == first {
		t.Error("a message arriving did not restart the watch, so a long reply would be cut short")
	}
	if got := sink.names(); slices.Contains(got, "BotStoppedSpeakingFrame") {
		t.Errorf("frames = %v, want the turn still held", got)
	}
}

// A held turn is not left with a watch running on it once the session is gone,
// since nothing is going to report that session finished now.
func TestDisconnectDropsAHeldTurn(t *testing.T) {
	s, _ := newTestService(t)
	ctx := context.Background()
	s.handle(ctx, decode(t, audioMessage([]byte{1, 2})))
	s.handle(ctx, decode(t, `{"serverContent":{"generationComplete":true,"interactionStatus":"IN_PROGRESS"}}`))

	s.disconnect()

	s.mu.Lock()
	held, watchdog := s.turnHeld, s.heldTurnWatchdog
	s.mu.Unlock()
	if held || watchdog != nil {
		t.Errorf("turnHeld = %v, watchdog = %v, want the held turn dropped", held, watchdog)
	}
}
