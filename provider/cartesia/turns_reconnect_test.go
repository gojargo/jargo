package cartesia_test

// How the turns STT survives its connection dropping, driven end to end through
// a pipeline against a fake Ink-2 server. A connection that is never confirmed
// is covered by turns_confirm_test.go, which can shorten the confirmation wait.

import (
	"reflect"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/service/stt"
	errs "github.com/gojargo/jargo/utils/errors"
)

func turnsService(url string) *stt.StreamService {
	return cartesia.NewTurnsSTT(cartesia.TurnsSTTConfig{APIKey: "test-key", URL: url, SampleRate: 16000})
}

// TestTurnsAFailedFirstConnectIsRetriedInTheBackground checks a connection
// refused at start is retried in the background while the audio keeps flowing.
func TestTurnsAFailedFirstConnectIsRetriedInTheBackground(t *testing.T) {
	server := &fakeInk2{rejectedHandshakes: 1}
	svc := turnsService(server.start(t))

	run(t, svc, 500*time.Millisecond)

	handshakes, audio := server.counts()
	if handshakes != 2 {
		t.Errorf("handshakes = %d, want 2", handshakes)
	}
	if len(audio) == 0 || audio[0] == 0 {
		t.Errorf("audio per connection = %v, want the first accepted connection to get audio", audio)
	}
	if !svc.Usable() {
		t.Error("service unusable")
	}
}

// TestTurnsADroppedConnectionIsReplacedByOneNewConnection checks a dropped
// connection is replaced by exactly one new connection.
func TestTurnsADroppedConnectionIsReplacedByOneNewConnection(t *testing.T) {
	server := &fakeInk2{dropFirstAfter: 10}
	run(t, turnsService(server.start(t)), time.Second)

	_, audio := server.counts()
	if len(audio) != 2 || audio[1] == 0 {
		t.Errorf("audio per connection = %v, want two connections, the second receiving audio", audio)
	}
}

// TestTurnsATurnCutShortByADropEndsWithItsTranscriptSoFar checks a turn the
// connection drops in the middle of ends at the drop, with what was transcribed
// so far as its transcript, unfinalized, and a proposed stop.
func TestTurnsATurnCutShortByADropEndsWithItsTranscriptSoFar(t *testing.T) {
	server := &fakeInk2{dropFirstAfter: 10, dropMidTurn: true}
	res := run(t, turnsService(server.start(t)), time.Second)

	ts, stops := turnEnds(res.down)
	if want := []transcript{{"hello there", false}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 1 {
		t.Errorf("proposed stops = %d, want 1", len(stops))
	}
	if _, audio := server.counts(); len(audio) != 2 {
		t.Errorf("connections = %d, want 2", len(audio))
	}
}

// TestTurnsSpeechContinuingAfterADropBecomesANewTurn checks speech carrying on
// after a drop is a turn of its own on the new connection, with its own
// proposed start and stop.
func TestTurnsSpeechContinuingAfterADropBecomesANewTurn(t *testing.T) {
	server := &fakeInk2{dropFirstAfter: 10, dropMidTurn: true, resumeTurn: true}
	res := run(t, turnsService(server.start(t)), time.Second)

	ts, stops := turnEnds(res.down)
	if want := []transcript{{"hello there", false}, {"how are you", true}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 2 {
		t.Errorf("proposed stops = %d, want 2", len(stops))
	}
	starts := 0
	for _, f := range res.down {
		if _, ok := f.(*frames.ProposedUserStartedSpeakingFrame); ok {
			starts++
		}
	}
	if starts != 2 {
		t.Errorf("proposed starts = %d, want 2 (one per turn)", starts)
	}
}

// TestTurnsATurnEndsWhenReconnectingGivesUp checks a turn cut short ends at the
// drop even when no connection replaces it.
func TestTurnsATurnEndsWhenReconnectingGivesUp(t *testing.T) {
	server := &fakeInk2{dropFirstAfter: 10, dropMidTurn: true, rejectAfter: 1}
	res := run(t, turnsService(server.start(t)), time.Second)

	ts, stops := turnEnds(res.down)
	if want := []transcript{{"hello there", false}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 1 {
		t.Errorf("proposed stops = %d, want 1", len(stops))
	}
}

// TestTurnsStoppingMidTurnLetsTheServerEndTheTurn checks stopping in the middle
// of a turn lets the server end it, rather than ending it as cut short.
func TestTurnsStoppingMidTurnLetsTheServerEndTheTurn(t *testing.T) {
	server := &fakeInk2{openTurnAfter: 5}
	res := run(t, turnsService(server.start(t)), 400*time.Millisecond)

	ts, stops := turnEnds(res.down)
	if want := []transcript{{"hello there", true}}; !reflect.DeepEqual(ts, want) {
		t.Errorf("transcripts = %v, want %v", ts, want)
	}
	if len(stops) != 1 {
		t.Errorf("proposed stops = %d, want 1", len(stops))
	}
}

// TestTurnsRejectedSettingsAreNotRetried checks settings the server rejects are
// not retried: the service reports one invalid request and becomes unusable.
func TestTurnsRejectedSettingsAreNotRetried(t *testing.T) {
	server := &fakeInk2{rejectSettings: true}
	svc := turnsService(server.start(t))
	res := run(t, svc, 500*time.Millisecond)

	if handshakes, _ := server.counts(); handshakes != 1 {
		t.Errorf("handshakes = %d, want 1", handshakes)
	}
	if svc.Usable() {
		t.Error("service still usable")
	}
	efs := errorFrames(res.up)
	if len(efs) != 1 || efs[0].Category != errs.InvalidRequest {
		t.Errorf("errors = %v, want one INVALID_REQUEST", efs)
	}
}

// TestTurnsAudioKeepsFlowingWhileTheProviderIsDown checks audio keeps flowing
// down the pipeline while the provider is unreachable.
func TestTurnsAudioKeepsFlowingWhileTheProviderIsDown(t *testing.T) {
	server := &fakeInk2{dropFirstAfter: 5, rejectAfter: 1}
	res := run(t, turnsService(server.start(t)), 500*time.Millisecond)

	if res.elapsed >= 3*time.Second {
		t.Errorf("run took %s, want < 3s", res.elapsed)
	}
	n := 0
	for _, f := range res.down {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			n++
		}
	}
	if n != 25 {
		t.Errorf("audio frames downstream = %d, want 25", n)
	}
}

// TestTurnsStoppingDoesNotReconnect checks stopping the pipeline does not
// reconnect.
func TestTurnsStoppingDoesNotReconnect(t *testing.T) {
	server := &fakeInk2{}
	run(t, turnsService(server.start(t)), 300*time.Millisecond)
	time.Sleep(300 * time.Millisecond)

	if handshakes, _ := server.counts(); handshakes != 1 {
		t.Errorf("handshakes = %d, want 1", handshakes)
	}
}
