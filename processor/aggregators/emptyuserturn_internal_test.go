package aggregators

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Empty user turn recovery is off with a realtime service, which hears the
// user's audio directly.

func TestEmptyTurnRecoveryIsOffInExplicitRealtimeMode(t *testing.T) {
	pair := New(frames.NewLLMContext(""), WithRealtimeServiceMode(true))
	if pair.User().emptyUserTurn != nil {
		t.Fatal("empty user turn recovery is on in realtime mode")
	}
}

func TestEmptyTurnRecoveryIsOffWhenARealtimeServiceAnnouncesItself(t *testing.T) {
	pair := New(frames.NewLLMContext(""))
	f := frames.NewLLMServiceMetadataFrame("FakeRealtimeLLM")
	f.Realtime = true
	pair.User().handleLLMServiceMetadata(f)
	if pair.User().emptyUserTurn != nil {
		t.Fatal("empty user turn recovery is on after a realtime service announced itself")
	}
}

func TestEmptyTurnRecoveryIsKeptForANonRealtimeService(t *testing.T) {
	pair := New(frames.NewLLMContext(""))
	pair.User().handleLLMServiceMetadata(frames.NewLLMServiceMetadataFrame("FakeLLM"))
	if pair.User().emptyUserTurn == nil {
		t.Fatal("empty user turn recovery is off for a service that is not realtime")
	}
}
