package livekit

import (
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport"
)

func TestValidate(t *testing.T) {
	full := Config{URL: "ws://x", APIKey: "k", APISecret: "s", RoomName: "r", Identity: "i"}
	if err := full.Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}

	for name, cfg := range map[string]Config{
		"no url":      {APIKey: "k", APISecret: "s", RoomName: "r", Identity: "i"},
		"no key":      {URL: "ws://x", APISecret: "s", RoomName: "r", Identity: "i"},
		"no secret":   {URL: "ws://x", APIKey: "k", RoomName: "r", Identity: "i"},
		"no room":     {URL: "ws://x", APIKey: "k", APISecret: "s", Identity: "i"},
		"no identity": {URL: "ws://x", APIKey: "k", APISecret: "s", RoomName: "r"},
	} {
		if err := cfg.Validate(); err == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
}

// TestValidateWithToken covers joining with an access token minted elsewhere:
// the token names the room and the participant, so only the URL is needed.
func TestValidateWithToken(t *testing.T) {
	if err := (Config{URL: "ws://x", Token: "t"}).Validate(); err != nil {
		t.Fatalf("config with a token rejected: %v", err)
	}
	if err := (Config{Token: "t"}).Validate(); err == nil {
		t.Fatal("config with a token but no URL accepted")
	}
}

func TestConnectValidatesConfig(t *testing.T) {
	if _, err := Connect(Config{}); err == nil {
		t.Fatal("Connect with empty config should fail validation")
	}
}

// A key a SIP caller presses reaches the handler with what LiveKit reported
// about it, and the keys pressed before anything registered are not lost.
func TestDeliverDTMF(t *testing.T) {
	c := &Connection{}

	// Pressed before the input transport has registered its handler.
	c.deliverDTMF(DTMFEvent{Tone: "1", Digit: "1", Code: 1, ParticipantID: "sip-caller"})
	c.deliverDTMF(DTMFEvent{Tone: "#", Digit: "#", Code: 11})

	var got []DTMFEvent
	c.OnDTMF(func(e DTMFEvent) { got = append(got, e) })
	if len(got) != 2 || got[0].Tone != "1" || got[0].ParticipantID != "sip-caller" || got[1].Code != 11 {
		t.Fatalf("keys held before a handler = %+v, want 1 then #", got)
	}

	c.deliverDTMF(DTMFEvent{Tone: "7"})
	if len(got) != 3 || got[2].Tone != "7" {
		t.Errorf("keys = %+v, want 7 delivered straight through", got)
	}
}

// The keys a SIP leg carries map to their RFC 4733 event codes.
func TestDTMFCodes(t *testing.T) {
	want := map[frames.KeypadEntry]uint32{"0": 0, "5": 5, "9": 9, "*": 10, "#": 11}
	for key, code := range want {
		if got, ok := dtmfCodes[key]; !ok || got != code {
			t.Errorf("code for %q = %d (%v), want %d", key, got, ok, code)
		}
	}
	for _, key := range []frames.KeypadEntry{"A", "x", "12", ""} {
		if _, ok := dtmfCodes[key]; ok {
			t.Errorf("%q has a code, want it refused", key)
		}
	}
}

// The output presses keys natively, as SIP telephone events rather than as
// tones in the audio, and refuses a key a SIP leg cannot carry without
// touching the room.
func TestOutputPressesKeysNatively(t *testing.T) {
	out := newOutput(&Connection{}, transport.Params{})
	if !out.SupportsNativeDTMF() {
		t.Fatal("the output does not press keys natively")
	}
	if err := out.WriteDTMFNative(t.Context(), frames.NewOutputDTMFFrame("A")); err != nil {
		t.Errorf("an invalid key = %v, want it refused quietly", err)
	}
}
