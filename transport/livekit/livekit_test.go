package livekit

import (
	"testing"

	"github.com/gojargo/jargo/frames"
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

func TestConnectValidatesConfig(t *testing.T) {
	if _, err := Connect(Config{}); err == nil {
		t.Fatal("Connect with empty config should fail validation")
	}
}

// A key a SIP caller presses reaches the handler as a keypad entry, and the keys
// pressed before anything registered are not lost.
func TestDeliverDTMF(t *testing.T) {
	c := &Connection{}

	// Pressed before the input transport has registered its handler.
	c.deliverDTMF("1")
	c.deliverDTMF("#")

	var got []frames.KeypadEntry
	c.OnDTMF(func(e frames.KeypadEntry) { got = append(got, e) })
	if len(got) != 2 || got[0] != frames.KeypadOne || got[1] != frames.KeypadPound {
		t.Fatalf("keys held before a handler = %v, want 1 then #", got)
	}

	c.deliverDTMF("7")
	if len(got) != 3 || got[2] != frames.KeypadSeven {
		t.Errorf("keys = %v, want 7 delivered straight through", got)
	}
}

// Anything that is not a keypad key is dropped rather than pushed downstream as
// a keypress nobody can act on.
func TestDeliverDTMFIgnoresUnknownKeys(t *testing.T) {
	c := &Connection{}
	var got []frames.KeypadEntry
	c.OnDTMF(func(e frames.KeypadEntry) { got = append(got, e) })

	for _, digit := range []string{"", "12", "x", "%"} {
		c.deliverDTMF(digit)
	}
	if len(got) != 0 {
		t.Errorf("delivered %v for keys that are not on a keypad", got)
	}
}
