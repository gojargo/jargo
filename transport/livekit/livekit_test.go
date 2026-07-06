package livekit

import "testing"

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
