package livekit

import "testing"

// TestGetClientIDIsTheParticipantIdentity checks the id a client event's client
// maps to is the identity of the participant it names.
func TestGetClientIDIsTheParticipantIdentity(t *testing.T) {
	tr := &Transport{}
	if got := tr.GetClientID(Client{ID: "alice"}); got != "alice" {
		t.Errorf("client id %q, want alice", got)
	}
	if got := tr.GetClientID("alice"); got != "" {
		t.Errorf("client id %q for something that is not a Client, want none", got)
	}
}
