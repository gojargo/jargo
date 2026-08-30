package rtvi_test

import (
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// clientReady sends a client-ready declaring version, or none at all when
// version is empty, and returns the bot-ready and the complaint that came back.
// The complaint is empty when the bot made none.
func clientReady(t *testing.T, c *client, version string) (botReady rtvi.Message, complaint string) {
	t.Helper()
	msg := rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: "req-1"}
	if version != "" {
		msg.Data = rtvi.ClientReadyData{
			Version: version,
			About:   rtvi.AboutClientData{Library: "test-client"},
		}
	}
	c.send(t, msg)

	deadline := time.After(3 * time.Second)
	for {
		select {
		case m := <-c.out:
			switch m.Type {
			case rtvi.TypeErrorResponse:
				d, ok := m.Data.(rtvi.ErrorResponseData)
				if !ok {
					t.Fatalf("error-response data = %+v, want a reason", m.Data)
				}
				complaint = d.Error
			case rtvi.TypeBotReady:
				return m, complaint
			}
		case <-deadline:
			t.Fatal("timed out waiting for a bot-ready")
			return rtvi.Message{}, ""
		}
	}
}

// botReadyVersion is the version a bot-ready declared.
func botReadyVersion(t *testing.T, m rtvi.Message) string {
	t.Helper()
	d, ok := m.Data.(rtvi.BotReadyData)
	if !ok {
		t.Fatalf("bot-ready data = %+v, want BotReadyData", m.Data)
	}
	return d.Version
}

// TestClientReadyAcceptsThisProtocolGeneration checks a client speaking this
// generation is answered with this implementation's version and told nothing is
// wrong, whatever minor and patch it declares.
func TestClientReadyAcceptsThisProtocolGeneration(t *testing.T) {
	declared := map[string][3]int{"2.0.0": {2, 0, 0}, "2.3.1": {2, 3, 1}}
	for _, version := range []string{"2.0.0", "2.3.1"} {
		t.Run(version, func(t *testing.T) {
			c := newClient(t)
			ready, complaint := clientReady(t, c, version)
			if complaint != "" {
				t.Errorf("a %s client was told %q, want nothing", version, complaint)
			}
			if got := botReadyVersion(t, ready); got != rtvi.ProtocolVersion {
				t.Errorf("bot-ready version = %q, want %q", got, rtvi.ProtocolVersion)
			}
			if got := c.proc.ClientVersion(); got != declared[version] {
				t.Errorf("ClientVersion() = %v, want %v", got, declared[version])
			}
		})
	}
}

// TestClientReadyAnswersALegacyClientWithItsOwnVersion checks the older protocol
// generation is served rather than turned away, and is told its own version
// back: advertising this one would push it onto paths it has no code for.
func TestClientReadyAnswersALegacyClientWithItsOwnVersion(t *testing.T) {
	for _, version := range []string{"1.0.0", "1.2.0", "1.4.0"} {
		t.Run(version, func(t *testing.T) {
			c := newClient(t)
			ready, complaint := clientReady(t, c, version)
			if complaint != "" {
				t.Errorf("a %s client was told %q, want nothing", version, complaint)
			}
			if got := botReadyVersion(t, ready); got != version {
				t.Errorf("bot-ready version = %q, want the client's own %q", got, version)
			}
		})
	}
}

// TestClientReadyReportsAnIncompatibleVersion checks a client from neither
// generation is told so, and is still answered: it is better placed than the bot
// to decide whether to carry on.
func TestClientReadyReportsAnIncompatibleVersion(t *testing.T) {
	for _, version := range []string{"0.3.0", "0.9.9", "3.0.0"} {
		t.Run(version, func(t *testing.T) {
			c := newClient(t)
			ready, complaint := clientReady(t, c, version)
			if !strings.Contains(complaint, version) || !strings.Contains(complaint, "not compatible") {
				t.Errorf("a %s client was told %q, want an incompatibility naming its version",
					version, complaint)
			}
			if !strings.Contains(complaint, "Compatibility issues may occur") {
				t.Errorf("complaint = %q, want the compatibility warning", complaint)
			}
			if ready.ID != "req-1" {
				t.Errorf("bot-ready id = %q, want the session to go ahead anyway", ready.ID)
			}
		})
	}
}

// TestClientReadyReportsAnUnreadableVersion checks a version that is not three
// numbers is refused rather than guessed at: the negotiation turns on the major
// version, so a version that cannot be read settles nothing.
func TestClientReadyReportsAnUnreadableVersion(t *testing.T) {
	for _, version := range []string{"not-a-version", "123", "1.2.3.0", "junk", "1.2"} {
		t.Run(version, func(t *testing.T) {
			c := newClient(t)
			_, complaint := clientReady(t, c, version)
			if !strings.Contains(complaint, "invalid client version format") ||
				!strings.Contains(complaint, version) {
				t.Errorf("a %q client was told %q, want an invalid-format complaint naming it",
					version, complaint)
			}
		})
	}
}

// TestClientReadyReportsAMissingVersion checks a client that declares no version
// is told so, and is still answered.
func TestClientReadyReportsAMissingVersion(t *testing.T) {
	c := newClient(t)
	ready, complaint := clientReady(t, c, "")
	if !strings.Contains(complaint, "unknown") {
		t.Errorf("complaint = %q, want it to say the version is unknown", complaint)
	}
	if got := botReadyVersion(t, ready); got != rtvi.ProtocolVersion {
		t.Errorf("bot-ready version = %q, want %q", got, rtvi.ProtocolVersion)
	}
}

// TestClientReadyStartsAudioWhateverTheVersion checks the handshake's other half
// happens regardless: a version complaint is advice, not a refusal, so the
// client is heard.
func TestClientReadyStartsAudioWhateverTheVersion(t *testing.T) {
	c := newClient(t)
	c.send(t, rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: "req-1",
		Data: rtvi.ClientReadyData{Version: "0.3.0"},
	})
	await[*frames.InputTransportStartAudioStreamingFrame](t, c)
}

// TestBotReadyDescribesTheBot checks the client is told what it is talking to,
// which is how it reports the server side of a session.
func TestBotReadyDescribesTheBot(t *testing.T) {
	c := newClient(t)
	ready, _ := clientReady(t, c, rtvi.ProtocolVersion)
	d, ok := ready.Data.(rtvi.BotReadyData)
	if !ok || d.About == nil {
		t.Fatalf("bot-ready data = %+v, want a description of the bot", ready.Data)
	}
	if d.About.Library != rtvi.LibraryName {
		t.Errorf("about.library = %q, want %q", d.About.Library, rtvi.LibraryName)
	}
}

// TestClientReadyDataRoundTrips checks the wire shape of a client-ready is the
// one the client SDKs send.
func TestClientReadyDataRoundTrips(t *testing.T) {
	raw := []byte(`{"version":"2.1.0","about":{"library":"rtvi-web","library_version":"1.2.3"}}`)
	got, err := rtvi.ParseClientReadyData(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Version != "2.1.0" || got.About.Library != "rtvi-web" || got.About.LibraryVersion != "1.2.3" {
		t.Errorf("client-ready = %+v, want the version and library the client declared", got)
	}
}
