package rtc_test

import (
	"testing"

	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
)

// TestGetClientIDIsThePeerConnectionID checks the id a client event's client
// maps to is the id of the peer connection it carries.
func TestGetClientIDIsThePeerConnectionID(t *testing.T) {
	conn, err := rtc.NewConnection(rtc.WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	tr := rtc.NewTransport(conn, transport.DefaultParams())

	if got := tr.GetClientID(conn); got != conn.PCID() {
		t.Errorf("client id %q, want the peer connection id %q", got, conn.PCID())
	}
	if got := tr.GetClientID("not a connection"); got != "" {
		t.Errorf("client id %q for something that is not a connection, want none", got)
	}
}
