package eval

import (
	"github.com/gojargo/jargo/transport/wsclient"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// ClientTransport is the harness's RTVI client transport. On connect it sends
// the connection's flags in an eval-connect message, ahead of the RTVI
// handshake, which is how a bot hosted by the development runner recognizes the
// harness and learns what the connection asks of it.
type ClientTransport struct {
	*wsclient.RTVITransport

	flags evalws.ConnectionFlags
}

// NewClientTransport builds the harness's transport to the bot at uri, sending
// flags first on every connection. The serializer defaults to a
// ClientSerializer when params does not set one.
func NewClientTransport(uri string, params wsclient.Params, flags evalws.ConnectionFlags) *ClientTransport {
	if params.Serializer == nil {
		params.Serializer = NewClientSerializer()
	}
	t := &ClientTransport{RTVITransport: wsclient.NewRTVITransport(uri, params), flags: flags}
	t.SetOpeningMessages(wsserver.Message{Data: []byte(evalws.ConnectMessage(flags))})
	return t
}

// Flags are the connection's flags, as sent in eval-connect.
func (t *ClientTransport) Flags() evalws.ConnectionFlags { return t.flags }
