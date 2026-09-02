package wsserver_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
)

// rtviSerializer is a test serializer whose wire is the one RTVI messages travel
// on, the way a browser client's is.
type rtviSerializer struct{ testSerializer }

func newRTVISerializer() *rtviSerializer {
	s := &rtviSerializer{}
	s.KeepRTVIMessages = true
	return s
}

// readRaw reads one message from the socket, reporting whether anything arrived
// before the wait ran out.
func (c *call) readRaw(t *testing.T, wait time.Duration) ([]byte, bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), wait)
	defer cancel()
	_, data, err := c.client.Read(ctx)
	if err != nil {
		return nil, false
	}
	return data, true
}

// botReady is an RTVI message of the kind a processor sends to a client.
func botReady() rtvi.Message {
	return rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeBotReady}
}

// isRTVI reports whether raw wire bytes are an RTVI message.
func isRTVI(t *testing.T, data []byte) bool {
	t.Helper()
	var m struct {
		Label string `json:"label"`
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return false
	}
	return m.Label == rtvi.MessageLabel
}

// A telephony provider's media stream is not where the RTVI protocol belongs. A
// pipeline that has an RTVI processor in it still runs one on a phone call, so
// the transport is what keeps the protocol off the wire.
func TestRTVIMessagesAreKeptOffATelephonyWire(t *testing.T) {
	c := dial(t, &testSerializer{}, wsserver.Params{Params: transport.Params{AudioOutEnabled: true}})
	defer c.shutdown(t)

	c.task.QueueFrame(frames.NewOutputTransportMessageUrgentFrame(botReady()))

	// Give the message every chance to arrive before concluding it did not.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		data, ok := c.readRaw(t, 300*time.Millisecond)
		if !ok {
			continue
		}
		if isRTVI(t, data) {
			t.Fatalf("an RTVI message reached the telephony wire: %s", data)
		}
	}
}

// The wire a browser client connects over is the one the protocol is for, so a
// serializer that says so has its messages passed through.
func TestRTVIMessagesReachAWireThatCarriesThem(t *testing.T) {
	c := dial(t, newRTVISerializer(), wsserver.Params{Params: transport.Params{AudioOutEnabled: true}})
	defer c.shutdown(t)

	c.task.QueueFrame(frames.NewOutputTransportMessageUrgentFrame(botReady()))

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		data, ok := c.readRaw(t, 500*time.Millisecond)
		if !ok {
			continue
		}
		if isRTVI(t, data) {
			return
		}
	}
	t.Fatal("the RTVI message never reached the wire that carries them")
}

// A message that is not RTVI is nobody's protocol to filter: it goes out on a
// telephony wire like any other, which is how a provider's own control messages
// are sent.
func TestNonRTVIMessagesStillReachATelephonyWire(t *testing.T) {
	c := dial(t, &testSerializer{}, wsserver.Params{Params: transport.Params{AudioOutEnabled: true}})
	defer c.shutdown(t)

	c.task.QueueFrame(frames.NewOutputTransportMessageUrgentFrame(
		map[string]any{"event": "provider-control"}))

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		data, ok := c.readRaw(t, 500*time.Millisecond)
		if !ok {
			continue
		}
		var m map[string]any
		if json.Unmarshal(data, &m) == nil && m["event"] == "provider-control" {
			return
		}
	}
	t.Fatal("the provider's own control message never reached the wire")
}
