package rtviws_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/rtviws"
)

func TestDeserializeWrapsRTVIMessage(t *testing.T) {
	raw, _ := json.Marshal(rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: "req-1",
	})

	f, err := rtviws.New().Deserialize(raw)
	if err != nil {
		t.Fatal(err)
	}
	msg, ok := f.(*frames.InputTransportMessageFrame)
	if !ok {
		t.Fatalf("expected *InputTransportMessageFrame, got %T", f)
	}
	if !bytes.Equal(msg.Message, raw) {
		t.Fatalf("wrapped payload = %s, want %s", msg.Message, raw)
	}
	// The wrapped bytes must parse back to the same RTVI message.
	in, err := rtvi.ParseIncoming(msg.Message)
	if err != nil || in.Type != rtvi.TypeClientReady || in.ID != "req-1" {
		t.Fatalf("round-trip failed: %+v (err %v)", in, err)
	}
}

func TestDeserializeCopiesInput(t *testing.T) {
	raw, _ := json.Marshal(rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady})
	buf := append([]byte(nil), raw...)

	f, _ := rtviws.New().Deserialize(buf)
	msg, ok := f.(*frames.InputTransportMessageFrame)
	if !ok {
		t.Fatalf("expected *InputTransportMessageFrame, got %T", f)
	}

	// Mutating the caller's buffer must not corrupt the frame's copy.
	for i := range buf {
		buf[i] = 'x'
	}
	if !bytes.Equal(msg.Message, raw) {
		t.Fatalf("frame aliased the caller's buffer: %s", msg.Message)
	}
}

func TestDeserializeRawAudio(t *testing.T) {
	pcm := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06}
	raw, _ := json.Marshal(rtviws.RawAudio(pcm, 16000, 1))

	f, err := rtviws.New().Deserialize(raw)
	if err != nil {
		t.Fatal(err)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("expected *InputAudioRawFrame, got %T", f)
	}
	if !bytes.Equal(af.Audio, pcm) || af.SampleRate != 16000 || af.NumChannels != 1 {
		t.Fatalf("unexpected audio frame: audio=%v rate=%d ch=%d", af.Audio, af.SampleRate, af.NumChannels)
	}
}

func TestDeserializeIgnoresNonRTVI(t *testing.T) {
	cases := map[string][]byte{
		"wrong label": []byte(`{"label":"other","type":"x"}`),
		"garbage":     []byte("not json"),
		"empty":       nil,
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			f, err := rtviws.New().Deserialize(data)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if f != nil {
				t.Fatalf("expected no frame, got %T", f)
			}
		})
	}
}

func TestSerializeDropsFrames(t *testing.T) {
	s := rtviws.New()
	for _, f := range []frames.Frame{
		frames.NewOutputAudioRawFrame([]byte{0, 0}, 16000, 1),
		frames.NewInterruptionFrame(),
		frames.NewEndFrame(),
	} {
		out, err := serialized(s.Serialize(f))
		if err != nil || out != nil {
			t.Fatalf("Serialize(%T) = %v, %v; want nil, nil", f, out, err)
		}
	}
}

// TestSerializeRTVIServerMessageToJSON is the outbound half of the wire: an RTVI
// server message the bot produced has to reach the client as its own JSON, since
// that is the only thing this wire carries.
func TestSerializeRTVIServerMessageToJSON(t *testing.T) {
	s := rtviws.New()
	msg := rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeBotReady}

	for _, f := range []frames.Frame{
		frames.NewOutputTransportMessageFrame(msg),
		frames.NewOutputTransportMessageUrgentFrame(msg),
	} {
		out, err := serialized(s.Serialize(f))
		if err != nil {
			t.Fatalf("Serialize(%T): %v", f, err)
		}
		var got struct {
			Label string `json:"label"`
			Type  string `json:"type"`
		}
		if err := json.Unmarshal(out, &got); err != nil {
			t.Fatalf("Serialize(%T) produced %q: %v", f, out, err)
		}
		if got.Label != rtvi.MessageLabel || got.Type != rtvi.TypeBotReady {
			t.Errorf("Serialize(%T) = %q, want the bot-ready message", f, out)
		}
	}
}

// TestSerializeDropsNonRTVIMessage checks a message that is not the protocol's
// is not written. Nothing else is defined on this wire, so a client would have
// no way to read it.
func TestSerializeDropsNonRTVIMessage(t *testing.T) {
	s := rtviws.New()
	f := frames.NewOutputTransportMessageUrgentFrame(map[string]any{"label": "other", "type": "x"})

	out, err := serialized(s.Serialize(f))
	if err != nil || out != nil {
		t.Fatalf("Serialize(non-RTVI message) = %q, %v; want nil, nil", out, err)
	}
}

// serialized unwraps a serializer result to the bytes it produced. These tests
// are about the wire format, not about how the message carrying it is framed.
func serialized(m wsserver.Message, err error) ([]byte, error) { return m.Data, err }
