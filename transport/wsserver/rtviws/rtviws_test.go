package rtviws_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
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
		out, err := s.Serialize(f)
		if err != nil || out != nil {
			t.Fatalf("Serialize(%T) = %v, %v; want nil, nil", f, out, err)
		}
	}
}
