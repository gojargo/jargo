package protobuf_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/protobuf"
)

// roundtrip encodes f and decodes what comes back, which is the only property
// that matters on a wire with a client at the other end.
func roundtrip(t *testing.T, s *protobuf.Serializer, f frames.Frame) frames.Frame {
	t.Helper()
	msg, err := s.Serialize(f)
	if err != nil {
		t.Fatalf("Serialize(%T): %v", f, err)
	}
	if msg.Empty() {
		t.Fatalf("Serialize(%T) produced nothing", f)
	}
	if !msg.Binary {
		t.Errorf("Serialize(%T) produced a text message; the encoding is binary", f)
	}
	got, err := s.Deserialize(msg.Data)
	if err != nil {
		t.Fatalf("Deserialize(%T): %v", f, err)
	}
	return got
}

func TestTextFrameRoundtrip(t *testing.T) {
	got := roundtrip(t, protobuf.New(), frames.NewTextFrame("hello world"))

	tf, ok := got.(*frames.TextFrame)
	if !ok {
		t.Fatalf("Deserialize returned %T, want *frames.TextFrame", got)
	}
	if tf.Text != "hello world" {
		t.Errorf("Text = %q, want %q", tf.Text, "hello world")
	}
}

func TestTranscriptionFrameRoundtrip(t *testing.T) {
	in := frames.NewTranscriptionFrame("Hello there!", "123", "2021-01-01")
	got := roundtrip(t, protobuf.New(), in)

	tf, ok := got.(*frames.TranscriptionFrame)
	if !ok {
		t.Fatalf("Deserialize returned %T, want *frames.TranscriptionFrame", got)
	}
	if tf.Text != in.Text || tf.UserID != in.UserID || tf.Timestamp != in.Timestamp {
		t.Errorf("got (%q, %q, %q), want (%q, %q, %q)",
			tf.Text, tf.UserID, tf.Timestamp, in.Text, in.UserID, in.Timestamp)
	}
}

// TestAudioFrameRoundtrip is the one that carries the conversation: the bot's
// speech out, and the client's microphone back in as the mic audio the pipeline
// runs on.
func TestAudioFrameRoundtrip(t *testing.T) {
	in := frames.NewOutputAudioRawFrame([]byte("1234567890"), 16000, 1)
	got := roundtrip(t, protobuf.New(), in)

	af, ok := got.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("Deserialize returned %T, want *frames.InputAudioRawFrame", got)
	}
	a := af.AudioData()
	if !bytes.Equal(a.Audio, in.Audio) {
		t.Errorf("Audio = %q, want %q", a.Audio, in.Audio)
	}
	if a.SampleRate != in.SampleRate || a.NumChannels != in.NumChannels {
		t.Errorf("got %d Hz / %d channels, want %d / %d",
			a.SampleRate, a.NumChannels, in.SampleRate, in.NumChannels)
	}
}

func TestInterruptionFrameRoundtrip(t *testing.T) {
	got := roundtrip(t, protobuf.New(), frames.NewInterruptionFrame())

	if _, ok := got.(*frames.InterruptionFrame); !ok {
		t.Fatalf("Deserialize returned %T, want *frames.InterruptionFrame", got)
	}
}

// TestRTVIMessageRoundtrip is what makes this the wire a client drives the bot
// over: the RTVI protocol travels inside the message frame, in both directions.
func TestRTVIMessageRoundtrip(t *testing.T) {
	msg := rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeBotReady}
	got := roundtrip(t, protobuf.New(), frames.NewOutputTransportMessageUrgentFrame(msg))

	mf, ok := got.(*frames.InputTransportMessageFrame)
	if !ok {
		t.Fatalf("Deserialize returned %T, want *frames.InputTransportMessageFrame", got)
	}
	var decoded struct {
		Label string `json:"label"`
		Type  string `json:"type"`
	}
	if err := json.Unmarshal(mf.Message, &decoded); err != nil {
		t.Fatalf("the message did not survive as JSON: %v", err)
	}
	if decoded.Label != rtvi.MessageLabel || decoded.Type != rtvi.TypeBotReady {
		t.Errorf("got %+v, want the bot-ready message", decoded)
	}
}

// TestPTSSurvivesTheRoundtrip covers the one piece of frame identity the wire
// carries back: a presentation timestamp describes the audio, so a client that
// sends one has it honored.
func TestPTSSurvivesTheRoundtrip(t *testing.T) {
	in := frames.NewOutputAudioRawFrame([]byte{1, 2, 3, 4}, 16000, 1)
	in.SetPTS(1234)

	got := roundtrip(t, protobuf.New(), in)

	pts, ok := got.Base().PTS()
	if !ok || pts != 1234 {
		t.Errorf("PTS = %d, %v; want 1234, true", pts, ok)
	}
}

// TestUnsentFramesProduceNothing checks a frame the format has no message for is
// dropped rather than written as an empty one.
func TestUnsentFramesProduceNothing(t *testing.T) {
	s := protobuf.New()
	for _, f := range []frames.Frame{
		frames.NewEndFrame(),
		frames.NewCancelFrame(),
		frames.NewUserSpeakingFrame(),
	} {
		msg, err := s.Serialize(f)
		if err != nil || !msg.Empty() {
			t.Errorf("Serialize(%T) = %v, %v; want nothing to send", f, msg, err)
		}
	}
}

// TestDeserializeRejectsAFrameItCannotRead checks a payload carrying none of the
// five frames is reported rather than turned into one. An empty payload is the
// case that matters: it decodes cleanly as a message with nothing set, which is
// what a client speaking a format this one does not know would send.
func TestDeserializeRejectsAFrameItCannotRead(t *testing.T) {
	s := protobuf.New()
	if _, err := s.Deserialize([]byte{}); !errors.Is(err, protobuf.ErrUnknownFrame) {
		t.Errorf("Deserialize(empty) = %v, want %v", err, protobuf.ErrUnknownFrame)
	}
}

// serializerContract is the compile-time check that this is a usable serializer.
var _ wsserver.Serializer = (*protobuf.Serializer)(nil)
