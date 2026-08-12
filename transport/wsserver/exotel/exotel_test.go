package exotel

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func pcm() []byte {
	b := make([]byte, 160*2)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

func TestDeserializeMediaIsRawPCM(t *testing.T) {
	s := New(Config{})
	payload := base64.StdEncoding.EncodeToString(pcm())
	f, err := s.Deserialize([]byte(`{"event":"media","media":{"payload":"` + payload + `"}}`))
	if err != nil {
		t.Fatalf("deserialize media: %v", err)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("media frame type = %T", f)
	}
	if af.SampleRate != defaultSampleRate {
		t.Fatalf("sample rate = %d, want %d", af.SampleRate, defaultSampleRate)
	}
	// Exotel is raw PCM: the decoded bytes must match the input exactly.
	if !bytes.Equal(af.Audio, pcm()) {
		t.Fatalf("decoded audio does not round-trip raw PCM")
	}
}

func TestSerializeAudioAfterStart(t *testing.T) {
	s := New(Config{})
	start := `{"event":"start","start":{"stream_sid":"stream-1","call_sid":"call-1"}}`
	if _, err := s.Deserialize([]byte(start)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(pcm(), defaultSampleRate, 1))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	var out mediaOut
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal media: %v", err)
	}
	if out.Event != "media" || out.StreamSID != "stream-1" {
		t.Fatalf("media = %+v", out)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
	if err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	// Raw PCM out: no companding, so bytes are unchanged.
	if !bytes.Equal(raw, pcm()) {
		t.Fatalf("payload is not raw PCM")
	}
}

func TestSerializeAudioDroppedBeforeStart(t *testing.T) {
	s := New(Config{})
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(pcm(), defaultSampleRate, 1))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	if msg != nil {
		t.Fatalf("audio before start should be dropped, got %q", msg)
	}
}

func TestSerializeInterruption(t *testing.T) {
	s := New(Config{})
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"stream_sid":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := s.Serialize(frames.NewInterruptionFrame())
	if err != nil {
		t.Fatalf("serialize interruption: %v", err)
	}
	var out clearOut
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal clear: %v", err)
	}
	if out.Event != "clear" || out.StreamSID != "stream-1" {
		t.Fatalf("clear = %+v", out)
	}
}

// TestSetupIsANoOp checks the serializer needs nothing from the StartFrame:
// Exotel audio is always 8 kHz, so there is no rate to reconcile.
func TestSetupIsANoOp(t *testing.T) {
	if err := New(Config{}).Setup(frames.NewStartFrame()); err != nil {
		t.Fatalf("Setup: %v", err)
	}
}
