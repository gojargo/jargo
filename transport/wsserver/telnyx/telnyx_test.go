package telnyx

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
)

func silence() []byte { return make([]byte, 160*2) }

func TestDeserializeStartLearnsEncoding(t *testing.T) {
	s := New(Config{})

	start := `{"event":"start","stream_id":"stream-1",` +
		`"start":{"call_control_id":"cc-1","media_format":{"encoding":"PCMA"}}}`
	if _, err := s.Deserialize([]byte(start)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}

	// After a PCMA start, inbound media must be decoded as A-law.
	payload := base64.StdEncoding.EncodeToString(g711.EncodeALaw(silence()))
	f, err := s.Deserialize([]byte(`{"event":"media","media":{"payload":"` + payload + `"}}`))
	if err != nil {
		t.Fatalf("deserialize media: %v", err)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("media frame type = %T", f)
	}
	if af.SampleRate != sampleRate || len(af.Audio) != len(silence()) {
		t.Fatalf("media frame = %d Hz, %d bytes", af.SampleRate, len(af.Audio))
	}
}

func TestSerializeAudioDefaultPCMU(t *testing.T) {
	s := New(Config{})
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(silence(), sampleRate, 1))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	var out mediaOut
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal media: %v", err)
	}
	if out.Event != "media" {
		t.Fatalf("event = %q, want media", out.Event)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
	if err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	if len(raw) != len(silence())/2 {
		t.Fatalf("companded payload len = %d, want %d", len(raw), len(silence())/2)
	}
}

func TestSerializeSendEncodingPCMA(t *testing.T) {
	s := New(Config{SendEncoding: EncodingPCMA})
	// A non-silent sample so μ-law and A-law encodings differ.
	pcm := make([]byte, 4)
	pcm[0], pcm[1] = 0x00, 0x40 // 0x4000
	msg, _ := s.Serialize(frames.NewTTSAudioRawFrame(pcm, sampleRate, 1))
	var out mediaOut
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	raw, _ := base64.StdEncoding.DecodeString(out.Media.Payload)
	if got, want := raw, g711.EncodeALaw(pcm); string(got) != string(want) {
		t.Fatalf("payload = % x, want A-law % x", got, want)
	}
}

func TestSerializeInterruptionClear(t *testing.T) {
	s := New(Config{})
	msg, err := s.Serialize(frames.NewInterruptionFrame())
	if err != nil {
		t.Fatalf("serialize interruption: %v", err)
	}
	var out event
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.Event != "clear" {
		t.Fatalf("event = %q, want clear", out.Event)
	}
}
