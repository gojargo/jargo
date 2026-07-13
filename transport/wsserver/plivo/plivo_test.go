package plivo

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
)

// silence is 20 ms of 8 kHz mono PCM (160 samples).
func silence() []byte { return make([]byte, 160*2) }

func TestDeserializeStartThenMedia(t *testing.T) {
	s := New(Config{})

	start := `{"event":"start","start":{"streamId":"stream-1","callId":"call-1"}}`
	f, err := s.Deserialize([]byte(start))
	if err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	if f != nil {
		t.Fatalf("start should carry no frame, got %T", f)
	}

	payload := base64.StdEncoding.EncodeToString(g711.EncodeULaw(silence()))
	media := `{"event":"media","media":{"payload":"` + payload + `"}}`
	f, err = s.Deserialize([]byte(media))
	if err != nil {
		t.Fatalf("deserialize media: %v", err)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("media frame type = %T, want *InputAudioRawFrame", f)
	}
	if af.SampleRate != sampleRate || af.NumChannels != 1 {
		t.Fatalf("media frame = %d Hz/%d ch, want %d/1", af.SampleRate, af.NumChannels, sampleRate)
	}
	if len(af.Audio) != len(silence()) {
		t.Fatalf("decoded audio len = %d, want %d", len(af.Audio), len(silence()))
	}
}

func TestDeserializeDTMF(t *testing.T) {
	s := New(Config{})
	f, err := s.Deserialize([]byte(`{"event":"dtmf","dtmf":{"digit":"5"}}`))
	if err != nil {
		t.Fatalf("deserialize dtmf: %v", err)
	}
	d, ok := f.(*frames.InputDTMFFrame)
	if !ok {
		t.Fatalf("dtmf frame type = %T, want *InputDTMFFrame", f)
	}
	if d.Button != frames.KeypadFive {
		t.Fatalf("dtmf button = %q, want 5", d.Button)
	}
}

func TestSerializeAudioDroppedBeforeStart(t *testing.T) {
	s := New(Config{})
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(silence(), sampleRate, 1))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	if msg != nil {
		t.Fatalf("audio before start should be dropped, got %q", msg)
	}
}

func TestSerializeAudioAfterStart(t *testing.T) {
	s := New(Config{})
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"streamId":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(silence(), sampleRate, 1))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	var out playAudio
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal playAudio: %v", err)
	}
	if out.Event != "playAudio" || out.StreamID != "stream-1" {
		t.Fatalf("playAudio = %+v, want event playAudio / stream-1", out)
	}
	if out.Media.ContentType != "audio/x-mulaw" || out.Media.SampleRate != sampleRate {
		t.Fatalf("media header = %+v", out.Media)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
	if err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	if len(raw) != len(silence())/2 {
		t.Fatalf("μ-law payload len = %d, want %d", len(raw), len(silence())/2)
	}
}

func TestSerializeInterruption(t *testing.T) {
	s := New(Config{})
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"streamId":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := s.Serialize(frames.NewInterruptionFrame())
	if err != nil {
		t.Fatalf("serialize interruption: %v", err)
	}
	var out clearAudio
	if err := json.Unmarshal(msg, &out); err != nil {
		t.Fatalf("unmarshal clearAudio: %v", err)
	}
	if out.Event != "clearAudio" || out.StreamID != "stream-1" {
		t.Fatalf("clearAudio = %+v", out)
	}
}
