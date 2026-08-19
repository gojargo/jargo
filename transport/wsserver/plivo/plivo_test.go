package plivo

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport/wsserver"
)

// wireRate is what Plivo streams at, whatever the pipeline runs at.
const wireRate = 8000

// silence is 20 ms of 8 kHz mono PCM (160 samples).
func silence() []byte { return make([]byte, 160*2) }

// ready returns a serializer set up for a pipeline running at rate. Every
// serializer is set up before it converts anything, so a test that skips it is
// testing a state the transport never puts it in.
func ready(t *testing.T, cfg Config, rate int) *Serializer {
	t.Helper()
	s := New(cfg)
	sf := frames.NewStartFrame()
	sf.AudioInSampleRate = rate
	sf.AudioOutSampleRate = rate
	if err := s.Setup(sf); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

func TestDeserializeStartThenMedia(t *testing.T) {
	s := ready(t, Config{}, wireRate)

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
	if af.SampleRate != wireRate || af.NumChannels != 1 {
		t.Fatalf("media frame = %d Hz/%d ch, want %d/1", af.SampleRate, af.NumChannels, wireRate)
	}
	if len(af.Audio) != len(silence()) {
		t.Fatalf("decoded audio len = %d, want %d", len(af.Audio), len(silence()))
	}
}

func TestDeserializeDTMF(t *testing.T) {
	s := ready(t, Config{}, wireRate)
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
	s := ready(t, Config{}, wireRate)
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(silence(), wireRate, 1))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	if msg != nil {
		t.Fatalf("audio before start should be dropped, got %q", msg)
	}
}

func TestSerializeAudioAfterStart(t *testing.T) {
	s := ready(t, Config{}, wireRate)
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"streamId":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(silence(), wireRate, 1))
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
	if out.Media.ContentType != "audio/x-mulaw" || out.Media.SampleRate != wireRate {
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
	s := ready(t, Config{}, wireRate)
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

// TestSetupTakesThePipelineRate checks the serializer converts to whatever rate
// the pipeline runs at, rather than forcing the pipeline down to the wire's.
func TestSetupTakesThePipelineRate(t *testing.T) {
	for _, rate := range []int{8000, 16000, 24000} {
		s := ready(t, Config{}, rate)
		if got := s.codec.SampleRate(); got != rate {
			t.Errorf("pipeline rate = %d, want %d", got, rate)
		}
		if got := s.codec.WireSampleRate(); got != wireRate {
			t.Errorf("wire rate = %d, want %d", got, wireRate)
		}
	}
	s := ready(t, Config{Audio: wsserver.AudioConfig{SampleRate: 24000}}, 16000)
	if got := s.codec.SampleRate(); got != 24000 {
		t.Errorf("pipeline rate = %d, want the configured 24000", got)
	}
}

// TestConvertsBetweenWireAndPipelineRates checks a pipeline above the wire rate
// gets its audio converted at each edge rather than being pinned to 8 kHz, and
// that the payload still announces the rate it is actually at.
func TestConvertsBetweenWireAndPipelineRates(t *testing.T) {
	const (
		pipelineRate = 16000
		samples      = 320 // 20ms at 16kHz
		// Enough audio that the filter length the resampler keeps in flight is
		// a small fraction of it, so what is measured is the steady-state rate
		// ratio rather than the startup transient.
		chunks = 200
	)
	s := ready(t, Config{}, pipelineRate)
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"streamId":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}

	sent := 0
	for range chunks {
		msg, err := s.Serialize(frames.NewTTSAudioRawFrame(make([]byte, samples*2), pipelineRate, 1))
		if err != nil {
			t.Fatalf("serialize: %v", err)
		}
		if msg == nil {
			continue // the resampler is still holding the filter delay back
		}
		var out playAudio
		if err := json.Unmarshal(msg, &out); err != nil {
			t.Fatalf("unmarshal playAudio: %v", err)
		}
		if out.Media.SampleRate != wireRate {
			t.Fatalf("payload announces %d Hz, want the wire's %d", out.Media.SampleRate, wireRate)
		}
		raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
		if err != nil {
			t.Fatalf("decode payload: %v", err)
		}
		sent += len(raw)
	}
	if want := chunks * samples * wireRate / pipelineRate; sent < want*9/10 || sent > want*11/10 {
		t.Errorf("outbound produced %d μ-law bytes, want about %d", sent, want)
	}

	// Inbound the other way: 8 kHz on the wire arrives as 16 kHz in the pipeline.
	payload := base64.StdEncoding.EncodeToString(make([]byte, 160))
	got := 0
	for range chunks {
		f, err := s.Deserialize([]byte(`{"event":"media","media":{"payload":"` + payload + `"}}`))
		if err != nil {
			t.Fatalf("deserialize media: %v", err)
		}
		if f == nil {
			continue
		}
		af, ok := f.(*frames.InputAudioRawFrame)
		if !ok {
			t.Fatalf("media frame type = %T, want *frames.InputAudioRawFrame", f)
		}
		if af.SampleRate != pipelineRate {
			t.Fatalf("frame sample rate = %d, want the pipeline's %d", af.SampleRate, pipelineRate)
		}
		got += len(af.Audio) / 2
	}
	if want := chunks * 160 * pipelineRate / wireRate; got < want*9/10 || got > want*11/10 {
		t.Errorf("inbound produced %d samples, want about %d", got, want)
	}
}
