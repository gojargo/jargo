package telnyx

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
)

// wireRate is what Telnyx streams at, whatever the pipeline runs at.
const wireRate = 8000

func silence() []byte { return make([]byte, 160*2) }

// ready returns a serializer set up for a pipeline running at rate. Every
// serializer is set up before it converts anything, so a test that skips it is
// testing a state the transport never puts it in.
//
// A configuration naming no credentials has the hang-up turned off, since Setup
// refuses one that is to hang up with nothing to authorize it. These tests are
// about the wire format, not about ending the call.
func ready(t *testing.T, cfg Config, rate int) *Serializer {
	t.Helper()
	if cfg.AutoHangUp == nil && cfg.APIKey == "" {
		off := false
		cfg.AutoHangUp = &off
	}
	s := New(cfg)
	setup := processor.Setup{AudioInSampleRate: rate, AudioOutSampleRate: rate}
	if err := s.Setup(setup); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

func TestDeserializeStartLearnsEncoding(t *testing.T) {
	s := ready(t, Config{}, wireRate)

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
	if af.SampleRate != wireRate || len(af.Audio) != len(silence()) {
		t.Fatalf("media frame = %d Hz, %d bytes", af.SampleRate, len(af.Audio))
	}
}

func TestSerializeAudioDefaultPCMU(t *testing.T) {
	s := ready(t, Config{}, wireRate)
	msg, err := s.Serialize(frames.NewTTSAudioRawFrame(silence(), wireRate, 1))
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
	s := ready(t, Config{SendEncoding: EncodingPCMA}, wireRate)
	// A non-silent sample so μ-law and A-law encodings differ.
	pcm := make([]byte, 4)
	pcm[0], pcm[1] = 0x00, 0x40 // 0x4000
	msg, _ := s.Serialize(frames.NewTTSAudioRawFrame(pcm, wireRate, 1))
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
	s := ready(t, Config{}, wireRate)
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
// gets its audio converted at each edge rather than being pinned to 8 kHz, for
// both compandings Telnyx speaks.
func TestConvertsBetweenWireAndPipelineRates(t *testing.T) {
	const (
		pipelineRate = 16000
		samples      = 320 // 20ms at 16kHz
		wireSamples  = 160 // the same 20ms at 8kHz
		// Enough audio that the filter length the resampler keeps in flight is
		// a small fraction of it, so what is measured is the steady-state rate
		// ratio rather than the startup transient.
		chunks = 200
	)
	for _, enc := range []string{EncodingPCMU, EncodingPCMA} {
		t.Run(enc, func(t *testing.T) {
			s := ready(t, Config{SendEncoding: enc}, pipelineRate)
			start := `{"event":"start","stream_id":"stream-1",` +
				`"start":{"call_control_id":"cc-1","media_format":{"encoding":"` + enc + `"}}}`
			if _, err := s.Deserialize([]byte(start)); err != nil {
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
				var out mediaOut
				if err := json.Unmarshal(msg, &out); err != nil {
					t.Fatalf("unmarshal media: %v", err)
				}
				raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
				if err != nil {
					t.Fatalf("decode payload: %v", err)
				}
				sent += len(raw)
			}
			if want := chunks * wireSamples; sent < want*9/10 || sent > want*11/10 {
				t.Errorf("outbound produced %d companded bytes, want about %d", sent, want)
			}

			payload := base64.StdEncoding.EncodeToString(make([]byte, wireSamples))
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
			if want := chunks * samples; got < want*9/10 || got > want*11/10 {
				t.Errorf("inbound produced %d samples, want about %d", got, want)
			}
		})
	}
}
