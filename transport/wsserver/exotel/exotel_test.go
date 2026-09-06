package exotel

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver"
)

func pcm() []byte {
	b := make([]byte, 160*2)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

// wireRate is Exotel's default stream rate, whatever the pipeline runs at.
const wireRate = 8000

// ready returns a serializer set up for a pipeline running at rate. Every
// serializer is set up before it converts anything, so a test that skips it is
// testing a state the transport never puts it in.
func ready(t *testing.T, cfg Config, rate int) *Serializer {
	t.Helper()
	s := New(cfg)
	setup := processor.Setup{AudioInSampleRate: rate, AudioOutSampleRate: rate}
	if err := s.Setup(setup); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

func TestDeserializeMediaIsRawPCM(t *testing.T) {
	s := ready(t, Config{}, wireRate)
	payload := base64.StdEncoding.EncodeToString(pcm())
	f, err := s.Deserialize([]byte(`{"event":"media","media":{"payload":"` + payload + `"}}`))
	if err != nil {
		t.Fatalf("deserialize media: %v", err)
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		t.Fatalf("media frame type = %T", f)
	}
	if af.SampleRate != wireRate {
		t.Fatalf("sample rate = %d, want %d", af.SampleRate, wireRate)
	}
	// Exotel is raw PCM: the decoded bytes must match the input exactly.
	if !bytes.Equal(af.Audio, pcm()) {
		t.Fatalf("decoded audio does not round-trip raw PCM")
	}
}

func TestSerializeAudioAfterStart(t *testing.T) {
	s := ready(t, Config{}, wireRate)
	start := `{"event":"start","start":{"stream_sid":"stream-1","call_sid":"call-1"}}`
	if _, err := s.Deserialize([]byte(start)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := serialized(s.Serialize(frames.NewTTSAudioRawFrame(pcm(), wireRate, 1)))
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
	s := ready(t, Config{}, wireRate)
	msg, err := serialized(s.Serialize(frames.NewTTSAudioRawFrame(pcm(), wireRate, 1)))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	if msg != nil {
		t.Fatalf("audio before start should be dropped, got %q", msg)
	}
}

func TestSerializeInterruption(t *testing.T) {
	s := ready(t, Config{}, wireRate)
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"stream_sid":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}
	msg, err := serialized(s.Serialize(frames.NewInterruptionFrame()))
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
// TestSetupTakesThePipelineRate checks the serializer converts to whatever rate
// the pipeline runs at, rather than forcing the pipeline down to the stream's.
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
}

// TestConvertsBetweenWireAndPipelineRates checks a pipeline above the stream
// rate gets its audio converted at each edge. Exotel streams linear PCM, so
// only the rate changes.
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
	s := ready(t, Config{}, pipelineRate)
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"stream_sid":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}

	sent := 0
	for range chunks {
		msg, err := serialized(s.Serialize(frames.NewTTSAudioRawFrame(make([]byte, samples*2), pipelineRate, 1)))
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
		sent += len(raw) / 2
	}
	if want := chunks * wireSamples; sent < want*9/10 || sent > want*11/10 {
		t.Errorf("outbound produced %d samples, want about %d", sent, want)
	}

	payload := base64.StdEncoding.EncodeToString(make([]byte, wireSamples*2))
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
}

// TestSerializeApplicationMessage covers the message path: a message the
// pipeline addresses to the client is sent as the provider's own JSON, and an
// RTVI message is not. RTVI is the protocol a browser client speaks, and Exotel
// expects its own control messages on this socket.
func TestSerializeApplicationMessage(t *testing.T) {
	s := ready(t, Config{}, wireRate)

	out, err := serialized(s.Serialize(frames.NewOutputTransportMessageUrgentFrame(
		map[string]any{"event": "provider-control"})))
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("Serialize produced %q: %v", out, err)
	}
	if got["event"] != "provider-control" {
		t.Errorf("Serialize = %q, want the provider's own message", out)
	}

	rtviMsg := rtvi.Message{Label: rtvi.MessageLabel, Type: rtvi.TypeBotReady}
	out, err = serialized(s.Serialize(frames.NewOutputTransportMessageUrgentFrame(rtviMsg)))
	if err != nil || out != nil {
		t.Errorf("Serialize(RTVI message) = %q, %v; want nil, nil", out, err)
	}
}

// serialized unwraps a serializer result to the bytes it produced. These tests
// are about the wire format, not about how the message carrying it is framed.
func serialized(m wsserver.Message, err error) ([]byte, error) { return m.Data, err }

// TestOutboundNamesTheStreamAsExotelDoes pins the wire key of the stream
// identifier on the messages the bot sends. Exotel spells it stream_sid in both
// directions, and asserting through the Go struct would pass whatever the tag
// said, so the raw JSON is what is read here.
func TestOutboundNamesTheStreamAsExotelDoes(t *testing.T) {
	s := ready(t, Config{}, wireRate)
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"stream_sid":"stream-1"}}`)); err != nil {
		t.Fatalf("deserialize start: %v", err)
	}

	audio, err := serialized(s.Serialize(frames.NewOutputAudioRawFrame(make([]byte, 320), wireRate, 1)))
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	cleared, err := serialized(s.Serialize(frames.NewInterruptionFrame()))
	if err != nil {
		t.Fatalf("serialize interruption: %v", err)
	}

	for name, msg := range map[string][]byte{"media": audio, "clear": cleared} {
		var raw map[string]any
		if err := json.Unmarshal(msg, &raw); err != nil {
			t.Fatalf("unmarshal %s: %v", name, err)
		}
		if raw["stream_sid"] != "stream-1" {
			t.Errorf("%s stream_sid = %v, want the identifier the stream opened with", name, raw["stream_sid"])
		}
		if _, ok := raw["streamSid"]; ok {
			t.Errorf("%s carries streamSid, which Exotel does not read", name)
		}
	}
}
