package twilio_test

// Ported from upstream's telephony serializer rate tests: outgoing audio is
// converted to the wire rate, incoming audio to the pipeline's, and an
// interruption discards what the outgoing resampler held of the cut-off reply.

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/audio/g711"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/twilio"
)

// rateChunks is how many frames each rate case sends.
const rateChunks = 60

// readySerializer is a Twilio serializer set up for a pipeline at rate, past
// the start message that names the stream.
func readySerializer(t *testing.T, rate int) *twilio.Serializer {
	t.Helper()
	off := false
	s := twilio.New(twilio.Config{AutoHangUp: &off})
	if err := s.Setup(processor.Setup{AudioInSampleRate: rate, AudioOutSampleRate: rate}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	if _, err := s.Deserialize([]byte(`{"event":"start","start":{"streamSid":"stream-sid","callSid":"CA1"}}`)); err != nil {
		t.Fatal(err)
	}
	return s
}

// repeated is b repeated n times.
func repeated(b []byte, n int) []byte {
	out := make([]byte, 0, len(b)*n)
	for range n {
		out = append(out, b...)
	}
	return out
}

// mediaPayload is the wire audio a media message carries.
func mediaPayload(t *testing.T, m wsserver.Message) []byte {
	t.Helper()
	var out struct {
		Media struct {
			Payload string `json:"payload"`
		} `json:"media"`
	}
	if err := json.Unmarshal(m.Data, &out); err != nil {
		t.Fatal(err)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Media.Payload)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// sentBytes sends f rateChunks times and counts the wire bytes sent.
func sentBytes(t *testing.T, s wsserver.Serializer, f frames.Frame) int {
	t.Helper()
	total := 0
	for range rateChunks {
		m, err := s.Serialize(f)
		if err != nil {
			t.Fatal(err)
		}
		if !m.Empty() {
			total += len(mediaPayload(t, m))
		}
	}
	return total
}

// test_audio_at_the_wire_rate_is_encoded_as_is
func TestRatesAudioAtTheWireRateIsEncodedAsIs(t *testing.T) {
	s := readySerializer(t, 16000)
	f := frames.NewOutputAudioRawFrame(repeated([]byte{0x00, 0x10}, 160), 8000, 1)
	if got := sentBytes(t, s, f); got != rateChunks*160 {
		t.Errorf("sent %d bytes, want %d", got, rateChunks*160)
	}
}

// test_audio_at_another_rate_is_converted_to_the_wire_rate
func TestRatesAudioAtAnotherRateIsConvertedToTheWireRate(t *testing.T) {
	s := readySerializer(t, 16000)
	f := frames.NewOutputAudioRawFrame(repeated([]byte{0x00, 0x10}, 480), 24000, 1)
	got := sentBytes(t, s, f)
	if float64(got) <= 0.9*rateChunks*160 || got > rateChunks*160 {
		t.Errorf("sent %d bytes, want in (%v, %d]", got, 0.9*rateChunks*160, rateChunks*160)
	}
}

// test_incoming_audio_is_converted_to_the_pipeline_rate
func TestRatesIncomingAudioIsConvertedToThePipelineRate(t *testing.T) {
	s := readySerializer(t, 16000)
	payload := base64.StdEncoding.EncodeToString(repeated([]byte{0x7f}, 160))
	msg := []byte(`{"event":"media","media":{"payload":"` + payload + `"}}`)
	received, n := 0, 0
	for range rateChunks {
		f, err := s.Deserialize(msg)
		if err != nil {
			t.Fatal(err)
		}
		if f == nil {
			continue
		}
		n++
		a, ok := f.(*frames.InputAudioRawFrame)
		if !ok {
			t.Fatalf("got %T, want audio", f)
		}
		if a.SampleRate != 16000 {
			t.Errorf("rate = %d, want 16000", a.SampleRate)
		}
		received += len(a.Audio) / 2
	}
	if n == 0 {
		t.Fatal("no frames")
	}
	if float64(received) <= 0.9*rateChunks*320 || received > rateChunks*320 {
		t.Errorf("received %d samples, want in (%v, %d]", received, 0.9*rateChunks*320, rateChunks*320)
	}
}

// test_an_interruption_discards_the_cut_off_audio: what follows an
// interruption starts clean, with none of the cut-off reply's tail in it.
func TestRatesAnInterruptionDiscardsTheCutOffAudio(t *testing.T) {
	s := readySerializer(t, 16000)
	loud := frames.NewOutputAudioRawFrame(repeated([]byte{0x00, 0x40}, 480), 24000, 1)
	sentBytes(t, s, loud)

	if _, err := s.Serialize(frames.NewInterruptionFrame()); err != nil {
		t.Fatal(err)
	}

	silence := frames.NewOutputAudioRawFrame(make([]byte, 960), 24000, 1)
	var m wsserver.Message
	for i := 0; m.Empty() && i < 10; i++ {
		var err error
		if m, err = s.Serialize(silence); err != nil {
			t.Fatal(err)
		}
	}
	pcm := g711.DecodeULaw(mediaPayload(t, m))
	peak := 0
	for i := 0; i+1 < len(pcm); i += 2 {
		v := int(int16(binary.LittleEndian.Uint16(pcm[i:])))
		peak = max(peak, v, -v)
	}
	if peak >= 100 {
		t.Errorf("the first message after the interruption peaks at %d, want the cut-off reply's tail discarded", peak)
	}
}
