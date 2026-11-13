package novasonic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// The rate the user's audio is sent to the session at: the input rate the
// session declares, whatever rate the transport delivered it at.

// tone is a 220 Hz sine of n samples, as 16-bit mono PCM at rate.
func tone(rate, n int) []byte {
	pcm := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/float64(rate)) * 8000)
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(v))
	}
	return pcm
}

// resampledTone is pcm at inRate converted to outRate as the start of a fresh
// stream.
func resampledTone(t *testing.T, pcm []byte, inRate, outRate int) []byte {
	t.Helper()
	r, err := resample.New(inRate, outRate, 1)
	if err != nil {
		t.Fatalf("resampler: %v", err)
	}
	return r.Process(pcm)
}

// lastUserAudio is the audio the last audioInput event on st carried.
func lastUserAudio(t *testing.T, st *fakeStream) []byte {
	t.Helper()
	for _, e := range slices.Backward(st.events()) {
		if e.name != "audioInput" {
			continue
		}
		content, _ := e.body[keyContent].(string)
		audio, err := base64.StdEncoding.DecodeString(content)
		if err != nil {
			t.Fatalf("audio content: %v", err)
		}
		return audio
	}
	t.Fatal("no user audio was sent")
	return nil
}

func sendUserAudio(t *testing.T, h *harness, audio []byte, rate int) {
	t.Helper()
	if err := h.s.ProcessFrame(context.Background(), frames.NewInputAudioRawFrame(audio, rate, 1),
		processor.Downstream); err != nil {
		t.Fatal(err)
	}
}

// Audio at another rate than the session declares is resampled to the declared
// rate, not sent as it is under a rate it is not at.
func TestNovaSonicSendsUserAudioAtTheDeclaredInputRate(t *testing.T) {
	h, st := liveStream(t)
	clip := tone(24000, 4800)

	sendUserAudio(t, h, clip, 24000)

	if got, want := lastUserAudio(t, st), resampledTone(t, clip, 24000, 16000); !bytes.Equal(got, want) {
		t.Errorf("sent %d bytes, want the clip resampled to the declared 16 kHz (%d bytes)", len(got), len(want))
	}
}

// The user's audio is dropped while the trigger plays, so the audio after it is
// resampled as the start of a fresh stream rather than carrying on from the
// audio before.
func TestNovaSonicUserAudioAfterTheTriggerIsANewStream(t *testing.T) {
	h := newHarness(t, Config{Model: firstGenerationModel})
	h.s.triggerAudio = make([]byte, 1280)
	h.seed(t, frames.NewLLMContext(AwaitTriggerAssistantResponseInstruction))
	h.s.mu.Lock()
	h.s.connectedTime = time.Now().Add(-time.Second)
	h.s.mu.Unlock()
	st := h.client.stream(t, 0)
	before, after := tone(24000, 4800), tone(24000, 4800)

	sendUserAudio(t, h, before, 24000)
	// The trigger plays here, and the user's audio is dropped while it does.
	h.s.TriggerAssistantResponse(context.Background())
	sendUserAudio(t, h, after, 24000)

	if got, want := lastUserAudio(t, st), resampledTone(t, after, 24000, 16000); !bytes.Equal(got, want) {
		t.Error("the audio after the trigger carries on from the audio before it, want a fresh stream")
	}
}
