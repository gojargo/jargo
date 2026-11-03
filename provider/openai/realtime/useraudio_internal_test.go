package realtime

import (
	"bytes"
	"context"
	"encoding/base64"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/openai/realtime/events"
)

// Ports of tests/test_openai_realtime_user_audio.py: with the server's turn
// detection off, user audio is appended to the input buffer and mirrored into a
// rolling pre-roll, which is put back after an interruption clears the buffer;
// with it on, no pre-roll is kept. Pipeline audio reaches the session at 24 kHz.

// kinds are the types of the client events the service sent, in order.
func (r *sentEvents) kinds() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.events))
	for _, e := range r.events {
		switch ev := e.(type) {
		case events.InputAudioBufferAppendEvent:
			out = append(out, ev.Type)
		case events.ClientEvent:
			out = append(out, ev.Type)
		case events.ConversationItemTruncateEvent:
			out = append(out, ev.Type)
		case events.ResponseCreateEvent:
			out = append(out, ev.Type)
		}
	}
	return out
}

// appendPayloads are the base64 audio of the appends the service sent.
func (r *sentEvents) appendPayloads() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, e := range r.events {
		if ev, ok := e.(events.InputAudioBufferAppendEvent); ok {
			out = append(out, ev.Audio)
		}
	}
	return out
}

const (
	kindAppend = "input_audio_buffer.append"
	kindClear  = "input_audio_buffer.clear"
	kindCancel = "response.cancel"
)

// userAudioService is a service whose events are captured, with the server's
// turn detection off when manual is set.
func userAudioService(t *testing.T, manual bool, prerollSecs *float64) (*Service, *sentEvents) {
	t.Helper()
	cfg := Config{UserAudioPrerollSecs: prerollSecs}
	if manual {
		cfg.SessionProperties = events.SessionProperties{
			Audio: &events.AudioConfiguration{Input: &events.AudioInput{TurnDetection: events.NoTurnDetection()}},
		}
	}
	s, sent := offlineService(t, cfg)
	s.Link(newRecorder())
	return s, sent
}

// sendUserAudio sends mono audio at the given rate as the input transport
// would deliver it.
func sendUserAudio(s *Service, rate int, data []byte) {
	s.sendAudio(context.Background(), frames.AudioRawData{Audio: data, SampleRate: rate, NumChannels: 1})
}

func (s *Service) prerollBytes() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]byte(nil), s.preroll...)
}

func b64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

func TestManualModeAppendsAndBuffersAudio(t *testing.T) {
	s, sent := userAudioService(t, true, nil)
	sendUserAudio(s, sampleRate, []byte{0xaa, 0xbb})
	sendUserAudio(s, sampleRate, []byte{0xcc, 0xdd})

	if got := sent.kinds(); !reflect.DeepEqual(got, []string{kindAppend, kindAppend}) {
		t.Errorf("events = %v, want two appends", got)
	}
	want := []string{b64([]byte{0xaa, 0xbb}), b64([]byte{0xcc, 0xdd})}
	if got := sent.appendPayloads(); !reflect.DeepEqual(got, want) {
		t.Errorf("payloads = %v, want %v", got, want)
	}
	if got := s.prerollBytes(); !bytes.Equal(got, []byte{0xaa, 0xbb, 0xcc, 0xdd}) {
		t.Errorf("pre-roll = %x, want aabbccdd", got)
	}
}

func TestManualModeReplaysPrerollAfterInterruption(t *testing.T) {
	s, sent := userAudioService(t, true, nil)
	sendUserAudio(s, sampleRate, []byte{0xaa, 0xbb})
	sendUserAudio(s, sampleRate, []byte{0xcc, 0xdd})

	s.handleInterruption(context.Background())

	// The clear comes before the replay, and the response cancel after.
	want := []string{kindAppend, kindAppend, kindClear, kindAppend, kindCancel}
	if got := sent.kinds(); !reflect.DeepEqual(got, want) {
		t.Errorf("events = %v, want %v", got, want)
	}
	payloads := sent.appendPayloads()
	if got := payloads[len(payloads)-1]; got != b64([]byte{0xaa, 0xbb, 0xcc, 0xdd}) {
		t.Errorf("replay = %q, want the whole buffered onset", got)
	}
	// The buffer is left intact (a rolling window), so a later interruption
	// can put it back again.
	if got := s.prerollBytes(); !bytes.Equal(got, []byte{0xaa, 0xbb, 0xcc, 0xdd}) {
		t.Errorf("pre-roll = %x, want it intact", got)
	}
}

func TestManualModeInterruptionWithEmptyBufferSkipsReplay(t *testing.T) {
	s, sent := userAudioService(t, true, nil)
	s.handleInterruption(context.Background())
	if got := sent.kinds(); !reflect.DeepEqual(got, []string{kindClear, kindCancel}) {
		t.Errorf("events = %v, want a clear and a cancel only", got)
	}
}

func TestManualModePrerollCappedToDefaultWindow(t *testing.T) {
	s, _ := userAudioService(t, true, nil)
	// 1s at 24kHz mono 16-bit is 48000 bytes; the most recent 0.5s is kept.
	sendUserAudio(s, sampleRate, make([]byte, 48000))
	if got := len(s.prerollBytes()); got != 24000 {
		t.Errorf("pre-roll = %d bytes, want 24000", got)
	}
}

func TestManualModePrerollSizedFromVADStartSecs(t *testing.T) {
	s, _ := userAudioService(t, true, nil)
	startSecs := 0.5
	s.handleSpeechControlParams(frames.NewSpeechControlParamsFrame(&vad.Params{StartSecs: startSecs}, nil))
	// 2s of audio, more than the window, so the window is what caps it.
	sendUserAudio(s, sampleRate, make([]byte, 96000))
	want := int(24000 * 1 * 2 * (startSecs + prerollMarginSecs))
	if got := len(s.prerollBytes()); got != want {
		t.Errorf("pre-roll = %d bytes, want %d", got, want)
	}
}

func TestManualModePrerollOverridePinsValueAndIgnoresVADParams(t *testing.T) {
	pinned := 0.1
	s, _ := userAudioService(t, true, &pinned)
	s.handleSpeechControlParams(frames.NewSpeechControlParamsFrame(&vad.Params{StartSecs: 0.5}, nil))
	// 0.1s at 24kHz mono 16-bit is 4800 bytes.
	sendUserAudio(s, sampleRate, make([]byte, 48000))
	if got := len(s.prerollBytes()); got != 4800 {
		t.Errorf("pre-roll = %d bytes, want 4800", got)
	}
}

func TestServerModeAppendsWithoutBuffering(t *testing.T) {
	s, sent := userAudioService(t, false, nil)
	sendUserAudio(s, sampleRate, []byte{0xaa, 0xbb})
	sendUserAudio(s, sampleRate, []byte{0xcc, 0xdd})
	if got := sent.kinds(); !reflect.DeepEqual(got, []string{kindAppend, kindAppend}) {
		t.Errorf("events = %v, want two appends", got)
	}
	if got := s.prerollBytes(); len(got) != 0 {
		t.Errorf("pre-roll = %x, want none", got)
	}
}

func TestServerModeInterruptionDoesNotClearOrReplay(t *testing.T) {
	s, sent := userAudioService(t, false, nil)
	sendUserAudio(s, sampleRate, []byte{0xaa, 0xbb})
	s.handleInterruption(context.Background())
	for _, k := range sent.kinds() {
		if k == kindClear || k == kindCancel {
			t.Errorf("events = %v, want the input buffer left alone", sent.kinds())
		}
	}
}

// sendSecondOfUserAudio sends 1s of audio in 20ms frames, the way an input
// transport delivers it.
func sendSecondOfUserAudio(s *Service, rate int) {
	frameBytes := rate * 2 / 50
	for range 50 {
		sendUserAudio(s, rate, bytes.Repeat([]byte{0x01, 0x02}, frameBytes/2))
	}
}

// (test_pipeline_audio_is_resampled_to_24khz is ported in resample_test.go.)

func TestPrerollHoldsResampledAudio(t *testing.T) {
	s, _ := userAudioService(t, true, nil)
	sendSecondOfUserAudio(s, 16000)
	// The default 0.5s window at 24kHz mono 16-bit.
	if got := len(s.prerollBytes()); got != 24000 {
		t.Errorf("pre-roll = %d bytes, want 24000", got)
	}
}

// The tests below are not ports of upstream tests.

// With the server's turn detection off, the user's stopping commits the turn
// and asks for a response; with it on, the server does both.
func TestUserStoppedSpeakingCommitsOnlyInManualMode(t *testing.T) {
	for _, manual := range []bool{true, false} {
		s, sent := userAudioService(t, manual, nil)
		s.handleUserStoppedSpeaking(context.Background())
		var want []string
		if manual {
			want = []string{"input_audio_buffer.commit", "response.create"}
		}
		got := sent.kinds()
		if manual && !reflect.DeepEqual(got, want) || !manual && len(got) != 0 {
			t.Errorf("manual=%v: events = %v, want %v", manual, got, want)
		}
	}
}

// Paused audio does not reach the session, and travels on either way.
func TestPausedAudioIsNotSent(t *testing.T) {
	s, sent := offlineService(t, Config{StartAudioPaused: true})
	r := newRecorder()
	s.Link(r)
	ctx := context.Background()
	frame := frames.NewInputAudioRawFrame([]byte{1, 2}, sampleRate, 1)
	if err := s.ProcessFrame(ctx, frame, processor.Downstream); err != nil {
		t.Fatalf("process: %v", err)
	}
	s.SetAudioInputPaused(false)
	frame = frames.NewInputAudioRawFrame([]byte{3, 4}, sampleRate, 1)
	if err := s.ProcessFrame(ctx, frame, processor.Downstream); err != nil {
		t.Fatalf("process: %v", err)
	}
	if got := sent.appendPayloads(); !reflect.DeepEqual(got, []string{b64([]byte{3, 4})}) {
		t.Errorf("payloads = %v, want only the audio sent after resuming", got)
	}
	forwarded := 0
	r.mu.Lock()
	for _, f := range r.got {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			forwarded++
		}
	}
	r.mu.Unlock()
	if forwarded != 2 {
		t.Errorf("forwarded %d audio frames, want both", forwarded)
	}
}
