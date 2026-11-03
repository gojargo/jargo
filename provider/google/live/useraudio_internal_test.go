package live

import (
	"bytes"
	"encoding/base64"
	"testing"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
)

// Ports of tests/test_gemini_live_user_audio.py. They drive the service's audio
// and turn handlers against a recording session and assert on the realtime
// input it sends: with server-side detection off, audio is kept while the user
// is not in a turn and replayed after the turn starts; with it on, audio is
// streamed continuously.

// audioService is a service on a ready, seeded session, with server-side
// detection off when vadDisabled, and the pre-roll pinned when prerollSecs is
// set.
func audioService(t *testing.T, vadDisabled bool, prerollSecs *float64) (*Service, *recordingSession) {
	t.Helper()
	srv := newRecordingSession(t)
	cfg := Config{APIKey: "test-key", BaseURL: wsURL(srv), UserAudioPrerollSecs: prerollSecs}
	if vadDisabled {
		off := true
		cfg.VAD = &VADParams{Disabled: &off}
	}
	s := startedServiceBuilt(t, New(cfg))
	s.readyForInput.Store(true)
	return s, srv
}

// kinds are the realtime inputs the session received, in order: "audio",
// "activity_start" or "activity_end".
func (r *recordingSession) kinds() []string {
	var out []string
	for _, m := range r.withKey("realtimeInput") {
		switch {
		case m["audio"] != nil:
			out = append(out, "audio")
		case m["activityStart"] != nil:
			out = append(out, "activity_start")
		case m["activityEnd"] != nil:
			out = append(out, "activity_end")
		}
	}
	return out
}

// audioPayloads are the audio chunks the session received, in order.
func (r *recordingSession) audioPayloads(t *testing.T) [][]byte {
	t.Helper()
	var out [][]byte
	for _, m := range r.withKey("realtimeInput") {
		a, ok := m["audio"].(map[string]any)
		if !ok {
			continue
		}
		data, _ := a["data"].(string)
		pcm, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			t.Fatalf("audio payload: %v", err)
		}
		out = append(out, pcm)
	}
	return out
}

func totalLen(chunks [][]byte) int {
	n := 0
	for _, c := range chunks {
		n += len(c)
	}
	return n
}

func equalKinds(got []string, want ...string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func (s *Service) prerollLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.preroll)
}

// While idle, audio is buffered as pre-roll and not streamed to the service.
func TestManualVADBuffersAudioUntilTurnStarts(t *testing.T) {
	s, srv := audioService(t, true, nil)

	s.sendAudio([]byte{0xaa, 0xbb}, 16000, 1)
	s.sendAudio([]byte{0xcc, 0xdd}, 16000, 1)
	settle()

	if got := srv.kinds(); len(got) != 0 {
		t.Errorf("sent %v, want idle audio buffered, not sent", got)
	}
	s.mu.Lock()
	buffered := append([]byte(nil), s.preroll...)
	s.mu.Unlock()
	if !bytes.Equal(buffered, []byte{0xaa, 0xbb, 0xcc, 0xdd}) {
		t.Errorf("pre-roll = %x, want aabbccdd", buffered)
	}
}

// The activity start is followed by a replay of the buffered pre-roll audio.
func TestManualVADFlushesPrerollAfterActivityStart(t *testing.T) {
	s, srv := audioService(t, true, nil)

	s.sendAudio([]byte{0xaa, 0xbb}, 16000, 1)
	s.sendAudio([]byte{0xcc, 0xdd}, 16000, 1)
	s.handleUserStartedSpeaking(t.Context())
	settle()

	if got := srv.kinds(); !equalKinds(got, "activity_start", "audio") {
		t.Errorf("sent %v, want the activity start and then the pre-roll", got)
	}
	if got := srv.audioPayloads(t); len(got) != 1 || !bytes.Equal(got[0], []byte{0xaa, 0xbb, 0xcc, 0xdd}) {
		t.Errorf("audio = %x, want the buffered onset replayed", got)
	}
	if n := s.prerollLen(); n != 0 {
		t.Errorf("pre-roll holds %d bytes, want it cleared so it is not replayed again", n)
	}
}

// Once the turn has started, audio is streamed in-band.
func TestManualVADStreamsAudioDuringTurn(t *testing.T) {
	s, srv := audioService(t, true, nil)

	s.handleUserStartedSpeaking(t.Context())
	settle()
	if got := srv.kinds(); !equalKinds(got, "activity_start") {
		t.Fatalf("sent %v, want only the activity start: nothing was buffered", got)
	}

	s.sendAudio([]byte{0x11, 0x22}, 16000, 1)
	s.sendAudio([]byte{0x33, 0x44}, 16000, 1)
	settle()

	if got := srv.kinds(); !equalKinds(got, "activity_start", "audio", "audio") {
		t.Errorf("sent %v, want the audio streamed", got)
	}
	got := srv.audioPayloads(t)
	if len(got) != 2 || !bytes.Equal(got[0], []byte{0x11, 0x22}) || !bytes.Equal(got[1], []byte{0x33, 0x44}) {
		t.Errorf("audio = %x, want both chunks", got)
	}
}

// The turn stopping sends the activity end.
func TestManualVADSendsActivityEndOnTurnStop(t *testing.T) {
	s, srv := audioService(t, true, nil)

	s.handleUserStartedSpeaking(t.Context())
	s.sendAudio([]byte{0x11, 0x22}, 16000, 1)
	s.handleUserStoppedSpeaking(t.Context())
	settle()

	if got := srv.kinds(); !equalKinds(got, "activity_start", "audio", "activity_end") {
		t.Errorf("sent %v, want the turn bracketed by activity signals", got)
	}
}

// Before the detection's parameters are known, the pre-roll keeps half a second.
func TestManualVADPrerollCappedToDefaultWindow(t *testing.T) {
	s, srv := audioService(t, true, nil)

	// One second at 16 kHz mono, 16-bit, is 32000 bytes; half a second is 16000.
	s.sendAudio(make([]byte, 32000), 16000, 1)
	if n := s.prerollLen(); n != 16000 {
		t.Errorf("pre-roll holds %d bytes, want 16000", n)
	}

	s.handleUserStartedSpeaking(t.Context())
	settle()
	if n := totalLen(srv.audioPayloads(t)); n != 16000 {
		t.Errorf("replayed %d bytes, want 16000", n)
	}
}

// The pipeline's detection parameters size the pre-roll to its start delay plus
// a margin.
func TestManualVADPrerollSizedFromVADStartSecs(t *testing.T) {
	s, srv := audioService(t, true, nil)

	const startSecs = 0.5
	s.handleSpeechControlParams(frames.NewSpeechControlParamsFrame(&vad.Params{StartSecs: startSecs}, nil))

	// Two seconds, comfortably more than the window, so the buffer is capped to
	// the window rather than by how much was sent.
	s.sendAudio(make([]byte, 64000), 16000, 1)
	want := int(16000 * 1 * 2 * (startSecs + prerollMarginSecs))
	if n := s.prerollLen(); n != want {
		t.Errorf("pre-roll holds %d bytes, want %d", n, want)
	}

	s.handleUserStartedSpeaking(t.Context())
	settle()
	if n := totalLen(srv.audioPayloads(t)); n != want {
		t.Errorf("replayed %d bytes, want %d", n, want)
	}
}

// A pre-roll pinned in the Config wins over the detection's parameters.
func TestManualVADPrerollOverridePinsValueAndIgnoresVADParams(t *testing.T) {
	pinned := 0.1
	s, srv := audioService(t, true, &pinned)

	s.handleSpeechControlParams(frames.NewSpeechControlParamsFrame(&vad.Params{StartSecs: 0.5}, nil))

	// A tenth of a second at 16 kHz mono, 16-bit, is 3200 bytes.
	s.sendAudio(make([]byte, 32000), 16000, 1)
	if n := s.prerollLen(); n != 3200 {
		t.Errorf("pre-roll holds %d bytes, want 3200", n)
	}

	s.handleUserStartedSpeaking(t.Context())
	settle()
	if n := totalLen(srv.audioPayloads(t)); n != 3200 {
		t.Errorf("replayed %d bytes, want 3200", n)
	}
}

// With server-side detection, audio is streamed continuously, even while idle.
func TestServerVADStreamsAudioWhenIdle(t *testing.T) {
	s, srv := audioService(t, false, nil)

	s.sendAudio([]byte{0xaa, 0xbb}, 16000, 1)
	s.sendAudio([]byte{0xcc, 0xdd}, 16000, 1)
	settle()

	if got := srv.kinds(); !equalKinds(got, "audio", "audio") {
		t.Errorf("sent %v, want both chunks streamed", got)
	}
	got := srv.audioPayloads(t)
	if len(got) != 2 || !bytes.Equal(got[0], []byte{0xaa, 0xbb}) || !bytes.Equal(got[1], []byte{0xcc, 0xdd}) {
		t.Errorf("audio = %x, want both chunks", got)
	}
	if n := s.prerollLen(); n != 0 {
		t.Errorf("pre-roll holds %d bytes, want none kept with server-side detection", n)
	}
}

// With server-side detection, the client sends no activity signals.
func TestServerVADDoesNotSendActivityMarkers(t *testing.T) {
	s, srv := audioService(t, false, nil)

	s.handleUserStartedSpeaking(t.Context())
	s.handleUserStoppedSpeaking(t.Context())
	settle()

	for _, k := range srv.kinds() {
		if k == "activity_start" || k == "activity_end" {
			t.Errorf("sent %s, want no activity signals", k)
		}
	}
}

// The tests below are not ports of upstream tests.

// Paused audio does not reach the session.
func TestPausedAudioIsNotSent(t *testing.T) {
	s, srv := audioService(t, false, nil)

	s.SetAudioInputPaused(true)
	s.sendAudio([]byte{1, 2}, 16000, 1)
	s.SetAudioInputPaused(false)
	s.sendAudio([]byte{3, 4}, 16000, 1)
	settle()

	if got := srv.audioPayloads(t); len(got) != 1 || !bytes.Equal(got[0], []byte{3, 4}) {
		t.Errorf("audio = %x, want only what was sent while not paused", got)
	}
}

// The user's text goes in as realtime input.
func TestUserTextIsSentAsRealtimeInput(t *testing.T) {
	s, srv := audioService(t, false, nil)

	s.sendUserText(t.Context(), "hello")
	settle()

	var texts []any
	for _, m := range srv.withKey("realtimeInput") {
		if v, ok := m["text"]; ok {
			texts = append(texts, v)
		}
	}
	if len(texts) != 1 || texts[0] != "hello" {
		t.Errorf("text inputs = %v, want the user's text", texts)
	}
}
