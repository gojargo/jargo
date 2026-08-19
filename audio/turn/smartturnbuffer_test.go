package turn

import (
	"encoding/binary"
	"testing"
	"time"
)

// bufferTestRate is the rate every buffer test runs at, so a 320-sample chunk is
// exactly 20 ms of audio.
const bufferTestRate = 16000

// chunkSamples is 20 ms at bufferTestRate.
const chunkSamples = 320

// recordingPredictor stands in for the model. It records the segment it was
// handed, so a test can assert on exactly what the model would have seen, and
// always reports the turn complete.
type recordingPredictor struct {
	captured []float32
	calls    int
}

func (p *recordingPredictor) predictEndpoint(audio []float32) (bool, float64, error) {
	p.captured = append([]float32(nil), audio...)
	p.calls++
	return true, 0.99, nil
}

// newBufferTestBase builds a base analyzer on a recording predictor, at the test
// sample rate.
func newBufferTestBase(t *testing.T, params Params) (*smartTurnBase, *recordingPredictor) {
	t.Helper()
	p := &recordingPredictor{}
	b := newSmartTurnBase(p, params)
	b.SetSampleRate(bufferTestRate)
	return b, p
}

// pcmOf builds n samples of little-endian PCM all holding value, the wire format
// AppendAudio takes.
func pcmOf(value int16, n int) []byte {
	pcm := make([]byte, n*2)
	for i := range n {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(value))
	}
	return pcm
}

// TestSmartTurnBufferKeepsSamplesAsInt16 covers where the conversion happens.
// The buffer holds the samples as they arrived, and the float32 conversion is
// deferred to the segment built once per turn, rather than run over every 20 ms
// chunk on the way in.
func TestSmartTurnBufferKeepsSamplesAsInt16(t *testing.T) {
	b, _ := newBufferTestBase(t, DefaultParams())

	want := []int16{100, -200, 300, -400}
	pcm := make([]byte, len(want)*2)
	for i, s := range want {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(s))
	}
	b.AppendAudio(pcm, true)

	if len(b.buffer) != 1 {
		t.Fatalf("buffered %d chunks, want 1", len(b.buffer))
	}
	got := b.buffer[0].samples
	if len(got) != len(want) {
		t.Fatalf("buffered %d samples, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("sample %d = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestSmartTurnSegmentIsNormalizedFloat32 covers the conversion itself: the
// model is handed float32 in [-1, 1], scaled by 32768. That divisor is a power
// of two, so the conversion is exact rather than approximate, and this asserts
// the exact value to keep it that way.
func TestSmartTurnSegmentIsNormalizedFloat32(t *testing.T) {
	b, p := newBufferTestBase(t, DefaultParams())

	const level = 16000
	for range 4 {
		b.AppendAudio(pcmOf(level, chunkSamples), true)
	}

	if _, _, err := b.processSpeechSegment(); err != nil {
		t.Fatalf("process segment: %v", err)
	}
	if p.calls != 1 {
		t.Fatalf("the model was run %d times, want 1", p.calls)
	}
	if len(p.captured) != 4*chunkSamples {
		t.Fatalf("the model saw %d samples, want %d", len(p.captured), 4*chunkSamples)
	}

	const want = float32(level) / 32768.0
	for i, got := range p.captured {
		if got != want {
			t.Fatalf("sample %d = %v, want exactly %v", i, got, want)
		}
		if got > 1 || got < -1 {
			t.Fatalf("sample %d = %v, outside [-1, 1]", i, got)
		}
	}
}

// TestSmartTurnTrimsThePreSpeechBuffer covers a line that stays open with nobody
// speaking on it. Nothing closes a turn that never opened, so without the trim
// the buffer would grow for as long as the call lasts.
func TestSmartTurnTrimsThePreSpeechBuffer(t *testing.T) {
	// A short window, so the trim can be observed without a slow test. Upstream
	// uses 0.1 + 0.2 + 0.5; the shape is the same, the scale is smaller.
	params := Params{PreSpeechMs: 50, StopSecs: 0.05, MaxDurationSecs: 0.1}
	b, _ := newBufferTestBase(t, params)

	const appends = 30
	for range appends {
		b.AppendAudio(pcmOf(0, chunkSamples), false)
		// The trim is by arrival time, so the chunks have to arrive apart.
		time.Sleep(10 * time.Millisecond)
	}

	// The window is 0.2 s, so at one chunk per 10 ms it holds about 20. Anything
	// short of every chunk ever appended proves the trim ran.
	if len(b.buffer) >= appends {
		t.Errorf("buffered %d of %d chunks with nobody speaking: the pre-speech trim never ran",
			len(b.buffer), appends)
	}
}

// TestSmartTurnClearKillsTheStaleSilenceTimer covers a turn ended from outside,
// by a maximum-speech cap or a mute, while the user was still speaking. That
// leaves the analyzer mid-turn with silence accumulating against the safety net.
// Clearing has to stop that count, or the net fires later on ambient silence and
// ends a turn that is no longer running, in the middle of the bot's reply.
func TestSmartTurnClearKillsTheStaleSilenceTimer(t *testing.T) {
	params := DefaultParams()
	params.StopSecs = 0.2
	b, _ := newBufferTestBase(t, params)

	for range 4 {
		b.AppendAudio(pcmOf(16000, chunkSamples), true)
	}
	if !b.SpeechTriggered() {
		t.Fatal("four chunks of speech did not open a turn")
	}

	b.Clear()
	if b.SpeechTriggered() {
		t.Fatal("clearing left the turn open")
	}

	// 30 chunks is 600 ms, three times the safety net. None of it belongs to a
	// turn, so none of it may end one.
	for i := range 30 {
		if state := b.AppendAudio(pcmOf(0, chunkSamples), false); state != Incomplete {
			t.Fatalf("silence chunk %d after the clear reported %v, want incomplete: "+
				"the stale silence timer fired a turn that was already over", i, state)
		}
	}
}

// TestSmartTurnClearDropsAudioFromTheEndedTurn is the other half of that
// contract. The audio buffered for the turn that was ended from outside must not
// reach the model when the next turn is analyzed, or the model scores an
// utterance spliced together from two different turns.
func TestSmartTurnClearDropsAudioFromTheEndedTurn(t *testing.T) {
	b, p := newBufferTestBase(t, DefaultParams())

	const (
		ended = 1000
		next  = 2000
	)
	for range 4 {
		b.AppendAudio(pcmOf(ended, chunkSamples), true)
	}
	b.Clear()

	for range 4 {
		b.AppendAudio(pcmOf(next, chunkSamples), true)
	}
	if _, _, err := b.processSpeechSegment(); err != nil {
		t.Fatalf("process segment: %v", err)
	}

	if len(p.captured) != 4*chunkSamples {
		t.Fatalf("the model saw %d samples, want only the %d from the new turn",
			len(p.captured), 4*chunkSamples)
	}
	const wantSample = float32(next) / 32768.0
	for i, got := range p.captured {
		if got != wantSample {
			t.Fatalf("sample %d = %v, want %v: audio from the turn that was cleared reached the model",
				i, got, wantSample)
		}
	}
}

// TestSmartTurnSilenceClosesTheTurn covers the safety net itself: with a turn
// open, silence past the stop window ends it without the model being consulted,
// and clears the analyzer behind it.
func TestSmartTurnSilenceClosesTheTurn(t *testing.T) {
	params := DefaultParams()
	params.StopSecs = 0.2
	b, p := newBufferTestBase(t, params)

	b.AppendAudio(pcmOf(16000, chunkSamples), true)

	// 20 ms per chunk, so the tenth crosses the 200 ms net.
	var state EndOfTurnState
	appended := 0
	for range 20 {
		appended++
		state = b.AppendAudio(pcmOf(0, chunkSamples), false)
		if state == Complete {
			break
		}
	}

	if state != Complete {
		t.Fatal("silence past the stop window never closed the turn")
	}
	if appended != 10 {
		t.Errorf("the turn closed after %d chunks of silence, want 10 (200 ms at 20 ms a chunk)", appended)
	}
	if p.calls != 0 {
		t.Errorf("the model was run %d times, want 0: the safety net decides without it", p.calls)
	}
	if b.speechTriggered || len(b.buffer) != 0 {
		t.Error("closing the turn left the analyzer holding state from it")
	}
}
