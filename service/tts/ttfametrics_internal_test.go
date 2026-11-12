package tts

import (
	"encoding/binary"
	"math"
	"testing"
	"time"
)

const ttfaTestRate = 16000

func meterSilence(seconds float64) []byte { return make([]byte, 2*int(ttfaTestRate*seconds)) }

// meterTone is a sustained 220 Hz tone: continuous energy, unlike an isolated
// blip.
func meterTone(seconds float64) []byte {
	n := int(ttfaTestRate * seconds)
	b := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/ttfaTestRate) * 5000)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

// measuredMeter is a meter whose TTFB was measured as ttfb, which arms the TTFA
// scan.
func measuredMeter(ttfb time.Duration) *ttfaMeter {
	m := &ttfaMeter{report: true}
	m.stopTTFB(ttfb)
	return m
}

func nearly(got, want, tol time.Duration) bool { return (got - want).Abs() <= tol }

// Ported from upstream (TestTTFAMetrics). The leading silence spans two chunks,
// and TTFA is TTFB plus all of it.
func TestTTFASpansMultipleChunks(t *testing.T) {
	m := measuredMeter(200 * time.Millisecond)
	if m.processTTFA(meterSilence(0.3), ttfaTestRate, 1) {
		t.Fatal("TTFA reported on a chunk of silence")
	}
	if !m.processTTFA(append(meterSilence(0.1), meterTone(0.3)...), ttfaTestRate, 1) {
		t.Fatal("no TTFA once the tone arrived")
	}
	if !nearly(m.ttfa, 600*time.Millisecond, 20*time.Millisecond) {
		t.Errorf("ttfa = %s, want about 600ms", m.ttfa)
	}
	if m.ttfb != 200*time.Millisecond {
		t.Errorf("ttfb = %s, want 200ms", m.ttfb)
	}
	if !nearly(m.leadingSilence, 400*time.Millisecond, 20*time.Millisecond) {
		t.Errorf("leading silence = %s, want about 400ms", m.leadingSilence)
	}
	if m.ttfa != m.ttfb+m.leadingSilence {
		t.Errorf("ttfa %s is not ttfb %s plus the leading silence %s", m.ttfa, m.ttfb, m.leadingSilence)
	}
}

// Ported from upstream. Later audio of the same response does not report again.
func TestTTFAReportedOncePerResponse(t *testing.T) {
	m := measuredMeter(100 * time.Millisecond)
	if !m.processTTFA(meterTone(0.2), ttfaTestRate, 1) {
		t.Fatal("no TTFA for a tone")
	}
	if m.processTTFA(meterTone(0.2), ttfaTestRate, 1) {
		t.Error("TTFA reported a second time for the same response")
	}
}

// Ported from upstream. With no TTFB measured the scan is never armed.
func TestNoTTFAWithoutTTFB(t *testing.T) {
	m := &ttfaMeter{report: true}
	if m.processTTFA(meterTone(0.2), ttfaTestRate, 1) {
		t.Error("TTFA reported with no TTFB measured")
	}
}

// Ported from upstream. A response that is all silence reports nothing.
func TestAllSilenceResponseEmitsNoTTFA(t *testing.T) {
	m := measuredMeter(100 * time.Millisecond)
	if m.processTTFA(meterSilence(0.5), ttfaTestRate, 1) {
		t.Error("TTFA reported for silence")
	}
}

// Ported from upstream. Each response is measured afresh: its own TTFB rearms
// the scan, with the buffer and the silence reset.
func TestTTFARearmedEachResponse(t *testing.T) {
	first := measuredMeter(100 * time.Millisecond)
	if !first.processTTFA(meterTone(0.2), ttfaTestRate, 1) {
		t.Fatal("no TTFA for the first response")
	}
	if !nearly(first.ttfa, 100*time.Millisecond, 20*time.Millisecond) {
		t.Errorf("first ttfa = %s, want about 100ms", first.ttfa)
	}
	second := measuredMeter(300 * time.Millisecond)
	if !second.processTTFA(append(meterSilence(0.2), meterTone(0.2)...), ttfaTestRate, 1) {
		t.Fatal("no TTFA for the second response")
	}
	if !nearly(second.ttfa, 500*time.Millisecond, 20*time.Millisecond) {
		t.Errorf("second ttfa = %s, want about 500ms", second.ttfa)
	}
}
