package gate_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/audio/noise/gate"
)

// tone builds n samples of constant amplitude, expressed as a fraction of full
// scale. A constant signal has an RMS equal to its amplitude, which makes the
// gate's threshold easy to reason about.
func tone(amplitude float64, n int) []byte {
	b := make([]byte, 2*n)
	s := int16(amplitude * 32768)
	for i := range n {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

// dbfs converts a dBFS threshold to the normalized amplitude it corresponds to.
func dbfs(db float64) float64 { return math.Pow(10, db/20) }

func isSilence(b []byte) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}

// filter starts the gate and runs one chunk through it.
func filter(t *testing.T, g *gate.Gate, pcm []byte) []byte {
	t.Helper()
	if err := g.Start(t.Context(), 16000); err != nil {
		t.Fatalf("Start: %v", err)
	}
	out, err := g.Filter(t.Context(), pcm)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	return out
}

// TestGatesBelowThreshold is the core contract: quiet audio becomes silence,
// louder audio passes through untouched.
func TestGatesBelowThreshold(t *testing.T) {
	const threshold = -40.0
	tests := []struct {
		name      string
		amplitude float64
		wantMuted bool
	}{
		{"well below the threshold", dbfs(-60), true},
		{"just below the threshold", dbfs(-41), true},
		{"just above the threshold", dbfs(-39), false},
		{"well above the threshold", dbfs(-6), false},
		{"digital silence", 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := tone(tt.amplitude, 160)
			out := filter(t, gate.New(threshold), in)

			if got := isSilence(out); got != tt.wantMuted {
				t.Errorf("silenced = %v, want %v", got, tt.wantMuted)
			}
			if len(out) != len(in) {
				t.Errorf("length = %d, want %d; the gate must not resize the chunk", len(out), len(in))
			}
		})
	}
}

// TestPassthroughIsUnchanged checks audio above the gate is returned byte for
// byte, not re-encoded.
func TestPassthroughIsUnchanged(t *testing.T) {
	in := tone(dbfs(-10), 160)
	out := filter(t, gate.New(-40), in)
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("byte %d changed: got %d, want %d", i, out[i], in[i])
		}
	}
}

// TestDefaultThreshold checks the documented -50 dBFS default, by placing signals
// either side of it.
func TestDefaultThreshold(t *testing.T) {
	if out := filter(t, gate.New(0), tone(dbfs(-55), 160)); !isSilence(out) {
		t.Error("-55 dBFS should be gated at the -50 dBFS default")
	}
	if out := filter(t, gate.New(0), tone(dbfs(-45), 160)); isSilence(out) {
		t.Error("-45 dBFS should pass at the -50 dBFS default")
	}
}

// TestEmptyChunk checks a zero-length chunk is returned as-is rather than
// dividing by zero when computing RMS.
func TestEmptyChunk(t *testing.T) {
	out := filter(t, gate.New(-40), []byte{})
	if len(out) != 0 {
		t.Errorf("length = %d, want 0", len(out))
	}
}

// TestFilterBeforeStart pins the fail-open behavior: Start computes the gate
// level, so a filter used before it lets everything through rather than
// silencing the whole call.
func TestFilterBeforeStart(t *testing.T) {
	g := gate.New(-40)
	out, err := g.Filter(t.Context(), tone(dbfs(-80), 160))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if isSilence(out) {
		t.Error("an unstarted gate should pass audio through, not silence it")
	}
}

func TestStop(t *testing.T) {
	if err := gate.New(-40).Stop(t.Context()); err != nil {
		t.Errorf("Stop: %v", err)
	}
}
