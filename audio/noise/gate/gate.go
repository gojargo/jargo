// Package gate provides a noise gate: an audio.Filter that silences a chunk of
// audio whose RMS energy falls below a threshold, suppressing low-level
// background hum and hiss between speech. It is a coarse, pure-Go complement to a
// spectral denoiser such as the rnnoise filter.
package gate

import (
	"context"
	"encoding/binary"
	"math"
	"sync"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
)

// defaultThresholdDB is the default gate level.
const defaultThresholdDB = -50.0

// Gate silences audio quieter than its threshold.
type Gate struct {
	mu          sync.Mutex
	thresholdDB float64
	gate        float64
	filtering   bool // false once a FilterEnableFrame switches gating off
}

// New builds a noise gate at thresholdDB dBFS; 0 uses -50 dBFS. Audio quieter
// than the threshold is replaced with silence.
func New(thresholdDB float64) *Gate {
	if thresholdDB == 0 {
		thresholdDB = defaultThresholdDB
	}
	return &Gate{thresholdDB: thresholdDB, filtering: true}
}

// ProcessFrame applies a runtime control frame. A FilterEnableFrame switches
// gating on or off, leaving the threshold in place so it can be switched back.
func (g *Gate) ProcessFrame(_ context.Context, f frames.FilterControlFrame) error {
	if enable, ok := f.(*frames.FilterEnableFrame); ok {
		g.mu.Lock()
		g.filtering = enable.Enable
		g.mu.Unlock()
	}
	return nil
}

// Start computes the normalized RMS gate from the threshold.
func (g *Gate) Start(context.Context, int) error {
	g.gate = math.Pow(10, g.thresholdDB/20)
	return nil
}

// Stop is a no-op.
func (g *Gate) Stop(context.Context) error { return nil }

// Filter silences pcm when its RMS energy is below the gate, else passes it
// through unchanged.
func (g *Gate) Filter(_ context.Context, pcm []byte) ([]byte, error) {
	g.mu.Lock()
	filtering := g.filtering
	g.mu.Unlock()

	n := len(pcm) / 2
	if n == 0 || !filtering {
		return pcm, nil
	}
	var sumSq float64
	for i := range n {
		s := float64(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768.0
		sumSq += s * s
	}
	if math.Sqrt(sumSq/float64(n)) < g.gate {
		return make([]byte, len(pcm)), nil
	}
	return pcm, nil
}

// Compile-time interface check.
var _ audio.Filter = (*Gate)(nil)
