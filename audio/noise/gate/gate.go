// Package gate provides a noise gate: an audio.Filter that silences a chunk of
// audio whose RMS energy falls below a threshold, suppressing low-level
// background hum and hiss between speech. It is a coarse, pure-Go complement to a
// spectral denoiser such as the rnnoise filter.
package gate

import (
	"context"
	"encoding/binary"
	"math"

	"github.com/gojargo/jargo/audio"
)

// defaultThresholdDB is the default gate level.
const defaultThresholdDB = -50.0

// Gate silences audio quieter than its threshold.
type Gate struct {
	thresholdDB float64
	gate        float64
}

// New builds a noise gate at thresholdDB dBFS; 0 uses -50 dBFS. Audio quieter
// than the threshold is replaced with silence.
func New(thresholdDB float64) *Gate {
	if thresholdDB == 0 {
		thresholdDB = defaultThresholdDB
	}
	return &Gate{thresholdDB: thresholdDB}
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
	n := len(pcm) / 2
	if n == 0 {
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
