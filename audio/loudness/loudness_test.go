package loudness_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/audio/loudness"
)

// The expected values below were produced by the reference implementation the
// upstream project measures loudness with (pyloudnorm), through the same call it
// makes: the whole chunk as one gating block, samples at their 16-bit
// magnitudes, and the result normalized from the -20..80 range.
//
// They are what makes this a port rather than an interpretation. A filter
// coefficient or a gating bound that is subtly wrong still produces plausible
// numbers, and only a comparison against the reference catches it.
func TestVolumeMatchesReference(t *testing.T) {
	tests := []struct {
		rate int
		kind string
		freq int
		amp  int
		want float64
	}{
		{rate: 8000, kind: "silence", freq: 0, amp: 0, want: 0.000000000000},
		{rate: 8000, kind: "sine", freq: 80, amp: 100, want: 0.538289631267},
		{rate: 8000, kind: "sine", freq: 80, amp: 1000, want: 0.738855249208},
		{rate: 8000, kind: "sine", freq: 80, amp: 8000, want: 0.919515615824},
		{rate: 8000, kind: "sine", freq: 80, amp: 32000, want: 1.000000000000},
		{rate: 8000, kind: "sine", freq: 440, amp: 100, want: 0.562013032542},
		{rate: 8000, kind: "sine", freq: 440, amp: 1000, want: 0.762509718245},
		{rate: 8000, kind: "sine", freq: 440, amp: 8000, want: 0.943171647091},
		{rate: 8000, kind: "sine", freq: 440, amp: 32000, want: 1.000000000000},
		{rate: 8000, kind: "sine", freq: 3000, amp: 100, want: 0.602273030137},
		{rate: 8000, kind: "sine", freq: 3000, amp: 1000, want: 0.802703206665},
		{rate: 8000, kind: "sine", freq: 3000, amp: 8000, want: 0.983321204063},
		{rate: 8000, kind: "sine", freq: 3000, amp: 32000, want: 1.000000000000},
		{rate: 16000, kind: "silence", freq: 0, amp: 0, want: 0.000000000000},
		{rate: 16000, kind: "sine", freq: 80, amp: 100, want: 0.538506720227},
		{rate: 16000, kind: "sine", freq: 80, amp: 1000, want: 0.739015876421},
		{rate: 16000, kind: "sine", freq: 80, amp: 8000, want: 0.919677773371},
		{rate: 16000, kind: "sine", freq: 80, amp: 32000, want: 1.000000000000},
		{rate: 16000, kind: "sine", freq: 440, amp: 100, want: 0.562178937597},
		{rate: 16000, kind: "sine", freq: 440, amp: 1000, want: 0.762695158972},
		{rate: 16000, kind: "sine", freq: 440, amp: 8000, want: 0.943359836132},
		{rate: 16000, kind: "sine", freq: 440, amp: 32000, want: 1.000000000000},
		{rate: 16000, kind: "sine", freq: 3000, amp: 100, want: 0.600796813208},
		{rate: 16000, kind: "sine", freq: 3000, amp: 1000, want: 0.801165701265},
		{rate: 16000, kind: "sine", freq: 3000, amp: 8000, want: 0.981829178718},
		{rate: 16000, kind: "sine", freq: 3000, amp: 32000, want: 1.000000000000},
	}

	for _, tt := range tests {
		name := tt.kind
		if tt.kind == "sine" {
			name = sineName(tt.freq, tt.amp)
		}
		t.Run(rateName(tt.rate)+"/"+name, func(t *testing.T) {
			pcm := signal(tt.kind, tt.rate, tt.freq, tt.amp)
			got := loudness.Volume(pcm, tt.rate)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("Volume() = %.12f, want %.12f (reference)", got, tt.want)
			}
		})
	}
}

// signal builds 20 ms of mono 16-bit PCM the same way the reference vectors were
// built: a sine truncated toward zero into 16-bit samples, or silence.
func signal(kind string, rate, freq, amp int) []byte {
	n := rate / 50 // 20 ms
	samples := make([]int16, n)
	if kind == "sine" {
		for i := range samples {
			t := float64(i) / float64(rate)
			samples[i] = int16(float64(amp) * math.Sin(2*math.Pi*float64(freq)*t))
		}
	}
	pcm := make([]byte, n*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(s))
	}
	return pcm
}

func rateName(rate int) string {
	if rate == 8000 {
		return "8kHz"
	}
	return "16kHz"
}

func sineName(freq, amp int) string {
	return "sine" + itoa(freq) + "Hz@" + itoa(amp)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b []byte
	for v > 0 {
		b = append([]byte{byte('0' + v%10)}, b...)
		v /= 10
	}
	return string(b)
}

// TestVolumeIsWeightedNotRaw checks the measurement is weighted for how the ear
// hears rather than being raw amplitude: the same amplitude down at a rumble
// reads quieter than up where speech sits.
func TestVolumeIsWeightedNotRaw(t *testing.T) {
	const rate, amp = 16000, 3000
	low := loudness.Volume(signal("sine", rate, 80, amp), rate)
	speech := loudness.Volume(signal("sine", rate, 440, amp), rate)
	high := loudness.Volume(signal("sine", rate, 3000, amp), rate)

	if !(low < speech && speech < high) {
		t.Errorf("volumes at one amplitude: 80Hz=%.4f 440Hz=%.4f 3kHz=%.4f, want increasing", low, speech, high)
	}
}

func TestSmooth(t *testing.T) {
	// A reading is pulled toward the new value by the factor, not replaced.
	if got := loudness.Smooth(1, 0, 0.2); math.Abs(got-0.2) > 1e-12 {
		t.Errorf("Smooth(1, 0, 0.2) = %v, want 0.2", got)
	}
	if got := loudness.Smooth(0, 1, 0.2); math.Abs(got-0.8) > 1e-12 {
		t.Errorf("Smooth(0, 1, 0.2) = %v, want 0.8", got)
	}
}

func TestNormalizeClamps(t *testing.T) {
	if got := loudness.Normalize(math.Inf(-1), -20, 80); got != 0 {
		t.Errorf("Normalize(-Inf) = %v, want 0", got)
	}
	if got := loudness.Normalize(200, -20, 80); got != 1 {
		t.Errorf("Normalize(200) = %v, want 1", got)
	}
	if got := loudness.Normalize(30, -20, 80); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("Normalize(30) = %v, want 0.5", got)
	}
}
