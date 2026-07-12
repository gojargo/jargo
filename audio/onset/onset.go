// Package onset detects the first audible sample in a stream of PCM audio. A TTS
// service measures the leading silence it pads onto the start of its response by
// feeding audio here as it streams; the leading-silence duration turns
// time-to-first-byte into time-to-first-audio (the perceived latency).
package onset

import "math"

// Detection parameters. Short-time RMS energy is measured over a frameMs window
// hopping every hopMs; an onset is the first run of windows whose energy stays
// above thresholdDB for at least minVoicedMs. Working on energy rather than
// per-sample amplitude rejects noise-floor blips, and the minimum duration
// rejects brief transients.
const (
	frameMs     = 10.0
	hopMs       = 1.0
	thresholdDB = -40.0 // energy gate in dBFS; above TTS noise-floor padding.
	minVoicedMs = 50.0
)

// Detect returns the per-channel sample index of the first sustained audible
// onset in pcm — 16-bit signed little-endian samples, numChannels interleaved —
// or -1 when no confirmed onset is present yet. Feed it a growing buffer as
// audio streams in and stop once it returns a non-negative index.
func Detect(pcm []byte, sampleRate, numChannels int) int {
	if sampleRate <= 0 {
		return -1
	}
	total := len(pcm) / 2
	if total == 0 {
		return -1
	}

	// Downmix to a mono signal normalized to [-1, 1], keeping the sign so that
	// mean-removed RMS measures energy correctly.
	var mono []float64
	if numChannels > 1 {
		usable := total - (total % numChannels)
		mono = make([]float64, usable/numChannels)
		for i := 0; i < usable; i += numChannels {
			var sum float64
			for c := range numChannels {
				sum += float64(sampleAt(pcm, i+c))
			}
			mono[i/numChannels] = sum / float64(numChannels) / 32768.0
		}
	} else {
		mono = make([]float64, total)
		for i := range total {
			mono[i] = float64(sampleAt(pcm, i)) / 32768.0
		}
	}

	n := len(mono)
	frame := roundPos(float64(sampleRate) * frameMs / 1000.0)
	hop := roundPos(float64(sampleRate) * hopMs / 1000.0)
	if n < frame {
		// Not enough audio for even one window; wait for more.
		return -1
	}

	gate := math.Pow(10, thresholdDB/20.0) // normalized RMS gate (-40 dBFS = 0.01).

	// Edge-pad so each window stays centered on its hop position and a constant
	// start isn't read as a step (which would carry energy).
	pad := frame / 2
	padded := make([]float64, n+2*pad)
	for i := range pad {
		padded[i] = mono[0]
		padded[pad+n+i] = mono[n-1]
	}
	copy(padded[pad:pad+n], mono)

	// Onset is the start of the first run of active windows lasting at least
	// min_voiced_ms. A window spreads a lone sample across ~frame_ms, so
	// minVoicedMs exceeding frameMs tells a real onset from a blip.
	minWindows := roundPos(minVoicedMs / hopMs)
	run := 0
	for j := 0; j*hop+frame <= len(padded); j++ {
		start := j * hop
		var mean float64
		for k := range frame {
			mean += padded[start+k]
		}
		mean /= float64(frame)
		var ss float64
		for k := range frame {
			d := padded[start+k] - mean
			ss += d * d
		}
		if math.Sqrt(ss/float64(frame)) > gate {
			run++
			if run >= minWindows {
				return (j - minWindows + 1) * hop
			}
		} else {
			run = 0
		}
	}
	return -1
}

// sampleAt reads the i-th 16-bit little-endian sample from pcm.
func sampleAt(pcm []byte, i int) int16 {
	return int16(uint16(pcm[2*i]) | uint16(pcm[2*i+1])<<8)
}

// roundPos rounds v to the nearest integer, clamped to at least 1.
func roundPos(v float64) int {
	n := int(math.Round(v))
	if n < 1 {
		return 1
	}
	return n
}
