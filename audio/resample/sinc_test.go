package resample_test

// Ported from upstream's streaming and one-shot sinc resampler tests: the
// output of a stream keeps pace with its input, chunking does not change it,
// the passband is kept and aliasing and images are removed, and a stream is
// flushed, reset and cleared as a stream ends.

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
)

//nolint:gochecknoglobals // the rate pairs every case runs over
var ratePairs = [][2]int{
	{24000, 8000},
	{24000, 16000},
	{48000, 8000},
	{44100, 16000},
	{8000, 24000},
	{16000, 24000},
	{16000, 44100},
}

const amplitude = 16000

// noisePCM is a second of deterministic Gaussian noise at rate.
func noisePCM(rate int) []int16 {
	rng := rand.New(rand.NewPCG(0, 0)) //nolint:gosec // a fixed test signal
	out := make([]int16, rate)
	for i := range out {
		out[i] = int16(rng.NormFloat64() * 4000)
	}
	return out
}

// tonePCM is secs of a tone at freq, truncated to int16 as upstream builds it.
func tonePCM(freq float64, rate int, secs float64) []int16 {
	n := int(float64(rate) * secs)
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(math.Sin(2*math.Pi*freq*float64(i)/float64(rate)) * amplitude)
	}
	return out
}

func pcmBytes(s []int16) []byte {
	b := make([]byte, 2*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return b
}

func pcmSamples(b []byte) []int16 {
	s := make([]int16, len(b)/2)
	for i := range s {
		s[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return s
}

// streamChunks feeds samples to r in chunks of size and returns each output.
func streamChunks(t *testing.T, r *resample.Resampler, samples []int16, size int) [][]int16 {
	t.Helper()
	var out [][]int16
	for k := 0; k+size <= len(samples); k += size {
		out = append(out, pcmSamples(r.Process(pcmBytes(samples[k:k+size]))))
	}
	return out
}

func newStream(t *testing.T, in, out int, cfg resample.Config) *resample.Resampler {
	t.Helper()
	r, err := resample.NewWithConfig(in, out, 1, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(r.Close)
	return r
}

func concat(chunks [][]int16) []int16 {
	var out []int16
	for _, c := range chunks {
		out = append(out, c...)
	}
	return out
}

// levelDB is the level of freq in samples relative to the test tones'
// amplitude, from a Blackman-windowed spectrum around the nearest bin.
func levelDB(samples []float64, rate int, freq float64) float64 {
	n := len(samples)
	windowSum := 0.0
	w := make([]float64, n)
	for i := range n {
		x := 2 * math.Pi * float64(i) / float64(n-1)
		w[i] = 0.42 - 0.5*math.Cos(x) + 0.08*math.Cos(2*x)
		windowSum += w[i]
	}
	k := int(math.Round(freq * float64(n) / float64(rate)))
	peak := 1e-12
	for bin := max(k-2, 0); bin <= k+2; bin++ {
		var acc complex128
		for i, v := range samples {
			acc += complex(v*w[i], 0) * cmplx.Exp(complex(0, -2*math.Pi*float64(bin)*float64(i)/float64(n)))
		}
		peak = max(peak, cmplx.Abs(acc)/(windowSum/2))
	}
	return 20 * math.Log10(peak/amplitude)
}

// test_every_chunk_keeps_pace_with_its_input: ported as is.
func TestSincStreamEveryChunkKeepsPace(t *testing.T) {
	for _, p := range ratePairs {
		in, out := p[0], p[1]
		r := newStream(t, in, out, resample.Config{})
		for i, c := range streamChunks(t, r, noisePCM(in), in/25) {
			if len(c) != out/25 {
				t.Errorf("%d->%d chunk %d: %d samples, want %d", in, out, i, len(c), out/25)
				break
			}
		}
	}
}

// test_streamed_output_matches_resampling_at_once and
// test_any_chunk_sizes_give_the_same_output: ported as is.
func TestSincStreamChunkingDoesNotChangeTheOutput(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1)) //nolint:gosec // fixed test chunking
	for _, p := range ratePairs {
		in, out := p[0], p[1]
		samples := noisePCM(in)
		atOnce := pcmSamples(newStream(t, in, out, resample.Config{}).Process(pcmBytes(samples)))

		even := concat(streamChunks(t, newStream(t, in, out, resample.Config{}), samples, in/25))
		if !equalSamples(even, atOnce[:len(even)]) || len(even) != len(atOnce) {
			t.Errorf("%d->%d: streamed in equal chunks differs from at once", in, out)
		}

		r := newStream(t, in, out, resample.Config{})
		var got []int16
		for pos := 0; ; {
			size := 1 + rng.IntN(in/10-1)
			if pos+size > len(samples) {
				break
			}
			got = append(got, pcmSamples(r.Process(pcmBytes(samples[pos:pos+size])))...)
			pos += size
		}
		if !equalSamples(got, atOnce[:len(got)]) {
			t.Errorf("%d->%d: streamed in random chunks differs from at once", in, out)
		}
	}
}

func equalSamples(a, b []int16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// resampledTone streams a second of a tone and returns the output past the
// first tenth of a second, where the filter is still filling.
func resampledTone(t *testing.T, freq float64, in, out int) []float64 {
	t.Helper()
	r := newStream(t, in, out, resample.Config{})
	s := concat(streamChunks(t, r, tonePCM(freq, in, 1), in/25))[out/10:]
	f := make([]float64, len(s))
	for i, v := range s {
		f[i] = float64(v)
	}
	return f
}

// test_passband_is_kept: ported as is.
func TestSincStreamPassbandIsKept(t *testing.T) {
	for _, c := range []struct {
		in, out int
		freq    float64
	}{{24000, 16000, 7000}, {24000, 8000, 3400}, {8000, 24000, 3400}} {
		if db := levelDB(resampledTone(t, c.freq, c.in, c.out), c.out, c.freq); db <= -0.1 {
			t.Errorf("%d->%d at %v Hz: %.3f dB, want above -0.1", c.in, c.out, c.freq, db)
		}
	}
}

// test_downsampling_removes_what_would_alias: ported as is.
func TestSincStreamDownsamplingRemovesAliases(t *testing.T) {
	for _, c := range []struct {
		in, out int
		freq    float64
	}{{24000, 16000, 10000}, {24000, 8000, 5000}} {
		probe := float64(c.out) - c.freq
		if db := levelDB(resampledTone(t, c.freq, c.in, c.out), c.out, probe); db >= -80 {
			t.Errorf("%d->%d, %v Hz folded to %v: %.1f dB, want below -80", c.in, c.out, c.freq, probe, db)
		}
	}
}

// test_upsampling_removes_images: ported as is.
func TestSincStreamUpsamplingRemovesImages(t *testing.T) {
	for _, c := range []struct {
		in, out int
		freq    float64
	}{{8000, 24000, 3000}, {16000, 48000, 7000}} {
		probe := float64(c.in) - c.freq
		if db := levelDB(resampledTone(t, c.freq, c.in, c.out), c.out, probe); db >= -80 {
			t.Errorf("%d->%d, image of %v Hz at %v: %.1f dB, want below -80", c.in, c.out, c.freq, probe, db)
		}
	}
}

// freshOutput is a 40 ms chunk of noise converted by a new stream.
func freshOutput(t *testing.T) []byte {
	t.Helper()
	return newStream(t, 24000, 8000, resample.Config{}).Process(pcmBytes(noisePCM(24000)[:960]))
}

// test_flush_emits_the_delayed_end_and_starts_over: ported as is.
func TestSincStreamFlushEmitsTheDelayedEnd(t *testing.T) {
	r := newStream(t, 24000, 8000, resample.Config{})
	streamChunks(t, r, tonePCM(1000, 24000, 1), 960)
	tail := pcmSamples(r.Flush())
	if len(tail) == 0 {
		t.Fatal("flush emitted nothing")
	}
	peak := 0
	for _, v := range tail {
		peak = max(peak, int(v), -int(v))
	}
	if peak <= amplitude/2 {
		t.Errorf("the tail peaks at %d, want the tone's end", peak)
	}
	if !bytes.Equal(r.Process(pcmBytes(noisePCM(24000)[:960])), freshOutput(t)) {
		t.Error("the stream after a flush does not start over")
	}
}

// test_flush_before_any_audio_is_empty and
// test_flush_without_new_audio_is_empty: ported as is.
func TestSincStreamFlushWithoutNewAudioIsEmpty(t *testing.T) {
	r := newStream(t, 24000, 8000, resample.Config{})
	if len(r.Flush()) != 0 {
		t.Error("flush before any audio is not empty")
	}
	chunk := pcmBytes(noisePCM(24000)[:960])
	r.Process(chunk)
	if len(r.Flush()) == 0 {
		t.Error("flush after audio is empty")
	}
	if len(r.Flush()) != 0 {
		t.Error("a second flush is not empty")
	}
	r.Process(chunk)
	r.Clear()
	if len(r.Flush()) != 0 {
		t.Error("flush after a clear is not empty")
	}
}

// test_reset_discards_the_held_audio: ported as is.
func TestSincStreamClearDiscardsTheHeldAudio(t *testing.T) {
	r := newStream(t, 24000, 8000, resample.Config{})
	streamChunks(t, r, tonePCM(1000, 24000, 1), 960)
	r.Clear()
	if !bytes.Equal(r.Process(pcmBytes(noisePCM(24000)[:960])), freshOutput(t)) {
		t.Error("the stream after a clear still carries the audio before it")
	}
}

// test_inactivity_discards_the_held_audio and
// test_a_long_pause_keeps_the_held_audio_by_default: ported as is.
func TestSincStreamInactivity(t *testing.T) {
	chunk := pcmBytes(noisePCM(24000)[:960])

	r := newStream(t, 24000, 8000, resample.Config{ClearAfter: 50 * time.Millisecond})
	streamChunks(t, r, tonePCM(1000, 24000, 1), 960)
	time.Sleep(100 * time.Millisecond)
	if !bytes.Equal(r.Process(chunk), freshOutput(t)) {
		t.Error("a pause past the idle window kept the held audio")
	}

	r = newStream(t, 24000, 8000, resample.Config{})
	streamChunks(t, r, tonePCM(1000, 24000, 1), 960)
	time.Sleep(100 * time.Millisecond)
	if bytes.Equal(r.Process(chunk), freshOutput(t)) {
		t.Error("a pause dropped the held audio with no idle window set")
	}
}

// test_output_lines_up_with_the_input: a clip converted in one pass lines up
// with the ideal tone at the output rate.
func TestSincClipLinesUpWithTheInput(t *testing.T) {
	for _, p := range ratePairs {
		in, out := p[0], p[1]
		src := make([]int16, in)
		for i := range src {
			src[i] = int16(math.RoundToEven(math.Sin(2*math.Pi*1000*float64(i)/float64(in)) * amplitude))
		}
		o, err := resample.Resample(pcmBytes(src), in, out, 1)
		if err != nil {
			t.Fatal(err)
		}
		s := pcmSamples(o)
		worst := 0.0
		for i := len(s) / 4; i < 3*len(s)/4; i++ {
			ideal := math.Sin(2*math.Pi*1000*float64(i)/float64(out)) * amplitude
			worst = max(worst, math.Abs(float64(s[i])-ideal))
		}
		if worst > 3 {
			t.Errorf("%d->%d: off the ideal by %v, want at most 3", in, out, worst)
		}
	}
}
