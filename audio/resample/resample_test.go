package resample_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gojargo/jargo/audio/resample"
)

func sine(freq float64, rate, n int) []byte {
	b := make([]byte, n*2)
	for i := range n {
		v := math.Sin(2*math.Pi*freq*float64(i)/float64(rate)) * 0.5 * math.MaxInt16
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v)))
	}
	return b
}

func TestIdentityWhenRatesMatch(t *testing.T) {
	r, err := resample.New(48000, 48000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	in := sine(440, 48000, 100)
	out := r.Process(in)
	if len(out) != len(in) {
		t.Fatalf("identity changed length: %d -> %d", len(in), len(out))
	}
}

// streamFrames runs `buffers` equal-sized buffers through r and returns the
// total output frames. Streaming a long signal makes the converter's fixed
// filter delay negligible relative to the total, so the output-to-input frame
// ratio converges to outRate/inRate even though any single call is off by the
// delay.
func streamFrames(r *resample.Resampler, freq float64, inRate, framesPerBuffer, buffers int) int {
	total := 0
	for range buffers {
		out := r.Process(sine(freq, inRate, framesPerBuffer))
		total += len(out) / 2
	}
	return total
}

// assertRatio checks got is within 2% of inFrames*outRate/inRate, tolerating
// the converter's filter delay.
func assertRatio(t *testing.T, got, inFrames, inRate, outRate int) {
	t.Helper()
	want := float64(inFrames) * float64(outRate) / float64(inRate)
	if math.Abs(float64(got)-want) > 0.02*want {
		t.Fatalf("got %d output frames, want ~%.0f (±2%%)", got, want)
	}
}

func TestUpsample24kTo48k(t *testing.T) {
	r, err := resample.New(24000, 48000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	// 500 * 480 = 240000 input frames @ 24k -> ~480000 frames @ 48k. The stream
	// is long enough that the fixed filter delay is well under the tolerance.
	got := streamFrames(r, 440, 24000, 480, 500)
	assertRatio(t, got, 240000, 24000, 48000)
}

func TestDownsample48kTo16k(t *testing.T) {
	r, err := resample.New(48000, 16000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	// 500 * 960 = 480000 input frames @ 48k -> ~160000 frames @ 16k.
	got := streamFrames(r, 440, 48000, 960, 500)
	assertRatio(t, got, 480000, 48000, 16000)
}

func TestCloseIsIdempotent(t *testing.T) {
	r, err := resample.New(24000, 48000, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r.Close()
	r.Close() // must not panic
}

// TestBatchKeepsTheWholeSignal covers the difference between the two entry
// points. A Resampler holds the filter delay back at the end of every call
// because more audio is expected, so a complete buffer pushed through one comes
// out a filter length short. Resample flushes it.
func TestBatchKeepsTheWholeSignal(t *testing.T) {
	const (
		inRate  = 48000
		outRate = 16000
		frames  = inRate // one second
	)
	in := sine(440, inRate, frames)
	want := frames * outRate / inRate

	batch, err := resample.Resample(in, inRate, outRate, 1)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	if got := len(batch) / 2; got != want {
		t.Errorf("Resample dropped audio: got %d frames, want %d", got, want)
	}

	r, err := resample.New(inRate, outRate, 1)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer r.Close()
	if got := len(r.Process(in)) / 2; got >= want {
		t.Errorf("a single Process call was expected to hold the filter delay back, got %d frames", got)
	}
}

func TestBatchPassesThroughWhenRatesMatch(t *testing.T) {
	in := sine(440, 16000, 100)
	out, err := resample.Resample(in, 16000, 16000, 1)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("passthrough changed length: %d -> %d", len(in), len(out))
	}
}

func TestBatchEmptyInput(t *testing.T) {
	out, err := resample.Resample(nil, 48000, 16000, 1)
	if err != nil {
		t.Fatalf("Resample: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("empty input produced %d bytes", len(out))
	}
}

// TestClearDropsFilterHistory checks the point of clearing: the filter carries
// the tail of the audio it last saw, so without a clear the end of one signal
// bleeds into the start of the next.
func TestClearDropsFilterHistory(t *testing.T) {
	loud := sine(440, 48000, 4800)  // 100ms at half scale
	silence := make([]byte, 4800*2) // 100ms of nothing

	bleed := func(reset bool) int {
		r, err := resample.New(48000, 16000, 1)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer r.Close()
		r.Process(loud)
		if reset {
			r.Clear()
		}
		return peak(r.Process(silence))
	}

	kept := bleed(false)
	if kept == 0 {
		t.Fatal("expected the filter to carry the tail of the previous signal into the silence")
	}
	// A cleared filter can still round a sample to the last bit; anything more
	// than that is the previous signal.
	if got := bleed(true); got > kept/100 {
		t.Errorf("after Clear the silence should be silent, got peak %d (uncleared peak is %d)", got, kept)
	}
}

// peak returns the largest absolute sample in a buffer of S16LE PCM.
func peak(pcm []byte) int {
	most := 0
	for i := 0; i+1 < len(pcm); i += 2 {
		v := int(int16(binary.LittleEndian.Uint16(pcm[i:])))
		if v < 0 {
			v = -v
		}
		if v > most {
			most = v
		}
	}
	return most
}

func TestQualityStrings(t *testing.T) {
	for q, want := range map[resample.Quality]string{
		resample.QualityVHQ: "VHQ",
		resample.QualityHQ:  "HQ",
		resample.QualityMQ:  "MQ",
		resample.QualityLQ:  "LQ",
		resample.QualityQQ:  "QQ",
	} {
		if got := q.String(); got != want {
			t.Errorf("Quality(%d).String() = %q, want %q", q, got, want)
		}
	}
	// The zero value is the default, and the default is the highest quality.
	if resample.Quality(0) != resample.QualityVHQ {
		t.Error("the zero Quality should be VHQ")
	}
}

// TestEveryQualityConverts guards the mapping onto each build's converter set:
// every recipe has to produce audio at roughly the right rate, whichever
// backend is compiled in.
func TestEveryQualityConverts(t *testing.T) {
	for _, q := range []resample.Quality{
		resample.QualityVHQ, resample.QualityHQ, resample.QualityMQ,
		resample.QualityLQ, resample.QualityQQ,
	} {
		t.Run(q.String(), func(t *testing.T) {
			r, err := resample.NewWithConfig(48000, 16000, 1, resample.Config{Quality: q})
			if err != nil {
				t.Fatalf("NewWithConfig: %v", err)
			}
			defer r.Close()
			got := len(streamFramesBytes(r, sine(440, 48000, 4800), 10)) / 2
			if want := 16000; got < want*9/10 || got > want*11/10 {
				t.Errorf("got %d frames for one second of audio, want about %d", got, want)
			}
		})
	}
}

// streamFramesBytes runs the same buffer through r n times and returns every
// byte it produced.
func streamFramesBytes(r *resample.Resampler, buf []byte, n int) []byte {
	var out []byte
	for range n {
		out = append(out, r.Process(buf)...)
	}
	return out
}

// TestStreamingLosesNoAudio guards the accounting in Process. A stream
// resampler holds a filter length of audio in flight at all times, so its output
// always trails the rate ratio by that much, but the gap must stay bounded: if
// it grows with the length of the stream, audio is being dropped rather than
// merely delayed.
//
// A converter is not obliged to consume a whole buffer in one call, and a
// backend that silently drops whatever it leaves behind fails here.
func TestStreamingLosesNoAudio(t *testing.T) {
	const (
		inRate          = 48000
		outRate         = 16000
		framesPerBuffer = 4800 // 100ms
		// A tenth of a second of slack, comfortably above any backend's filter
		// length and far below the loss a dropped buffer would show.
		maxTrailingFrames = outRate / 10
	)

	shortfall := func(buffers int) int {
		r, err := resample.New(inRate, outRate, 1)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer r.Close()
		got := streamFrames(r, 440, inRate, framesPerBuffer, buffers)
		return buffers*framesPerBuffer*outRate/inRate - got
	}

	short, long := shortfall(20), shortfall(400)
	if short > maxTrailingFrames || long > maxTrailingFrames {
		t.Errorf("output trails the input by more than a filter length: %d frames over 20 buffers, %d over 400", short, long)
	}
	// Twenty times the audio must not mean twenty times the gap.
	if long > short*4 && long-short > outRate/50 {
		t.Errorf("the gap grows with the stream, so audio is being dropped: "+
			"%d frames over 20 buffers, %d over 400", short, long)
	}
}
