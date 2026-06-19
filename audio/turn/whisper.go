package turn

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"
)

// Whisper-style log-mel feature extraction for Smart Turn v3, ported from the
// numpy reference (transformers.WhisperFeatureExtractor with chunk_length=8).
// The model consumes a fixed [80, 800] feature matrix computed from exactly 8
// seconds of 16 kHz audio.

const (
	nFFT       = 400
	hopLength  = 160
	nMels      = 80
	melSR      = 16000
	melFloor   = 1e-10
	normVarEps = 1e-7
	nSamples   = melSR * 8  // 128000
	nFrames    = 800        // feature frames after dropping the trailing frame
	numFreqs   = nFFT/2 + 1 // 201
)

//nolint:gochecknoglobals // precomputed feature-extraction constants
var (
	hannWindow   [nFFT]float64
	melFilters   [nMels][numFreqs]float64 // projects a power spectrum onto the mel scale; mel-major so the projection's inner loop over frequency bins is contiguous
	melLo, melHi [nMels]int               // inclusive range of nonzero bins per mel filter; the filters are triangular, so only a handful of bins are nonzero
	hannDFTPower [numFreqs]float64         // |DFT(hann)|^2, used to shortcut constant frames
	featuresInit bool
)

// featureWorkspace bundles the FFT and scratch buffers one log-mel extraction
// needs. A gonum FFT holds mutable work buffers and is not safe for concurrent
// use, so each computeLogMel call borrows a workspace from the pool — one per
// concurrent turn analysis — rather than sharing a global.
type featureWorkspace struct {
	fft      *fourier.FFT
	windowed []float64    // windowed frame, length nFFT
	coeffs   []complex128 // FFT output, length numFreqs
}

//nolint:gochecknoglobals // reusable FFT/scratch workspaces, see featureWorkspace
var workspacePool = sync.Pool{
	New: func() any {
		return &featureWorkspace{
			fft:      fourier.NewFFT(nFFT),
			windowed: make([]float64, nFFT),
			coeffs:   make([]complex128, numFreqs),
		}
	},
}

// initFeatures precomputes the window, mel filterbank and DFT tables. It runs
// lazily on first use rather than in init() so importing the package stays
// cheap.
func initFeatures() {
	if featuresInit {
		return
	}
	// Periodic Hann window: np.hanning(N+1)[:-1].
	for n := range nFFT {
		hannWindow[n] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(nFFT))
	}
	buildMelFilterbank()

	// |DFT(hann)|^2 per bin, for the constant-frame shortcut in framePower.
	coeffs := fourier.NewFFT(nFFT).Coefficients(nil, hannWindow[:])
	for k := range numFreqs {
		re, im := real(coeffs[k]), imag(coeffs[k])
		hannDFTPower[k] = re*re + im*im
	}
	featuresInit = true
}

func hertzToMelSlaney(freq float64) float64 {
	const minLogHertz = 1000.0
	const minLogMel = 15.0
	logstep := 27.0 / math.Log(6.4)
	if freq >= minLogHertz {
		return minLogMel + math.Log(freq/minLogHertz)*logstep
	}
	return 3.0 * freq / 200.0
}

func melToHertzSlaney(mel float64) float64 {
	const minLogHertz = 1000.0
	const minLogMel = 15.0
	logstep := math.Log(6.4) / 27.0
	if mel >= minLogMel {
		return minLogHertz * math.Exp(logstep*(mel-minLogMel))
	}
	return 200.0 * mel / 3.0
}

// buildMelFilterbank constructs the Slaney-normalized triangular mel
// filterbank, matching the numpy reference's vectorized construction.
func buildMelFilterbank() {
	const maxFreq = melSR / 2.0

	melMin := hertzToMelSlaney(0.0)
	melMax := hertzToMelSlaney(maxFreq)

	// num_mel_filters + 2 mel points, linearly spaced, mapped back to hertz.
	filterFreqs := make([]float64, nMels+2)
	for i := range filterFreqs {
		mel := melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
		filterFreqs[i] = melToHertzSlaney(mel)
	}

	// FFT bin center frequencies: linspace(0, sr/2, num_frequency_bins).
	fftFreqs := make([]float64, numFreqs)
	for i := range fftFreqs {
		fftFreqs[i] = maxFreq * float64(i) / float64(numFreqs-1)
	}

	filterDiff := make([]float64, nMels+1)
	for i := range nMels + 1 {
		filterDiff[i] = filterFreqs[i+1] - filterFreqs[i]
	}

	for b := range numFreqs {
		for m := range nMels {
			// slopes[b][j] = filterFreqs[j] - fftFreqs[b]
			down := -(filterFreqs[m] - fftFreqs[b]) / filterDiff[m]
			up := (filterFreqs[m+2] - fftFreqs[b]) / filterDiff[m+1]
			v := math.Min(down, up)
			if v < 0 {
				v = 0
			}
			enorm := 2.0 / (filterFreqs[m+2] - filterFreqs[m]) // Slaney area normalization
			melFilters[m][b] = v * enorm
		}
	}

	// Record each triangular filter's nonzero bin span so the projection skips
	// the long runs of zero weights.
	for m := range nMels {
		lo, hi := 0, -1 // empty by default: lo > hi means no nonzero bins
		for b := range numFreqs {
			if melFilters[m][b] > 0 {
				if hi < lo {
					lo = b
				}
				hi = b
			}
		}
		melLo[m], melHi[m] = lo, hi
	}
}

// normalizePadded returns the audio padded or truncated to exactly nSamples and
// normalized to zero mean and unit variance, in float64.
func normalizePadded(audio []float32) []float64 {
	x := make([]float64, nSamples)
	for i := 0; i < nSamples && i < len(audio); i++ {
		x[i] = float64(audio[i])
	}

	var mean float64
	for _, v := range x {
		mean += v
	}
	mean /= float64(nSamples)

	var variance float64
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float64(nSamples)
	std := math.Sqrt(variance + normVarEps)

	for i := range x {
		x[i] = (x[i] - mean) / std
	}
	return x
}

// reflectPad pads x by pad samples on each side, matching np.pad(mode="reflect").
func reflectPad(x []float64, pad int) []float64 {
	n := len(x)
	out := make([]float64, n+2*pad)
	for j := range pad {
		out[j] = x[pad-j]
		out[pad+n+j] = x[n-2-j]
	}
	copy(out[pad:], x)
	return out
}

// framePower computes the power spectrum of the frame starting at the given
// offset in the reflect-padded signal. A constant frame — common, since most of
// the fixed 8-second window is silence padding that normalization turns into a
// single repeated value — has power c^2 * |DFT(hann)|^2, so the full DFT is
// skipped for it.
func (ws *featureWorkspace) framePower(padded []float64, start int, power *[numFreqs]float64) {
	c := padded[start]
	constant := true
	for n := 1; n < nFFT; n++ {
		if padded[start+n] != c {
			constant = false
			break
		}
	}
	if constant {
		cc := c * c
		for k := range numFreqs {
			power[k] = cc * hannDFTPower[k]
		}
		return
	}

	for n := range nFFT {
		ws.windowed[n] = padded[start+n] * hannWindow[n]
	}
	ws.fft.Coefficients(ws.coeffs, ws.windowed)
	for k := range numFreqs {
		re, im := real(ws.coeffs[k]), imag(ws.coeffs[k])
		power[k] = re*re + im*im
	}
}

// frameLogMel computes the log10 mel spectrum of the frame starting at the
// given sample offset in the reflect-padded signal, writing nMels values into
// dst.
func (ws *featureWorkspace) frameLogMel(padded []float64, start int, dst []float64) {
	var power [numFreqs]float64
	ws.framePower(padded, start, &power)

	for m := range nMels {
		var sum float64
		filt := &melFilters[m]
		for b := melLo[m]; b <= melHi[m]; b++ {
			sum += filt[b] * power[b]
		}
		if sum < melFloor {
			sum = melFloor
		}
		dst[m] = math.Log10(sum)
	}
}

// computeLogMel computes Whisper-style log-mel features for the audio, returning
// an [nMels*nFrames] row-major (mel-major) slice ready as the model's
// [1, 80, 800] input. The audio is float32 PCM normalized to [-1,1] at 16 kHz;
// it is zero-padded or truncated to 8 seconds internally.
func computeLogMel(audio []float32) []float32 {
	initFeatures()

	x := normalizePadded(audio)
	padded := reflectPad(x, nFFT/2)

	ws := workspacePool.Get().(*featureWorkspace)
	defer workspacePool.Put(ws)

	mels := make([]float64, nMels*nFrames)
	logMel := make([]float64, nMels)
	maxLog := math.Inf(-1)

	// One extra frame is computed then dropped, matching the reference; only the
	// first nFrames are kept.
	for f := range nFrames {
		ws.frameLogMel(padded, f*hopLength, logMel)
		for m := range nMels {
			mels[m*nFrames+f] = logMel[m]
			if logMel[m] > maxLog {
				maxLog = logMel[m]
			}
		}
	}

	// Dynamic-range clamp and scale: max(v, max-8); (v+4)/4.
	floor := maxLog - 8.0
	out := make([]float32, nMels*nFrames)
	for i, v := range mels {
		if v < floor {
			v = floor
		}
		out[i] = float32((v + 4.0) / 4.0)
	}
	return out
}
