package turn

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/internal/onnxrt"
)

//go:embed smart-turn-v3.2-cpu.onnx
//nolint:gochecknoglobals // embedded model weights
var smartTurnModel []byte

var errUnexpectedTensor = errors.New("turn: unexpected output tensor type")

// segment is one appended chunk of turn audio with the wall-clock time it
// arrived, used to bound the buffer and to locate the pre-speech window.
type segment struct {
	at      time.Time
	samples []int16
}

// predictor is the model half of a smart-turn analyzer: it scores a float32
// audio segment for turn completion. smartTurnBase drives it.
type predictor interface {
	predictEndpoint(audio []float32) (complete bool, probability float64, err error)
}

// smartTurnBase holds the audio buffering and silence tracking shared by
// smart-turn analyzers. A concrete analyzer embeds it and supplies itself as
// the predictor.
type smartTurnBase struct {
	self   predictor
	params Params

	sampleRate int
	stopMs     float64

	buffer          []segment
	speechTriggered bool
	silenceMs       float64
	speechStart     time.Time
	vadStartSecs    float64
}

// Params are the analysis parameters this analyzer runs with.
func (b *smartTurnBase) Params() Params { return b.params }

func newSmartTurnBase(self predictor, params Params) *smartTurnBase {
	return &smartTurnBase{
		self:   self,
		params: params,
		stopMs: params.StopSecs * 1000,
	}
}

// SetSampleRate sets the input sample rate.
func (b *smartTurnBase) SetSampleRate(sampleRate int) { b.sampleRate = sampleRate }

// SpeechTriggered reports whether speech has been heard since the turn began.
func (b *smartTurnBase) SpeechTriggered() bool { return b.speechTriggered }

// UpdateVADStartSecs stores the VAD start delay used to widen the pre-speech
// window.
func (b *smartTurnBase) UpdateVADStartSecs(secs float64) { b.vadStartSecs = secs }

// AppendAudio buffers a chunk of turn audio and tracks silence. It returns
// Complete only when accumulated silence crosses the stop-seconds safety net;
// the model decision happens in AnalyzeEndOfTurn.
func (b *smartTurnBase) AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState {
	samples := pcmToInt16(buffer)
	now := time.Now()
	b.buffer = append(b.buffer, segment{at: now, samples: samples})

	state := Incomplete
	switch {
	case isSpeech:
		b.silenceMs = 0
		b.speechTriggered = true
		if b.speechStart.IsZero() {
			b.speechStart = now
		}
	case b.speechTriggered:
		chunkMs := float64(len(samples)) / (float64(b.sampleRate) / 1000)
		b.silenceMs += chunkMs
		if b.silenceMs >= b.stopMs {
			state = Complete
			b.clear(Complete)
		}
	default:
		// Trim pre-speech buffer to bound growth before the turn starts.
		maxBufferSecs := b.params.PreSpeechMs/1000 + b.params.StopSecs + b.params.MaxDurationSecs
		cutoff := now.Add(-time.Duration(maxBufferSecs * float64(time.Second)))
		for len(b.buffer) > 0 && b.buffer[0].at.Before(cutoff) {
			b.buffer = b.buffer[1:]
		}
	}
	return state
}

// AnalyzeEndOfTurn runs the model over the buffered turn and returns the
// predicted state and completion probability. On Complete it clears the buffer.
func (b *smartTurnBase) AnalyzeEndOfTurn() (EndOfTurnState, float64, error) {
	state, prob, err := b.processSpeechSegment()
	if err != nil {
		return Incomplete, 0, err
	}
	if state == Complete {
		b.clear(Complete)
	}
	return state, prob, nil
}

// Clear resets the analyzer to its initial state.
func (b *smartTurnBase) Clear() { b.clear(Complete) }

func (b *smartTurnBase) clear(state EndOfTurnState) {
	// On an incomplete turn the user is still considered speaking.
	b.speechTriggered = state == Incomplete
	b.buffer = nil
	b.speechStart = time.Time{}
	b.silenceMs = 0
}

// processSpeechSegment assembles the analyzed audio segment and runs the model.
func (b *smartTurnBase) processSpeechSegment() (EndOfTurnState, float64, error) {
	if len(b.buffer) == 0 {
		return Incomplete, 0, nil
	}

	// Start the segment a little before speech onset.
	effectivePreSpeechMs := b.params.PreSpeechMs + b.vadStartSecs*1000
	startTime := b.speechStart.Add(-time.Duration(effectivePreSpeechMs * float64(time.Millisecond)))
	startIndex := 0
	for i := range b.buffer {
		if !b.buffer[i].at.Before(startTime) {
			startIndex = i
			break
		}
	}

	var total int
	for _, seg := range b.buffer[startIndex:] {
		total += len(seg.samples)
	}
	if total == 0 {
		return Incomplete, 0, nil
	}
	audio := make([]float32, 0, total)
	for _, seg := range b.buffer[startIndex:] {
		for _, s := range seg.samples {
			audio = append(audio, float32(s)/32768.0)
		}
	}

	// Keep only the most recent MaxDurationSecs of audio.
	maxSamples := int(b.params.MaxDurationSecs * float64(b.sampleRate))
	if len(audio) > maxSamples {
		audio = audio[len(audio)-maxSamples:]
	}

	complete, prob, err := b.self.predictEndpoint(audio)
	if err != nil {
		return Incomplete, 0, err
	}
	if complete {
		return Complete, prob, nil
	}
	return Incomplete, prob, nil
}

// SmartTurnV3 is an end-of-turn Analyzer backed by the smart-turn-v3 ONNX model.
type SmartTurnV3 struct {
	*smartTurnBase
	session *onnxrt.Session
}

// TurnOption configures a SmartTurnV3 analyzer.
type TurnOption func(*Params)

// WithParams sets the analysis parameters.
func WithParams(p Params) TurnOption {
	return func(dst *Params) { *dst = p }
}

// NewSmartTurnV3 loads the embedded smart-turn-v3 model and returns an analyzer.
// It requires the ONNX runtime to be locatable (see the onnxrt package).
func NewSmartTurnV3(opts ...TurnOption) (*SmartTurnV3, error) {
	params := DefaultParams()
	for _, opt := range opts {
		opt(&params)
	}

	// One thread per Run, for the reason given in the VAD analyzer: this model
	// runs only at the end of an utterance, so a spinning pool sized to the host
	// burns cores between inferences without making any of them faster.
	session, err := onnxrt.NewWithOptions(smartTurnModel,
		[]string{"input_features"}, []string{"logits"},
		onnxrt.Options{IntraOpThreads: 1})
	if err != nil {
		return nil, fmt.Errorf("turn: load Smart Turn model: %w", err)
	}

	s := &SmartTurnV3{session: session}
	s.smartTurnBase = newSmartTurnBase(s, params)
	return s, nil
}

// predictEndpoint computes Whisper log-mel features over the last 8 seconds of
// the segment and runs the model, returning whether the turn is complete and
// the completion probability.
func (s *SmartTurnV3) predictEndpoint(audio []float32) (bool, float64, error) {
	audio, err := resampleToModelRate(audio, s.sampleRate)
	if err != nil {
		return false, 0, err
	}
	audio = lastNSamples(audio, nSamples) // keep the last 8s, zero-pad the front
	features := computeLogMel(audio)
	prob, err := s.runModel(features)
	if err != nil {
		return false, 0, err
	}
	return prob > 0.5, prob, nil
}

// runModel runs the model on a precomputed [80*800] feature matrix and returns
// the completion probability. The model's output is already sigmoid-activated.
func (s *SmartTurnV3) runModel(features []float32) (float64, error) {
	outs, err := s.session.Run([]onnxrt.Tensor{
		onnxrt.Float32([]int64{1, nMels, nFrames}, features),
	})
	if err != nil {
		return 0, err
	}
	if len(outs) != 1 || len(outs[0].F32) == 0 {
		return 0, fmt.Errorf("%w: got %d outputs", errUnexpectedTensor, len(outs))
	}
	return float64(outs[0].F32[0]), nil
}

// Close releases the model session.
func (s *SmartTurnV3) Close() error {
	if s.session == nil {
		return nil
	}
	err := s.session.Close()
	s.session = nil
	return err
}

// lastNSamples returns the last n samples of audio, zero-padding at the front
// if audio is shorter than n.
func lastNSamples(audio []float32, n int) []float32 {
	if len(audio) == n {
		return audio
	}
	if len(audio) > n {
		return audio[len(audio)-n:]
	}
	out := make([]float32, n)
	copy(out[n-len(audio):], audio)
	return out
}

// modelResampleQuality is the conversion quality used to reach the model rate.
// A shorter filter than the pipeline default is enough here: the difference
// between the two sits well below the noise floor of the log-mel features the
// model reads, so it cannot move a prediction, and this runs at the end of every
// user turn.
const modelResampleQuality = resample.QualityHQ

// resampleToModelRate converts audio to the 16 kHz the model expects, through
// the pipeline's own converter: a sinc polyphase filter, or libsoxr itself under
// the libsoxr tag. It is used only when a non-16 kHz turn stream is configured;
// the turntaking processor normally feeds 16 kHz directly.
//
// The filter is the point. Rate conversion without one folds everything above
// the new Nyquist back down into the band below it, and for a 48 kHz stream that
// is most of the spectrum landing on top of the speech. The model reads log-mel
// features of exactly that band, so the aliases would not be noise it could see
// past: they would be the features.
//
// The turn's audio is complete by the time it gets here, so it converts in a
// single pass rather than through a stream resampler. A stream resampler holds
// its filter delay back for audio it expects to follow, and here nothing does:
// it would clip the last millisecond or two off the end of the turn, which is
// the part the model reads most closely.
//
// The audio reached the analyzer as 16-bit PCM and is converted back to it here,
// which is what the converter takes and costs nothing that was not already spent
// quantizing it in the first place.
func resampleToModelRate(audio []float32, inRate int) ([]float32, error) {
	if inRate == melSR || inRate == 0 || len(audio) == 0 {
		return audio, nil
	}

	pcm := make([]byte, len(audio)*2)
	for i, s := range audio {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(f32ToS16(s)))
	}

	out, err := resample.ResampleQuality(pcm, inRate, melSR, 1, modelResampleQuality)
	if err != nil {
		return nil, fmt.Errorf("turn: resample %d Hz to %d Hz: %w", inRate, melSR, err)
	}

	res := make([]float32, len(out)/2)
	for i := range res {
		res[i] = float32(int16(binary.LittleEndian.Uint16(out[i*2:]))) / 32768
	}
	return res, nil
}

// f32ToS16 converts a normalized float sample to S16 with rounding and clamping.
// Sinc interpolation can overshoot past [-1, 1), so the clamp is load-bearing.
func f32ToS16(f float32) int16 {
	v := float64(f) * 32768
	if v >= 0 {
		v += 0.5
	} else {
		v -= 0.5
	}
	switch {
	case v > 32767:
		return 32767
	case v < -32768:
		return -32768
	default:
		return int16(v)
	}
}

// pcmToInt16 reinterprets mono 16-bit little-endian PCM as int16 samples.
func pcmToInt16(pcm []byte) []int16 {
	n := len(pcm) / 2
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return out
}

var _ Analyzer = (*SmartTurnV3)(nil)
