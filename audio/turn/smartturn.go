package turn

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/gojargo/jargo/internal/onnxrt"
	ort "github.com/yalue/onnxruntime_go"
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

func newSmartTurnBase(self predictor, params Params) *smartTurnBase {
	return &smartTurnBase{
		self:   self,
		params: params,
		stopMs: params.StopSecs * 1000,
	}
}

// SetSampleRate sets the input sample rate.
func (b *smartTurnBase) SetSampleRate(sampleRate int) { b.sampleRate = sampleRate }

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

	session, err := onnxrt.New(smartTurnModel, []string{"input_features"}, []string{"logits"})
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
	if s.sampleRate != melSR && s.sampleRate != 0 {
		audio = resampleLinear(audio, s.sampleRate, melSR)
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
	in, err := ort.NewTensor(ort.NewShape(1, nMels, nFrames), features)
	if err != nil {
		return 0, err
	}
	defer func() { _ = in.Destroy() }()

	outs, err := s.session.Run([]ort.Value{in})
	if err != nil {
		return 0, err
	}
	defer func() {
		for _, v := range outs {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()

	out, ok := outs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("%w: %T", errUnexpectedTensor, outs[0])
	}
	return float64(out.GetData()[0]), nil
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

// resampleLinear resamples float32 PCM from inRate to outRate by linear
// interpolation. It is used only when a non-16 kHz turn stream is configured;
// the turntaking processor normally feeds 16 kHz directly.
func resampleLinear(in []float32, inRate, outRate int) []float32 {
	if inRate == outRate || len(in) == 0 {
		return in
	}
	outLen := len(in) * outRate / inRate
	out := make([]float32, outLen)
	ratio := float64(inRate) / float64(outRate)
	for i := range out {
		src := float64(i) * ratio
		j := int(src)
		frac := src - float64(j)
		if j+1 < len(in) {
			out[i] = in[j]*(1-float32(frac)) + in[j+1]*float32(frac)
		} else {
			out[i] = in[len(in)-1]
		}
	}
	return out
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
