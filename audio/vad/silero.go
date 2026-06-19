package vad

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/gojargo/jargo/internal/onnxrt"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed silero_vad.onnx
//nolint:gochecknoglobals // embedded model weights
var sileroModel []byte

var (
	errUnsupportedSampleRate = errors.New("vad: Silero supports only 8000 or 16000 Hz")
	errUnexpectedTensor      = errors.New("vad: unexpected output tensor type")
)

// modelResetInterval is how often the Silero model's recurrent state is reset.
// The model does not need unbounded history, and periodic resets keep memory
// and drift in check.
const modelResetInterval = 5 * time.Second

// Silero is a voice activity Analyzer backed by the Silero VAD ONNX model. It
// supports 8 kHz and 16 kHz mono input and manages the model's recurrent state
// and input context across calls.
type Silero struct {
	*stateMachine

	session *onnxrt.Session

	// Recurrent state [2,1,128] fed back into the model each call, and the
	// trailing input context the model expects prepended to each frame.
	state       []float32
	context     []float32
	contextSize int

	lastReset time.Time
}

// Option configures a Silero analyzer.
type Option func(*config)

type config struct {
	params Params
}

// WithParams sets the detection parameters.
func WithParams(p Params) Option {
	return func(c *config) { c.params = p }
}

// NewSilero loads the embedded Silero VAD model and returns an analyzer. It
// requires the ONNX runtime to be locatable (see the onnxrt package); the
// returned error explains how to configure it when it is not.
func NewSilero(opts ...Option) (*Silero, error) {
	cfg := config{params: DefaultParams()}
	for _, opt := range opts {
		opt(&cfg)
	}

	session, err := onnxrt.New(sileroModel, []string{"input", "state", "sr"}, []string{"output", "stateN"})
	if err != nil {
		return nil, fmt.Errorf("vad: load Silero model: %w", err)
	}

	s := &Silero{
		session: session,
		state:   make([]float32, 2*1*128),
	}
	s.stateMachine = newStateMachine(s, cfg.params)
	return s, nil
}

// SetSampleRate sets the input sample rate. Silero supports only 8 kHz and
// 16 kHz.
func (s *Silero) SetSampleRate(sampleRate int) error {
	if sampleRate != 8000 && sampleRate != 16000 {
		return fmt.Errorf("%w: got %d", errUnsupportedSampleRate, sampleRate)
	}
	if sampleRate == 16000 {
		s.contextSize = 64
	} else {
		s.contextSize = 32
	}
	s.Reset()
	s.setSampleRate(sampleRate)
	return nil
}

// numFramesRequired is the number of samples per analysis frame: 512 at 16 kHz,
// 256 at 8 kHz.
func (s *Silero) numFramesRequired() int {
	if s.sampleRate == 16000 {
		return 512
	}
	return 256
}

// voiceConfidence runs the model over one frame and returns its speech
// probability in [0,1]. buffer is exactly numFramesRequired samples of mono
// 16-bit PCM.
func (s *Silero) voiceConfidence(buffer []byte) float64 {
	samples := pcmToFloat32(buffer)

	// The model expects the previous trailing context prepended to the frame;
	// the new context is the trailing samples of this combined input.
	input := make([]float32, 0, len(s.context)+len(samples))
	input = append(input, s.context...)
	input = append(input, samples...)

	conf, err := s.run(input)
	if err != nil {
		// An inference failure should not crash the pipeline; treat the frame
		// as silence so the state machine errs toward quiet.
		return 0
	}

	s.context = append(s.context[:0], input[len(input)-s.contextSize:]...)

	now := time.Now()
	if now.Sub(s.lastReset) >= modelResetInterval {
		s.resetModel()
		s.lastReset = now
	}
	return conf
}

// run executes one inference pass, feeding the recurrent state back in and
// reading the updated state out.
func (s *Silero) run(input []float32) (float64, error) {
	inT, err := ort.NewTensor(ort.NewShape(1, int64(len(input))), input)
	if err != nil {
		return 0, err
	}
	defer func() { _ = inT.Destroy() }()
	stT, err := ort.NewTensor(ort.NewShape(2, 1, 128), s.state)
	if err != nil {
		return 0, err
	}
	defer func() { _ = stT.Destroy() }()
	srT, err := ort.NewTensor(ort.NewShape(1), []int64{int64(s.sampleRate)})
	if err != nil {
		return 0, err
	}
	defer func() { _ = srT.Destroy() }()

	outs, err := s.session.Run([]ort.Value{inT, stT, srT})
	if err != nil {
		return 0, err
	}
	defer destroyAll(outs)

	out, ok := outs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("%w: %T", errUnexpectedTensor, outs[0])
	}
	conf := float64(out.GetData()[0])

	if newState, ok := outs[1].(*ort.Tensor[float32]); ok {
		copy(s.state, newState.GetData())
	}
	return conf, nil
}

// Reset clears the model state, the input context and the detection state
// machine.
func (s *Silero) Reset() {
	s.resetModel()
	if s.stateMachine != nil {
		s.reset()
	}
}

func (s *Silero) resetModel() {
	for i := range s.state {
		s.state[i] = 0
	}
	s.context = make([]float32, s.contextSize)
	s.lastReset = time.Now()
}

// Close releases the model session.
func (s *Silero) Close() error {
	if s.session == nil {
		return nil
	}
	err := s.session.Close()
	s.session = nil
	return err
}

// pcmToFloat32 converts mono 16-bit little-endian PCM to float32 normalized to
// [-1,1).
func pcmToFloat32(pcm []byte) []float32 {
	n := len(pcm) / 2
	out := make([]float32, n)
	for i := range n {
		s := int16(binary.LittleEndian.Uint16(pcm[i*2:]))
		out[i] = float32(s) / 32768.0
	}
	return out
}

func destroyAll(vs []ort.Value) {
	for _, v := range vs {
		if v != nil {
			_ = v.Destroy()
		}
	}
}

var _ Analyzer = (*Silero)(nil)
