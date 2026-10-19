package vad

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/gojargo/jargo/internal/onnxrt"
)

//go:embed silero_vad.onnx
//nolint:gochecknoglobals // embedded model weights
var sileroModel []byte

var (
	errUnsupportedSampleRate = errors.New("vad: Silero supports only 8000 or 16000 Hz")
	errUnexpectedTensor      = errors.New("vad: unexpected output tensor type")
	errClosedModel           = errors.New("vad: shared SileroModel is closed or uninitialized")
)

// Model state resets.
//
// The model carries a recurrent state from one window to the next. Non-speech
// shifts that state so the model under-reports the speech that follows: music
// strongly at either sample rate, and long silence or noise strongly at 8 kHz
// and slightly at 16 kHz. A reset clears it.
const (
	// modelResetIntervalSecs is the seconds of audio between resets. After
	// non-speech that suppresses detection, speech is detected again within
	// about one interval. A shorter interval recovers sooner but puts more
	// resets just ahead of speech, where the model occasionally confirms speech
	// a few hundred milliseconds late.
	modelResetIntervalSecs = 2.0
	// A due reset waits for clear non-speech: the VAD must be quiet and
	// modelResetLowConfidenceWindows consecutive windows (about 160 ms) must
	// score below modelResetMaxConfidence. One low window can be a dip inside a
	// word or the soft start of one; a run of them can't. The threshold sits far
	// below the VAD's speech threshold, so borderline windows never count.
	modelResetMaxConfidence        = 0.1
	modelResetLowConfidenceWindows = 5
)

// Silero is a voice activity Analyzer backed by the Silero VAD ONNX model. It
// supports 8 kHz and 16 kHz mono input and manages the model's recurrent state
// and input context across calls.
//
// The model's recurrent state is reset about every two seconds of audio, but
// only during clear non-speech. Non-speech such as music, or long silence at
// 8 kHz, leaves the state unable to detect the speech that follows until it is
// reset. A reset during speech makes the model briefly lose confidence, which
// can end the user's speech early, so a due reset waits until the user is
// quiet.
type Silero struct {
	*stateMachine

	session       *onnxrt.Session
	sharedSession bool

	// Recurrent state [2,1,128] fed back into the model each call, and the
	// trailing input context the model expects prepended to each frame.
	state       []float32
	context     []float32
	contextSize int

	// Windows scored since the model was last reset, and how many of the most
	// recent windows in a row scored below modelResetMaxConfidence.
	windowsSinceReset    int
	lowConfidenceWindows int
}

// SileroModel is a reusable ONNX session holding the loaded Silero VAD model.
// It must outlive any analyzers built from it.
type SileroModel struct {
	session *onnxrt.Session
}

// NewSileroModel loads the embedded Silero VAD model into a reusable session.
func NewSileroModel() (*SileroModel, error) {
	session, err := onnxrt.NewWithOptions(sileroModel,
		[]string{"input", "state", "sr"}, []string{"output", "stateN"},
		onnxrt.Options{IntraOpThreads: 1})
	if err != nil {
		return nil, fmt.Errorf("vad: load Silero model: %w", err)
	}
	return &SileroModel{session: session}, nil
}

// Close releases the shared model session.
func (m *SileroModel) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	err := m.session.Close()
	m.session = nil
	return err
}

// Option configures a Silero analyzer.
type Option func(*config)

type config struct {
	params         Params
	model          *SileroModel
	modelSpecified bool
}

// WithParams sets the detection parameters.
func WithParams(p Params) Option {
	return func(c *config) { c.params = p }
}

// WithModel configures the analyzer to reuse a shared SileroModel session.
func WithModel(m *SileroModel) Option {
	return func(c *config) {
		c.model = m
		c.modelSpecified = true
	}
}

// NewSilero loads the embedded Silero VAD model and returns an analyzer. It
// requires the ONNX runtime to be locatable (see the onnxrt package); the
// returned error explains how to configure it when it is not.
func NewSilero(opts ...Option) (*Silero, error) {
	cfg := config{params: DefaultParams()}
	for _, opt := range opts {
		opt(&cfg)
	}

	var (
		session       *onnxrt.Session
		sharedSession bool
	)
	if cfg.modelSpecified {
		if cfg.model == nil || cfg.model.session == nil {
			return nil, errClosedModel
		}
		session = cfg.model.session
		sharedSession = true
	} else {
		// One thread per Run. The model is tiny and runs once per 32 ms of audio, so
		// the runtime's default pool wins nothing and costs a great deal: its workers
		// spin between inferences, burning whole cores per stream whatever the
		// container's CPU limit says.
		var err error
		session, err = onnxrt.NewWithOptions(sileroModel,
			[]string{"input", "state", "sr"}, []string{"output", "stateN"},
			onnxrt.Options{IntraOpThreads: 1})
		if err != nil {
			return nil, fmt.Errorf("vad: load Silero model: %w", err)
		}
	}

	s := &Silero{
		session:       session,
		sharedSession: sharedSession,
		state:         make([]float32, 2*1*128),
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

	s.maybeResetModel(conf)
	return conf
}

// maybeResetModel resets the model's state once a reset is due and the user is
// quiet.
//
// It is called after each window is scored, so a reset takes effect from the
// next window. A reset clears the model's recurrent state and the audio context
// it prepends to each window; the VAD's own state is untouched.
//
// A reset becomes due after modelResetIntervalSecs of audio, and happens on the
// first window after that where the user is quiet and the recent windows are
// clearly non-speech. The detection state is still the state from before this
// window, since the caller updates it from the confidence returned here. In
// practice a reset follows soon after the user stops speaking, and recurs every
// interval during non-speech.
//
// Time is counted in windows of audio rather than by the wall clock, so where
// resets land depends only on the audio.
func (s *Silero) maybeResetModel(confidence float64) {
	s.windowsSinceReset++
	if confidence < modelResetMaxConfidence {
		s.lowConfidenceWindows++
	} else {
		s.lowConfidenceWindows = 0
	}

	if float64(s.windowsSinceReset*s.numFramesRequired()) >= modelResetIntervalSecs*float64(s.sampleRate) &&
		s.stateMachine.state == StateQuiet &&
		s.lowConfidenceWindows >= modelResetLowConfidenceWindows {
		s.resetModel()
		s.windowsSinceReset = 0
	}
}

// run executes one inference pass, feeding the recurrent state back in and
// reading the updated state out.
func (s *Silero) run(input []float32) (float64, error) {
	outs, err := s.session.Run([]onnxrt.Tensor{
		onnxrt.Float32([]int64{1, int64(len(input))}, input),
		onnxrt.Float32([]int64{2, 1, 128}, s.state),
		onnxrt.Int64([]int64{1}, []int64{int64(s.sampleRate)}),
	})
	if err != nil {
		return 0, err
	}
	if len(outs) != 2 || len(outs[0].F32) == 0 {
		return 0, fmt.Errorf("%w: got %d outputs", errUnexpectedTensor, len(outs))
	}
	conf := float64(outs[0].F32[0])
	copy(s.state, outs[1].F32)
	return conf, nil
}

// Reset clears the model state, the input context, the reset schedule and the
// detection state machine, as for a new stream.
func (s *Silero) Reset() {
	s.resetModel()
	s.windowsSinceReset = 0
	s.lowConfidenceWindows = 0
	if s.stateMachine != nil {
		s.reset()
	}
}

func (s *Silero) resetModel() {
	for i := range s.state {
		s.state[i] = 0
	}
	s.context = make([]float32, s.contextSize)
}

// Close releases the model session if it is owned by this analyzer.
// Analyzers sharing a model session do not close the underlying session.
func (s *Silero) Close() error {
	if s.session == nil || s.sharedSession {
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

var _ Analyzer = (*Silero)(nil)
