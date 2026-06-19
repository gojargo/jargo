// Package vad provides voice activity detection: it tells the pipeline when a
// user is speaking. The Analyzer interface is the contract a detector
// implements; the package ships Silero, an ONNX-based detector, as the default.
//
// An analyzer is fed raw mono 16-bit PCM and runs a confidence model over
// fixed-size frames, smoothing the result through a small state machine
// (quiet → starting → speaking → stopping → quiet) so brief dips or spikes do
// not flip the speaking decision. The turntaking package drives an analyzer and
// turns its state transitions into speaking and interruption frames.
package vad

import "math"

// State is the voice-activity state of the audio stream.
type State int

const (
	// StateQuiet means no voice activity is detected.
	StateQuiet State = iota + 1
	// StateStarting means voice activity has begun but is not yet confirmed.
	StateStarting
	// StateSpeaking means voice activity is confirmed and ongoing.
	StateSpeaking
	// StateStopping means voice activity is ending but not yet confirmed quiet.
	StateStopping
)

// String returns the lowercase state name.
func (s State) String() string {
	switch s {
	case StateQuiet:
		return "quiet"
	case StateStarting:
		return "starting"
	case StateSpeaking:
		return "speaking"
	case StateStopping:
		return "stopping"
	default:
		return "unknown"
	}
}

// Default detection parameters, matching the Silero defaults jargo ports.
const (
	defaultConfidence = 0.7
	defaultStartSecs  = 0.2
	defaultStopSecs   = 0.2
)

// Params configures voice activity detection.
type Params struct {
	// Confidence is the minimum model confidence in [0,1] for a frame to count
	// as speech.
	Confidence float64
	// StartSecs is how long speech must persist before the state is confirmed
	// as speaking.
	StartSecs float64
	// StopSecs is how long silence must persist before the state is confirmed
	// as quiet.
	StopSecs float64
}

// DefaultParams returns the default detection parameters.
func DefaultParams() Params {
	return Params{Confidence: defaultConfidence, StartSecs: defaultStartSecs, StopSecs: defaultStopSecs}
}

// Analyzer detects voice activity in a mono 16-bit PCM stream.
type Analyzer interface {
	// SetSampleRate sets the stream sample rate in Hz and resets state. It is
	// called once before analysis begins and returns an error if the rate is
	// unsupported.
	SetSampleRate(sampleRate int) error
	// AnalyzeAudio feeds a chunk of mono 16-bit PCM and returns the resulting
	// state. Chunks need not align to the analyzer's frame size; audio is
	// buffered internally.
	AnalyzeAudio(buffer []byte) State
	// Params returns the detection parameters.
	Params() Params
	// Reset clears all internal state.
	Reset()
	// Close releases any resources (for example a model session).
	Close() error
}

// confidencer is the model-specific half of an analyzer: it sizes the analysis
// frame and scores a frame for voice activity. The stateMachine drives it.
type confidencer interface {
	numFramesRequired() int
	voiceConfidence(buffer []byte) float64
}

// stateMachine is the model-agnostic detection logic shared by analyzers. A
// concrete analyzer embeds it and supplies itself as the confidencer, so the
// machine can score frames without inheritance. Gating is confidence-only:
// jargo relies on the neural model's confidence rather than an additional
// volume threshold.
type stateMachine struct {
	self   confidencer
	params Params

	sampleRate    int
	frameNumBytes int
	startFrames   int
	stopFrames    int
	startingCount int
	stoppingCount int
	state         State
	buf           []byte
}

func newStateMachine(self confidencer, params Params) *stateMachine {
	return &stateMachine{self: self, params: params, state: StateQuiet}
}

// Params implements part of Analyzer.
func (m *stateMachine) Params() Params { return m.params }

// setSampleRate recomputes the frame size and the start/stop frame counts. The
// concrete analyzer calls it from its SetSampleRate after validating the rate.
func (m *stateMachine) setSampleRate(sampleRate int) {
	m.sampleRate = sampleRate
	frames := m.self.numFramesRequired()
	m.frameNumBytes = frames * 2 // mono, 16-bit
	framesPerSec := float64(frames) / float64(sampleRate)
	m.startFrames = int(math.Round(m.params.StartSecs / framesPerSec))
	m.stopFrames = int(math.Round(m.params.StopSecs / framesPerSec))
	m.reset()
}

func (m *stateMachine) reset() {
	m.startingCount = 0
	m.stoppingCount = 0
	m.state = StateQuiet
	m.buf = m.buf[:0]
}

// advance moves the state machine by one analysis frame given whether that
// frame scored as speech.
func (m *stateMachine) advance(speaking bool) {
	if speaking {
		switch m.state {
		case StateQuiet:
			m.state = StateStarting
			m.startingCount = 1
		case StateStarting:
			m.startingCount++
		case StateStopping:
			m.state = StateSpeaking
			m.stoppingCount = 0
		case StateSpeaking:
			// Already speaking; nothing to confirm.
		}
		return
	}
	switch m.state {
	case StateStarting:
		m.state = StateQuiet
		m.startingCount = 0
	case StateSpeaking:
		m.state = StateStopping
		m.stoppingCount = 1
	case StateStopping:
		m.stoppingCount++
	case StateQuiet:
		// Already quiet; nothing to confirm.
	}
}

// AnalyzeAudio buffers the chunk and advances the state machine over every
// complete frame it now holds, returning the resulting state.
func (m *stateMachine) AnalyzeAudio(buffer []byte) State {
	m.buf = append(m.buf, buffer...)
	if len(m.buf) < m.frameNumBytes {
		return m.state
	}

	for len(m.buf) >= m.frameNumBytes {
		frame := m.buf[:m.frameNumBytes]
		m.buf = m.buf[m.frameNumBytes:]
		m.advance(m.self.voiceConfidence(frame) >= m.params.Confidence)
	}

	if m.state == StateStarting && m.startingCount >= m.startFrames {
		m.state = StateSpeaking
		m.startingCount = 0
	}
	if m.state == StateStopping && m.stoppingCount >= m.stopFrames {
		m.state = StateQuiet
		m.stoppingCount = 0
	}

	// Compact the buffer so consumed frames do not pin the backing array.
	if len(m.buf) == 0 {
		m.buf = m.buf[:0]
	}
	return m.state
}
