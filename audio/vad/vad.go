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

import (
	"fmt"
	"math"

	"github.com/gojargo/jargo/audio/loudness"
)

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

// Default detection parameters.
const (
	defaultConfidence = 0.7
	defaultStartSecs  = 0.2
	defaultMinVolume  = 0.6

	// DefaultStopSecs is the recommended silence window before a stop is
	// reported. It is exported because the measured STT latencies a turn
	// strategy sizes its safety net against were taken with it, so a strategy
	// can say so when a pipeline runs a different one.
	DefaultStopSecs = 0.2
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
	// MinVolume is how loud a frame must be, on the 0..1 scale audio/loudness
	// measures, to count as speech however confident the model is. It keeps a
	// confident guess at something barely audible from opening a turn.
	MinVolume float64
}

// DefaultParams returns the default detection parameters.
func DefaultParams() Params {
	return Params{
		Confidence: defaultConfidence,
		StartSecs:  defaultStartSecs,
		StopSecs:   DefaultStopSecs,
		MinVolume:  defaultMinVolume,
	}
}

// Analyzer detects voice activity in a mono 16-bit PCM stream. An analyzer is
// built by embedding *Base, which supplies the detection logic and every method
// here, and supplying a Scorer.
type Analyzer interface {
	// vadBase is provided only by Base: an analyzer is built by embedding Base,
	// which is what lets this interface grow in a minor release without breaking
	// an analyzer written outside jargo.
	vadBase()

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
	// SetParams applies new detection parameters at runtime and resets the
	// detection state, so a change takes effect from the next chunk rather than
	// part-way through the one being decided.
	SetParams(params Params)
	// Reset clears all internal state.
	Reset()
	// Close releases any resources (for example a model session).
	Close() error
}

// Scorer is the model-specific half of an analyzer: it sizes the analysis frame
// and scores a frame for voice activity. Base drives it.
type Scorer interface {
	// NumFramesRequired is the number of samples one analysis frame holds at
	// the current sample rate.
	NumFramesRequired() int
	// VoiceConfidence scores one analysis frame, exactly NumFramesRequired
	// samples of mono 16-bit PCM, as the probability in [0,1] that it is
	// speech.
	VoiceConfidence(buffer []byte) float64
}

// Base is the model-agnostic detection logic shared by analyzers. A concrete
// analyzer embeds *Base and supplies itself as the Scorer, so the base can
// score frames without inheritance. A frame counts as speech only
// when the model is confident enough about it and it is loud enough to be worth
// hearing, so a confident guess at something barely audible does not open a
// turn.
type Base struct {
	self   Scorer
	params Params

	sampleRate    int
	frameNumBytes int
	startFrames   int
	stopFrames    int
	startingCount int
	stoppingCount int
	state         State
	buf           []byte
	// volume measures the stream over a rolling window, which is what loudness
	// needs: a gating block is far longer than one analysis frame, so a frame
	// measured on its own is not a loudness reading at all.
	//
	// It and prevVolume carry across a parameter or rate change, as the
	// detection state does not: the rolling window and its smoothing follow the
	// audio stream, which is continuous across such a change.
	volume loudness.Tracker
	// prevVolume is the running volume each new reading is smoothed against.
	prevVolume float64
}

// NewBase builds the detection logic for an analyzer that scores frames with
// scorer, usually the analyzer embedding it.
func NewBase(scorer Scorer, params Params) *Base {
	return &Base{self: scorer, params: params, state: StateQuiet}
}

// vadBase seals Analyzer: a type is an analyzer by embedding Base.
func (*Base) vadBase() {}

// SampleRate is the stream sample rate in Hz, zero until SetSampleRate is
// called.
func (m *Base) SampleRate() int { return m.sampleRate }

// Close releases nothing: the base holds no resources. An analyzer holding a
// model session overrides it.
func (*Base) Close() error { return nil }

// Params implements part of Analyzer.
func (m *Base) Params() Params { return m.params }

// SetParams implements part of Analyzer. It recomputes the frame counts the
// parameters size and resets the detection state, so the new values apply from
// a clean start rather than to a decision already part-way made.
func (m *Base) SetParams(params Params) {
	m.params = params
	if m.sampleRate > 0 {
		_ = m.SetSampleRate(m.sampleRate)
	}
}

// SetSampleRate sets the stream sample rate, recomputes the frame size and the
// start and stop frame counts, and resets the detection state. An analyzer that
// supports only some rates validates the rate in its own SetSampleRate and then
// calls this one.
func (m *Base) SetSampleRate(sampleRate int) error {
	if sampleRate <= 0 {
		return fmt.Errorf("%w: got %d", errInvalidSampleRate, sampleRate)
	}
	m.sampleRate = sampleRate
	frames := m.self.NumFramesRequired()
	m.frameNumBytes = frames * 2 // mono, 16-bit
	framesPerSec := float64(frames) / float64(sampleRate)
	m.startFrames = int(math.Round(m.params.StartSecs / framesPerSec))
	m.stopFrames = int(math.Round(m.params.StopSecs / framesPerSec))
	m.resetDetection()
	return nil
}

// resetDetection returns the decision to its starting point without touching the
// audio waiting to be decided on. Changing a parameter mid-stream restarts the
// decision, but the stream itself carries on, so the partial frame already
// buffered is still the audio that comes next.
func (m *Base) resetDetection() {
	m.startingCount = 0
	m.stoppingCount = 0
	m.state = StateQuiet
}

// Reset clears everything the base holds, the buffered audio included. It is
// for a break in the stream rather than a change of mind about it.
func (m *Base) Reset() {
	m.resetDetection()
	m.buf = m.buf[:0]
}

// advance moves the state machine by one analysis frame given whether that
// frame scored as speech.
func (m *Base) advance(speaking bool) {
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
func (m *Base) AnalyzeAudio(buffer []byte) State {
	m.buf = append(m.buf, buffer...)
	if len(m.buf) < m.frameNumBytes {
		return m.state
	}

	for len(m.buf) >= m.frameNumBytes {
		frame := m.buf[:m.frameNumBytes]
		m.buf = m.buf[m.frameNumBytes:]
		confidence := m.self.VoiceConfidence(frame)
		m.volume.Update(frame, m.sampleRate)
		m.prevVolume = loudness.Smooth(m.volume.Volume(), m.prevVolume, loudness.SmoothingFactor)
		m.advance(confidence >= m.params.Confidence && m.prevVolume >= m.params.MinVolume)
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
