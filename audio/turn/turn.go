// Package turn provides end-of-turn detection: it decides when a user has
// actually finished speaking, as opposed to merely pausing. The Analyzer
// interface is the contract; the package ships SmartTurnV3, an ONNX model that
// predicts turn completion from the audio of the user's turn, as the default.
//
// Unlike a voice activity detector, which reacts to silence, a turn analyzer
// listens to the shape of speech — intonation, trailing words — to tell a
// mid-sentence pause from a finished thought. The turntaking package feeds it
// the user's audio and asks it, when speech stops, whether the turn is over.
package turn

// EndOfTurnState is the result of end-of-turn analysis.
type EndOfTurnState int

const (
	// Incomplete means the user is likely still speaking or about to continue.
	Incomplete EndOfTurnState = iota
	// Complete means the user has finished their turn.
	Complete
)

// String returns the lowercase state name.
func (s EndOfTurnState) String() string {
	if s == Complete {
		return "complete"
	}
	return "incomplete"
}

// Default timing parameters, matching the Smart Turn defaults jargo ports.
const (
	defaultStopSecs        = 3.0
	defaultPreSpeechMs     = 500.0
	defaultMaxDurationSecs = 8.0
)

// Params configures end-of-turn analysis.
type Params struct {
	// StopSecs is the silence duration that forces a turn complete even if the
	// model has not fired, as a safety net.
	StopSecs float64
	// PreSpeechMs is how much audio before speech onset to include in the
	// analyzed segment.
	PreSpeechMs float64
	// MaxDurationSecs caps the analyzed segment length; only the most recent
	// audio is kept.
	MaxDurationSecs float64
}

// DefaultParams returns the default analysis parameters.
func DefaultParams() Params {
	return Params{StopSecs: defaultStopSecs, PreSpeechMs: defaultPreSpeechMs, MaxDurationSecs: defaultMaxDurationSecs}
}

// Analyzer decides when a user's turn has ended.
type Analyzer interface {
	// SetSampleRate sets the input sample rate in Hz. It is called once before
	// analysis begins.
	SetSampleRate(sampleRate int)
	// AppendAudio adds a chunk of the user's audio along with whether voice
	// activity considers it speech. It returns Complete when a non-model
	// condition (the silence safety net) ends the turn, otherwise Incomplete.
	AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState
	// AnalyzeEndOfTurn runs the model over the buffered turn audio and returns
	// the predicted state and the model's completion probability in [0,1].
	AnalyzeEndOfTurn() (EndOfTurnState, float64, error)
	// UpdateVADStartSecs informs the analyzer of the VAD start delay so it can
	// align its pre-speech buffering.
	UpdateVADStartSecs(secs float64)
	// Clear resets the analyzer to its initial state.
	Clear()
	// Close releases any resources (for example a model session).
	Close() error
}
