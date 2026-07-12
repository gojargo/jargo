package turns

import (
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
)

const (
	// defaultSTTTimeout is the safety-net wait for a finalized transcript after
	// speech stops, used when no STTMetadataFrame reports the real p99 latency.
	defaultSTTTimeout = 2 * time.Second
	// defaultUserSpeechTimeout is the policy floor a speech-timeout stop waits
	// after the VAD reports the user stopped.
	defaultUserSpeechTimeout = 600 * time.Millisecond
)

// boolValue returns *p, or def when p is nil.
func boolValue(p *bool, def bool) bool {
	if p == nil {
		return def
	}
	return *p
}

// TurnAnalyzerConfig configures a TurnAnalyzerStop strategy.
type TurnAnalyzerConfig struct {
	// Analyzer is the end-of-turn model (e.g. Smart Turn V3). Required.
	Analyzer turn.Analyzer
	// WaitForTranscript holds the turn open until a transcript arrives; nil
	// defaults to true. Set false for realtime services that bypass STT.
	WaitForTranscript *bool
}

// TurnAnalyzerStop ends a turn using an end-of-turn model fed the user's audio,
// gated on a finalized transcript (or a safety-net timeout). This is the
// Smart-Turn stop strategy.
type TurnAnalyzerStop struct {
	StopStrategyBase
	analyzer       turn.Analyzer
	waitForTx      bool
	sttTimeout     time.Duration
	vadSpeaking    bool
	turnComplete   bool
	txFinalized    bool
	timeoutExpired bool
	haveText       bool
	cancelTimeout  func()
}

// NewTurnAnalyzerStop builds a Smart-Turn stop strategy.
func NewTurnAnalyzerStop(cfg TurnAnalyzerConfig) *TurnAnalyzerStop {
	s := &TurnAnalyzerStop{
		analyzer:   cfg.Analyzer,
		waitForTx:  boolValue(cfg.WaitForTranscript, true),
		sttTimeout: defaultSTTTimeout,
	}
	s.EnableUserSpeakingFrames = true
	return s
}

// Process feeds the analyzer and decides end-of-turn.
func (s *TurnAnalyzerStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.StartFrame:
		s.analyzer.SetSampleRate(fr.AudioInSampleRate)
	case *frames.STTMetadataFrame:
		if fr.TTFSP99Latency > 0 {
			s.sttTimeout = fr.TTFSP99Latency
		}
	case *frames.VADUserStartedSpeakingFrame:
		s.analyzer.UpdateVADStartSecs(fr.StartSecs)
		s.vadSpeaking = true
		s.turnComplete = false
		s.txFinalized = false
		s.timeoutExpired = false
		s.cancel()
	case *frames.InputAudioRawFrame:
		if s.analyzer.AppendAudio(fr.Audio, s.vadSpeaking) == turn.Complete {
			s.turnComplete = true
			s.maybeTrigger()
		}
	case *frames.VADUserStoppedSpeakingFrame:
		s.vadSpeaking = false
		state, _, err := s.analyzer.AnalyzeEndOfTurn()
		s.turnComplete = err == nil && state == turn.Complete
		wait := max(0, s.sttTimeout-time.Duration(fr.StopSecs*float64(time.Second)))
		s.cancel()
		s.cancelTimeout = s.after(wait, func() {
			s.timeoutExpired = true
			s.cancelTimeout = nil
			s.maybeTrigger()
		})
	case *frames.TranscriptionFrame:
		if fr.Text != "" {
			s.haveText = true
		}
		if fr.Finalized {
			s.txFinalized = true
		}
		s.maybeTrigger()
	}
	return Continue
}

func (s *TurnAnalyzerStop) maybeTrigger() {
	if !s.turnComplete {
		return
	}
	if !s.waitForTx {
		s.fire()
		return
	}
	if !s.haveText {
		return
	}
	if s.txFinalized || s.timeoutExpired {
		s.fire()
	}
}

func (s *TurnAnalyzerStop) fire() {
	s.cancel()
	s.TriggerStopped()
}

func (s *TurnAnalyzerStop) cancel() {
	if s.cancelTimeout != nil {
		s.cancelTimeout()
		s.cancelTimeout = nil
	}
}

// Reset clears per-turn state.
func (s *TurnAnalyzerStop) Reset() {
	s.cancel()
	s.turnComplete = false
	s.txFinalized = false
	s.timeoutExpired = false
	s.haveText = false
	s.analyzer.Clear()
}

// Cleanup stops the timeout.
func (s *TurnAnalyzerStop) Cleanup() { s.cancel() }

// SpeechTimeoutConfig configures a SpeechTimeoutStop strategy.
type SpeechTimeoutConfig struct {
	// UserSpeechTimeout is the silence the user gets to resume after the VAD
	// stops; 0 uses 600ms.
	UserSpeechTimeout time.Duration
	// WaitForTranscript holds the turn open until a transcript arrives; nil
	// defaults to true.
	WaitForTranscript *bool
}

// SpeechTimeoutStop ends a turn purely on silence timers after the VAD reports
// the user stopped — no model. It is the model-free default stop strategy.
type SpeechTimeoutStop struct {
	StopStrategyBase
	userSpeechTimeout time.Duration
	waitForTx         bool
	sttTimeout        time.Duration

	vadSpeaking    bool
	haveText       bool
	userSpeechDone bool
	sttDone        bool
	cancelUser     func()
	cancelSTT      func()
}

// NewSpeechTimeoutStop builds a speech-timeout stop strategy.
func NewSpeechTimeoutStop(cfg SpeechTimeoutConfig) *SpeechTimeoutStop {
	timeout := cfg.UserSpeechTimeout
	if timeout == 0 {
		timeout = defaultUserSpeechTimeout
	}
	s := &SpeechTimeoutStop{
		userSpeechTimeout: timeout,
		waitForTx:         boolValue(cfg.WaitForTranscript, true),
		sttTimeout:        defaultSTTTimeout,
	}
	s.EnableUserSpeakingFrames = true
	return s
}

// Process runs the silence timers and decides end-of-turn.
func (s *SpeechTimeoutStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.STTMetadataFrame:
		if fr.TTFSP99Latency > 0 {
			s.sttTimeout = fr.TTFSP99Latency
		}
	case *frames.VADUserStartedSpeakingFrame:
		s.vadSpeaking = true
		s.reset()
	case *frames.VADUserStoppedSpeakingFrame:
		s.vadSpeaking = false
		s.startTimers(fr.StopSecs)
	case *frames.TranscriptionFrame:
		if fr.Text != "" {
			s.haveText = true
		}
		if fr.Finalized {
			s.sttDone = true
			s.cancelSTTTimer()
		}
		s.maybeTrigger()
	}
	return Continue
}

func (s *SpeechTimeoutStop) startTimers(stopSecs float64) {
	s.cancelTimers()
	s.userSpeechDone = false
	s.cancelUser = s.after(s.userSpeechTimeout, func() {
		s.userSpeechDone = true
		s.cancelUser = nil
		s.maybeTrigger()
	})
	wait := s.sttTimeout - time.Duration(stopSecs*float64(time.Second))
	if wait <= 0 {
		s.sttDone = true
		return
	}
	s.sttDone = false
	s.cancelSTT = s.after(wait, func() {
		s.sttDone = true
		s.cancelSTT = nil
		s.maybeTrigger()
	})
}

func (s *SpeechTimeoutStop) maybeTrigger() {
	if s.vadSpeaking {
		return
	}
	if s.waitForTx && !s.haveText {
		return
	}
	if s.userSpeechDone && s.sttDone {
		s.cancelTimers()
		s.TriggerStopped()
	}
}

func (s *SpeechTimeoutStop) reset() {
	s.cancelTimers()
	s.haveText = false
	s.userSpeechDone = false
	s.sttDone = false
}

func (s *SpeechTimeoutStop) cancelTimers() {
	if s.cancelUser != nil {
		s.cancelUser()
		s.cancelUser = nil
	}
	s.cancelSTTTimer()
}

func (s *SpeechTimeoutStop) cancelSTTTimer() {
	if s.cancelSTT != nil {
		s.cancelSTT()
		s.cancelSTT = nil
	}
}

// Reset clears per-turn state.
func (s *SpeechTimeoutStop) Reset() { s.reset() }

// Cleanup stops the timers.
func (s *SpeechTimeoutStop) Cleanup() { s.cancelTimers() }

// ExternalStopConfig configures an ExternalStop strategy.
type ExternalStopConfig struct {
	// Timeout debounces late transcripts after the external stop; 0 uses 500ms.
	Timeout time.Duration
	// WaitForTranscript holds the turn open until a transcript arrives; nil
	// defaults to true.
	WaitForTranscript *bool
}

// ExternalStop ends a turn when another processor emits a UserStoppedSpeakingFrame.
type ExternalStop struct {
	StopStrategyBase
	timeout      time.Duration
	waitForTx    bool
	userSpeaking bool
	haveText     bool
	seenInterim  bool
	cancelTimer  func()
}

// NewExternalStop builds an external stop strategy.
func NewExternalStop(cfg ExternalStopConfig) *ExternalStop {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 500 * time.Millisecond
	}
	s := &ExternalStop{timeout: timeout, waitForTx: boolValue(cfg.WaitForTranscript, true)}
	s.EnableUserSpeakingFrames = false
	return s
}

// Process relays an external stop, debounced for late transcripts.
func (s *ExternalStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.UserStartedSpeakingFrame:
		s.userSpeaking = true
	case *frames.UserStoppedSpeakingFrame:
		s.userSpeaking = false
		s.cancelT()
		s.cancelTimer = s.after(s.timeout, func() {
			s.cancelTimer = nil
			s.maybeTrigger()
		})
	case *frames.InterimTranscriptionFrame:
		s.seenInterim = true
	case *frames.TranscriptionFrame:
		if fr.Text != "" {
			s.haveText = true
		}
		s.seenInterim = false
	}
	return Continue
}

func (s *ExternalStop) maybeTrigger() {
	if s.userSpeaking {
		return
	}
	if !s.waitForTx {
		s.TriggerStopped()
		return
	}
	if !s.seenInterim && s.haveText {
		s.TriggerStopped()
	}
}

func (s *ExternalStop) cancelT() {
	if s.cancelTimer != nil {
		s.cancelTimer()
		s.cancelTimer = nil
	}
}

// Reset clears per-turn state.
func (s *ExternalStop) Reset() {
	s.cancelT()
	s.haveText = false
	s.seenInterim = false
}

// Cleanup stops the debounce timer.
func (s *ExternalStop) Cleanup() { s.cancelT() }

// ExternalCompletionStop finalizes a turn when an external judge emits a
// UserTurnInferenceCompletedFrame. It is the base for LLM-gated completion.
type ExternalCompletionStop struct {
	StopStrategyBase
}

// NewExternalCompletionStop builds an external-completion stop strategy.
func NewExternalCompletionStop() *ExternalCompletionStop {
	s := &ExternalCompletionStop{}
	s.EnableUserSpeakingFrames = true
	return s
}

// Process finalizes the turn on a completion frame.
func (s *ExternalCompletionStop) Process(f frames.Frame) ProcessFrameResult {
	if _, ok := f.(*frames.UserTurnInferenceCompletedFrame); ok {
		s.TriggerFinalized()
	}
	return Continue
}

// deferredStop wraps a stop strategy and drops its finalization, keeping its
// inference-triggered and frame outputs. Use Deferred to build one.
type deferredStop struct {
	inner StopStrategy
}

// Deferred wraps inner so it can drive inference-triggering but never finalize a
// turn; pair it with a finalizer such as ExternalCompletionStop.
func Deferred(inner StopStrategy) StopStrategy { return &deferredStop{inner: inner} }

func (d *deferredStop) attach(env strategyEnv) {
	env.stopped = nil // suppress finalization
	d.inner.attach(env)
}

func (d *deferredStop) Process(f frames.Frame) ProcessFrameResult { return d.inner.Process(f) }
func (d *deferredStop) Reset()                                    { d.inner.Reset() }
func (d *deferredStop) Cleanup()                                  { d.inner.Cleanup() }
