package turns

import (
	"log/slog"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// defaultUserSpeechTimeout is the policy floor a speech-timeout stop waits
// after the VAD reports the user stopped.
const defaultUserSpeechTimeout = 600 * time.Millisecond

// warnStopWindow reports a stop window the STT latencies were not measured
// against, and one wide enough to leave no transcript wait at all. Both show up
// as turns that end late, and both are settings rather than anything about this
// turn, so each is said once.
func warnStopWindow(warned *bool, stopWindow, sttTimeout time.Duration) {
	if *warned {
		return
	}
	recommended := time.Duration(vad.DefaultStopSecs * float64(time.Second))
	if stopWindow != recommended {
		*warned = true
		slog.Warn("turns: the VAD stop window is not the one the STT latencies were "+
			"measured with; re-measure and set TTFSP99 on the STT config",
			"stop_window", stopWindow, "recommended", recommended)
	}
	if sttTimeout > 0 && stopWindow >= sttTimeout {
		*warned = true
		slog.Warn("turns: the VAD stop window covers the STT p99 latency, leaving no "+
			"transcript wait; a turn now ends on the stop timeout",
			"stop_window", stopWindow, "stt_p99", sttTimeout)
	}
}

// TurnAnalyzerConfig configures a TurnAnalyzerStop strategy.
type TurnAnalyzerConfig struct {
	// Analyzer is the end-of-turn model (e.g. Smart Turn V3). Required.
	Analyzer turn.Analyzer
	// WaitForTranscript holds the turn open until a transcript arrives; nil
	// defaults to true. Set false for realtime services that bypass STT.
	WaitForTranscript *bool
}

// analyzerMetricsProcessor labels the metrics an end-of-turn analyzer produces.
const analyzerMetricsProcessor = "TurnAnalyzer"

// TurnAnalyzerStop ends a turn using an end-of-turn model fed the user's audio,
// gated on a finalized transcript (or a safety-net timeout). This is the
// Smart-Turn stop strategy.
type TurnAnalyzerStop struct {
	StopStrategyBase
	analyzer turn.Analyzer
	// analyzerErr is why there is no analyzer, for a chain built by
	// DefaultStopStrategies where the model could not be loaded. It is reported
	// from Setup, which is the first call that can report anything.
	analyzerErr error
	waitForTx   bool
	// sttTimeout is the transcript wait, the p99 the STT publishes at start. Zero
	// until it does, and zero for a service the wait means nothing for.
	sttTimeout time.Duration
	// stopWindow is the silence window the VAD required before its last stop. It
	// outlives the turn that observed it, so a transcript arriving with no VAD
	// stop behind it can still discount it from the STT budget.
	stopWindow     time.Duration
	stopSecsWarned bool

	text           string
	turnComplete   bool
	vadSpeaking    bool
	vadStopped     bool
	txFinalized    bool
	timeoutExpired bool
	cancelTimeout  func()
}

// NewTurnAnalyzerStop builds a Smart-Turn stop strategy.
func NewTurnAnalyzerStop(cfg TurnAnalyzerConfig) *TurnAnalyzerStop {
	s := &TurnAnalyzerStop{
		analyzer:  cfg.Analyzer,
		waitForTx: boolOr(cfg.WaitForTranscript, true),
	}
	s.EnableUserSpeakingFrames = true
	return s
}

// Process feeds the analyzer and decides end-of-turn.
func (s *TurnAnalyzerStop) Process(f frames.Frame) ProcessFrameResult {
	if s.analyzer == nil {
		// There is no model, and Setup has already reported why. Nothing here
		// can decide a turn, so every frame passes to the rest of the chain.
		return Continue
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Publish the end-of-turn parameters the pipeline is running under, so a
		// processor downstream can size its own behavior to them and clients and
		// observers can mirror them.
		s.Broadcast(func() frames.Frame {
			params := s.analyzer.Params()
			return frames.NewSpeechControlParamsFrame(nil, &params)
		})
	case *frames.STTMetadataFrame:
		s.sttTimeout = fr.TTFSP99Latency
		s.stopSecsWarned = false
	case *frames.VADUserStartedSpeakingFrame:
		s.analyzer.UpdateVADStartSecs(fr.StartSecs)
		s.vadSpeaking = true
		s.discardPendingEndOfTurn()
	case *frames.InputAudioRawFrame:
		if s.analyzer.AppendAudio(fr.Audio, s.vadSpeaking) == turn.Complete {
			// A streaming analyzer decides inside AppendAudio, and a batch one
			// reports Complete here only on the silence safety net. Either way the
			// prediction is consumed now, while it is fresh.
			_, prob, err := s.analyzer.AnalyzeEndOfTurn()
			s.reportPrediction(true, prob, err)
			s.turnComplete = true
			s.maybeTrigger()
		}
	case *frames.VADUserStoppedSpeakingFrame:
		s.handleVADStopped(fr)
	case *frames.TranscriptionFrame:
		s.handleTranscription(fr)
	case *frames.InterimTranscriptionFrame:
		// An interim means more transcription is still on the way, so an earlier
		// finalized transcript no longer covers all of the user's speech. Without
		// this, a transcript finalized during a pause too short for the VAD to
		// report a stop (and so a new start, which is what normally clears the
		// flag) would leave the flag stale and trigger the turn at the next VAD
		// stop while the tail of the utterance is still in flight. STT endpointers
		// that finalize on silences shorter than the VAD stop window do exactly
		// that.
		s.txFinalized = false
	}
	return Continue
}

// handleVADStopped consumes the analyzer's verdict for the speech that just
// ended and arms the safety net that releases the turn if no transcript lands.
func (s *TurnAnalyzerStop) handleVADStopped(fr *frames.VADUserStoppedSpeakingFrame) {
	s.vadSpeaking = false
	s.stopWindow = time.Duration(fr.StopSecs * float64(time.Second))
	s.vadStopped = true
	warnStopWindow(&s.stopSecsWarned, s.stopWindow, s.sttTimeout)

	// The STT budget is measured from the moment the user actually stopped
	// speaking, which the VAD only reports a stop window later. Anchoring the
	// safety net to that absolute deadline keeps the wait fixed however long
	// end-of-turn analysis (model inference, say) takes before the timer is
	// armed.
	deadline := s.sttDeadline(fr)

	start := time.Now()
	state, prob, err := s.analyzer.AnalyzeEndOfTurn()
	s.reportPrediction(state == turn.Complete, prob, err, start)
	s.turnComplete = err == nil && state == turn.Complete

	s.armTimeoutFor(max(0, time.Until(deadline)))

	// A finalized transcript may already have satisfied the trigger conditions
	// while the analyzer was running, as may Complete itself when waitForTx is
	// false. The verdict is only known now, so re-check here rather than wait the
	// safety net out.
	s.maybeTrigger()
}

// handleTranscription records the transcript and releases the turn, or falls
// back to an inactivity timer when no VAD stop stands behind the speech.
func (s *TurnAnalyzerStop) handleTranscription(fr *frames.TranscriptionFrame) {
	// Only whether there is text matters, not what it says.
	s.text = fr.Text

	switch {
	case fr.Finalized:
		s.txFinalized = true
		// Nothing more is coming, so release the turn now if the analyzer agrees.
		s.maybeTrigger()
	case s.timeoutExpired && s.turnComplete:
		// The safety net elapsed before the transcript arrived. Now that it has,
		// stop the turn at once instead of waiting a second time.
		s.fire()
		return
	}

	// Fallback for a transcript with no VAD stop behind it: the analyzer is never
	// asked for a verdict on speech the VAD did not bracket, so one would never
	// arrive. Assume the turn is complete and measure inactivity from this
	// transcript instead. This is also the recovery path when a verdict reached
	// just before the turn opened was discarded with the rest of the previous
	// turn's state.
	if !s.vadSpeaking && !s.vadStopped {
		s.turnComplete = true
		s.armTimeout()
	}
}

// sttDeadline returns the instant the safety net expires for the speech the VAD
// just reported the end of: the STT budget counted from the end of the speech
// itself, which is the stop window before the VAD said so. A frame carrying no
// timestamp falls back to counting from now.
func (s *TurnAnalyzerStop) sttDeadline(fr *frames.VADUserStoppedSpeakingFrame) time.Time {
	stopped := fr.Timestamp
	if stopped.IsZero() {
		stopped = time.Now()
	}
	return stopped.Add(-s.stopWindow).Add(s.sttTimeout)
}

// armTimeout restarts the safety net, measured from now over what is left of the
// STT budget once the VAD's own silence window is discounted.
func (s *TurnAnalyzerStop) armTimeout() {
	s.armTimeoutFor(max(0, s.sttTimeout-s.stopWindow))
}

// armTimeoutFor restarts the safety net over d.
func (s *TurnAnalyzerStop) armTimeoutFor(d time.Duration) {
	s.cancel()
	s.cancelTimeout = s.after(d, func() {
		s.timeoutExpired = true
		s.cancelTimeout = nil
		s.maybeTrigger()
	})
}

// reportPrediction emits what the analyzer decided, so a turn that ended on the
// safety-net timeout can be told from one the analyzer judged unfinished.
func (s *TurnAnalyzerStop) reportPrediction(complete bool, prob float64, err error, start ...time.Time) {
	if err != nil || s.env.push == nil {
		return
	}
	turn := frames.TurnMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: analyzerMetricsProcessor},
		Complete:        complete,
		Probability:     prob,
	}
	if len(start) > 0 {
		turn.E2EProcessing = time.Since(start[0])
	}
	slog.Debug("end of turn prediction",
		"complete", complete, "probability", prob, "took", turn.E2EProcessing)
	s.env.push(frames.NewMetricsFrame(turn), processor.Downstream)
}

func (s *TurnAnalyzerStop) maybeTrigger() {
	if !s.turnComplete {
		return
	}
	if !s.waitForTx {
		// The analyzer drives turn-end; transcripts are bookkeeping.
		s.fire()
		return
	}
	if s.text == "" {
		return
	}
	if s.txFinalized {
		s.fire()
		return
	}
	// Non-finalized: release only once no safety net is pending, meaning it has
	// already elapsed or none was ever armed.
	if s.cancelTimeout == nil {
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

// TurnStarted resets the bookkeeping but keeps the analyzer's buffered
// pre-speech audio for the turn now beginning.
func (s *TurnAnalyzerStop) TurnStarted() { s.resetState() }

// TurnStopped resets the bookkeeping and clears the analyzer's buffered speech.
func (s *TurnAnalyzerStop) TurnStopped() {
	s.resetState()
	if s.analyzer != nil {
		s.analyzer.Clear()
	}
}

// resetState clears turn-scoped state. It runs at both turn boundaries.
//
// vadSpeaking is deliberately left alone: whether the user is speaking belongs
// to the user, not to the turn, and the VAD reports it only on transitions.
// Clearing it at a turn start the VAD did not drive (a turn opened from a
// transcript, mid-utterance) would leave it wrong until the user next stopped.
func (s *TurnAnalyzerStop) resetState() {
	s.text = ""
	s.discardPendingEndOfTurn()
}

// discardPendingEndOfTurn drops whatever end-of-turn conclusion has been reached
// so far. It runs at a turn boundary, and whenever the VAD reports the user
// speaking again, which makes an earlier conclusion stale mid-turn.
func (s *TurnAnalyzerStop) discardPendingEndOfTurn() {
	s.turnComplete = false
	s.txFinalized = false
	s.timeoutExpired = false
	s.vadStopped = false
	s.cancel()
}

// Setup tells the analyzer the pipeline's input rate, which is known before any
// audio arrives. A chain whose analyzer could not be built fails here, which is
// the first point that can say so.
func (s *TurnAnalyzerStop) Setup(st processor.Setup) error {
	if s.analyzerErr != nil {
		return s.analyzerErr
	}
	s.analyzer.SetSampleRate(st.AudioInSampleRate)
	return nil
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
	// sttTimeout is the transcript wait, the p99 the STT publishes at start. Zero
	// until it does, and zero for a service the wait means nothing for.
	sttTimeout time.Duration
	// stopWindow is the silence window the VAD required before its last stop.
	stopWindow     time.Duration
	stopSecsWarned bool

	haveText       bool
	vadSpeaking    bool
	vadStopped     bool
	txFinalized    bool
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
		waitForTx:         boolOr(cfg.WaitForTranscript, true),
	}
	s.EnableUserSpeakingFrames = true
	return s
}

// Process runs the silence timers and decides end-of-turn.
func (s *SpeechTimeoutStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.STTMetadataFrame:
		s.sttTimeout = fr.TTFSP99Latency
		s.stopSecsWarned = false
	case *frames.VADUserStartedSpeakingFrame:
		s.vadSpeaking = true
		s.discardPendingEndOfTurn()
	case *frames.VADUserStoppedSpeakingFrame:
		s.handleVADStopped(fr)
	case *frames.TranscriptionFrame:
		s.handleTranscription(fr)
	case *frames.InterimTranscriptionFrame:
		// More transcription is still on the way, so an earlier finalized
		// transcript no longer covers all of the user's speech and must not be
		// allowed to skip the STT safety net at the next VAD stop.
		s.txFinalized = false
	}
	return Continue
}

// handleVADStopped starts the silence timers the turn now waits on.
func (s *SpeechTimeoutStop) handleVADStopped(fr *frames.VADUserStoppedSpeakingFrame) {
	s.vadSpeaking = false
	s.stopWindow = time.Duration(fr.StopSecs * float64(time.Second))
	s.vadStopped = true
	warnStopWindow(&s.stopSecsWarned, s.stopWindow, s.sttTimeout)

	// The speech timeout is the policy floor and always runs. Any earlier run of
	// it, from the fallback below, is superseded here.
	s.restartUserSpeechTimer()

	// The STT wait is only a safety net. Skip it when the transcript is already
	// finalized, or when the VAD's own silence window already covered it.
	s.sttDone = false
	if s.txFinalized || s.sttWait() <= 0 {
		s.sttDone = true
		return
	}
	s.cancelSTT = s.after(s.sttWait(), func() {
		s.sttDone = true
		s.cancelSTT = nil
		s.maybeTrigger()
	})
}

// handleTranscription records the transcript and either releases the turn or,
// with no VAD stop behind it, measures inactivity from the transcript itself.
func (s *SpeechTimeoutStop) handleTranscription(fr *frames.TranscriptionFrame) {
	if fr.Text != "" {
		s.haveText = true
	}
	if fr.Finalized {
		s.txFinalized = true
		// STT says nothing more is coming, so the safety net has nothing left to
		// wait for.
		if !s.sttDone {
			s.sttDone = true
			s.cancelSTTTimer()
		}
	}

	// Both waits already done means the turn was only waiting on text.
	if s.userSpeechDone && s.sttDone {
		s.maybeTrigger()
		return
	}

	// Fallback for a transcript with no VAD stop behind it: measure inactivity
	// since the last transcript instead. The STT wait is defined relative to a VAD
	// stop and means nothing here, so it is marked done at once.
	if !s.vadSpeaking && !s.vadStopped {
		s.sttDone = true
		s.restartUserSpeechTimer()
	}
}

// sttWait is what is left of the STT budget once the VAD's own silence window is
// discounted.
func (s *SpeechTimeoutStop) sttWait() time.Duration { return s.sttTimeout - s.stopWindow }

// restartUserSpeechTimer cancels any running speech timer and starts a fresh one.
func (s *SpeechTimeoutStop) restartUserSpeechTimer() {
	if s.cancelUser != nil {
		s.cancelUser()
		s.cancelUser = nil
	}
	s.userSpeechDone = false
	s.cancelUser = s.after(s.userSpeechTimeout, func() {
		s.userSpeechDone = true
		s.cancelUser = nil
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

// reset clears turn-scoped state. clearVADSpeaking is set only at the end of a
// turn: the flag reflects the live VAD state rather than anything turn-scoped,
// and the VAD re-emits a start only after a stop. Clearing it at a turn start the
// VAD did not drive would leave the strategy believing there is no VAD reference
// at all, and treating every transcript as a standalone utterance.
func (s *SpeechTimeoutStop) reset(clearVADSpeaking bool) {
	s.haveText = false
	if clearVADSpeaking {
		s.vadSpeaking = false
	}
	s.discardPendingEndOfTurn()
}

// discardPendingEndOfTurn drops whatever progress toward an end-of-turn has been
// made. It runs at a turn boundary, and whenever the VAD reports the user
// speaking again, which makes earlier progress stale mid-turn.
func (s *SpeechTimeoutStop) discardPendingEndOfTurn() {
	s.txFinalized = false
	s.vadStopped = false
	s.userSpeechDone = false
	s.sttDone = false
	s.cancelTimers()
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

// TurnStarted readies the strategy to detect the end of the turn now starting.
func (s *SpeechTimeoutStop) TurnStarted() { s.reset(false) }

// TurnStopped clears per-turn state once the turn has ended.
func (s *SpeechTimeoutStop) TurnStopped() { s.reset(true) }

// Cleanup stops the timers.
func (s *SpeechTimeoutStop) Cleanup() { s.cancelTimers() }

// ExternalStopConfig configures an ExternalStop strategy.
type ExternalStopConfig struct {
	// Timeout is the short delay used internally to handle consecutive or
	// slightly delayed transcriptions; 0 uses 500ms.
	Timeout time.Duration
	// WaitForTranscript holds the turn open until transcript text arrives after
	// the external stop signal; nil defaults to true. Set it false when local
	// turn detection is the intended driver of the conversation, so transcripts
	// are off the latency critical path.
	WaitForTranscript *bool
}

// ExternalStop takes its cue for the end of a turn from another processor. It is
// the counterpart to ExternalStart and takes the same two signals:
//
//   - A ProposedUserStoppedSpeakingFrame is a service proposing that the turn
//     has ended. This strategy decides, and emits the UserStoppedSpeakingFrame
//     itself. It may also hold the turn open past the proposal, which is what
//     WaitForTranscript does.
//   - A UserStoppedSpeakingFrame means the turn end was already decided and
//     announced elsewhere. This strategy adopts that decision and emits nothing.
//
// To shift the timing further, embed it and override the finalization both paths
// reach once they decide the turn is over.
type ExternalStop struct {
	StopStrategyBase
	timeout   time.Duration
	waitForTx bool

	text         string
	userSpeaking bool
	seenInterim  bool
	// announcedElsewhere records which of the two signals ended this turn, so
	// finalization knows whether the turn frame still has to be emitted. It is
	// read rather than passed along, because finalization can land from the
	// transcript timer long after the signal that set it.
	announcedElsewhere bool
	// turnOpen keeps the timer quiet between turns. With the transcript wait off
	// the timer path would otherwise fire on every tick forever.
	turnOpen    bool
	cancelTimer func()
}

// NewExternalStop builds an external stop strategy.
func NewExternalStop(cfg ExternalStopConfig) *ExternalStop {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 500 * time.Millisecond
	}
	s := &ExternalStop{timeout: timeout, waitForTx: boolOr(cfg.WaitForTranscript, true)}
	s.EnableUserSpeakingFrames = true
	return s
}

// ResolvesProposedTurnStopFrames reports that this strategy resolves proposals
// into turn stops.
func (s *ExternalStop) ResolvesProposedTurnStopFrames() bool { return true }

// Setup starts the timer that retries finalization while a turn is open, so a
// transcript arriving after the stop signal still ends the turn.
func (s *ExternalStop) Setup(processor.Setup) error {
	// Under the shared lock, because strategies can be swapped on a running
	// pipeline and the timer this arms runs under it.
	s.env.locked(s.armTick)
	return nil
}

// Process records the external signals and the transcripts around them. It
// always continues, so the rest of the stop chain is evaluated.
func (s *ExternalStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.ProposedUserStartedSpeakingFrame:
		s.handleStartedSpeaking(false)
	case *frames.ProposedUserStoppedSpeakingFrame:
		s.handleStoppedSpeaking(false)
	case *frames.UserStartedSpeakingFrame:
		s.handleStartedSpeaking(true)
	case *frames.UserStoppedSpeakingFrame:
		s.handleStoppedSpeaking(true)
	case *frames.InterimTranscriptionFrame:
		s.seenInterim = true
	case *frames.TranscriptionFrame:
		s.text += fr.Text
		// A final result has landed, so the interim it supersedes is spent.
		s.seenInterim = false
		// Restart the aggregation timer.
		s.armTick()
	}
	return Continue
}

func (s *ExternalStop) handleStartedSpeaking(announcedElsewhere bool) {
	s.userSpeaking = true
	s.announcedElsewhere = announcedElsewhere
}

func (s *ExternalStop) handleStoppedSpeaking(announcedElsewhere bool) {
	s.userSpeaking = false
	s.announcedElsewhere = announcedElsewhere
	s.maybeTrigger()
}

// armTick restarts the retry timer. It fires repeatedly while a turn is open, so
// finalization that could not go through when the stop signal arrived is retried
// once the transcript lands.
func (s *ExternalStop) armTick() {
	s.cancelT()
	s.cancelTimer = s.after(s.timeout, func() {
		s.cancelTimer = nil
		if s.turnOpen {
			s.maybeTrigger()
		}
		s.armTick()
	})
}

func (s *ExternalStop) maybeTrigger() {
	if s.userSpeaking {
		return
	}
	if !s.waitForTx {
		// Fire as soon as the external stop signal arrives: transcripts, if any,
		// are off the latency critical path.
		s.finalize()
		return
	}
	if !s.seenInterim && s.text != "" {
		s.finalize()
	}
}

// finalize ends the turn, emitting the turn frame unless it was already
// announced elsewhere.
func (s *ExternalStop) finalize() {
	if s.announcedElsewhere {
		slog.Debug("turns: adopting a user turn stop decided elsewhere")
		off := false
		s.TriggerStoppedOverriding(StoppedOverrides{EnableUserSpeakingFrames: &off})
		return
	}
	slog.Debug("turns: resolving a proposed user turn stop")
	s.TriggerStopped()
}

func (s *ExternalStop) cancelT() {
	if s.cancelTimer != nil {
		s.cancelTimer()
		s.cancelTimer = nil
	}
}

// TurnStarted readies the strategy to detect the end of the turn now starting.
func (s *ExternalStop) TurnStarted() {
	s.reset()
	s.turnOpen = true
}

// TurnStopped clears per-turn state once the turn has ended.
func (s *ExternalStop) TurnStopped() {
	s.reset()
	s.turnOpen = false
}

// reset clears per-turn state. It runs at both turn boundaries.
func (s *ExternalStop) reset() {
	s.text = ""
	s.userSpeaking = false
	s.seenInterim = false
	s.announcedElsewhere = false
	s.armTick()
}

// Cleanup stops the retry timer.
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

func (d *deferredStop) attach(_ StopStrategy, env strategyEnv) {
	env.stopped = nil // suppress finalization
	// The wrapped strategy is what decides, so it is what the decisions are
	// attributed to, not this wrapper.
	d.inner.attach(d.inner, env)
}

// ResolvesProposedTurnStopFrames reports what the wrapped strategy does with
// proposals. Frame processing is forwarded, so a proposal reaches the inner
// strategy and is resolved there: deferring finalization changes when the turn
// ends, not who decides it.
func (d *deferredStop) ResolvesProposedTurnStopFrames() bool {
	return d.inner.ResolvesProposedTurnStopFrames()
}

func (d *deferredStop) Process(f frames.Frame) ProcessFrameResult { return d.inner.Process(f) }
func (d *deferredStop) TurnStarted()                              { d.inner.TurnStarted() }
func (d *deferredStop) TurnStopped()                              { d.inner.TurnStopped() }
func (d *deferredStop) Cleanup()                                  { d.inner.Cleanup() }

func (d *deferredStop) Setup(s processor.Setup) error { return d.inner.Setup(s) }
