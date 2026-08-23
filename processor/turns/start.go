package turns

import (
	"log/slog"
	"regexp"
	"strings"
	"time"

	"github.com/gojargo/jargo/frames"
)

// wakePhraseDefaultTimeout is how long a wake-phrase session stays awake without
// activity before requiring the phrase again.
const wakePhraseDefaultTimeout = 10 * time.Second

// wakeAccumLimit caps the wake-phrase match accumulator.
const wakeAccumLimit = 250

// VADStart opens a user turn as soon as the VAD reports speech.
type VADStart struct {
	StartStrategyBase
}

// NewVADStart builds a VAD-based start strategy.
func NewVADStart() *VADStart {
	s := &VADStart{}
	s.EnableInterruptions = true
	s.EnableUserSpeakingFrames = true
	return s
}

// Process triggers the turn on a VAD speech-start.
func (s *VADStart) Process(f frames.Frame) ProcessFrameResult {
	if _, ok := f.(*frames.VADUserStartedSpeakingFrame); ok {
		s.TriggerStarted()
		return Stop
	}
	return Continue
}

// TranscriptionStartConfig configures a TranscriptionStart strategy.
type TranscriptionStartConfig struct {
	// UseInterim also triggers on interim transcripts; nil defaults to true.
	UseInterim *bool
}

// TranscriptionStart opens a turn on a transcript, a fallback for soft speech a
// VAD misses.
type TranscriptionStart struct {
	StartStrategyBase
	useInterim bool
}

// NewTranscriptionStart builds a transcription-based start strategy.
func NewTranscriptionStart(cfg TranscriptionStartConfig) *TranscriptionStart {
	s := &TranscriptionStart{useInterim: cfg.UseInterim == nil || *cfg.UseInterim}
	s.EnableInterruptions = true
	s.EnableUserSpeakingFrames = true
	return s
}

// Process triggers the turn on a transcript.
func (s *TranscriptionStart) Process(f frames.Frame) ProcessFrameResult {
	switch f.(type) {
	case *frames.TranscriptionFrame:
		s.TriggerStarted()
		return Stop
	case *frames.InterimTranscriptionFrame:
		if s.useInterim {
			s.TriggerStarted()
			return Stop
		}
	}
	return Continue
}

// MinWordsStartConfig configures a MinWordsStart strategy.
type MinWordsStartConfig struct {
	// MinWords is the word count required to open a turn while the bot is
	// speaking (to gate barge-in); a single word suffices when the bot is silent.
	MinWords int
	// UseInterim counts interim transcripts too; nil defaults to true.
	UseInterim *bool
}

// MinWordsStart opens a turn only once enough words are heard, raising the bar
// for interrupting the bot.
type MinWordsStart struct {
	StartStrategyBase
	minWords    int
	useInterim  bool
	botSpeaking bool
}

// NewMinWordsStart builds a min-words start strategy.
func NewMinWordsStart(cfg MinWordsStartConfig) *MinWordsStart {
	s := &MinWordsStart{minWords: cfg.MinWords, useInterim: cfg.UseInterim == nil || *cfg.UseInterim}
	s.EnableInterruptions = true
	s.EnableUserSpeakingFrames = true
	return s
}

// TurnStarted clears the bot-speaking flag: the turn that just started will have
// interrupted the bot, so the rest of it counts against the single-word threshold
// without waiting for the bot-stopped frame to catch up.
func (s *MinWordsStart) TurnStarted() { s.botSpeaking = false }

// Process counts words and triggers once the threshold is met.
func (s *MinWordsStart) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.BotStartedSpeakingFrame:
		s.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		s.botSpeaking = false
	case *frames.TranscriptionFrame:
		return s.check(fr.Text)
	case *frames.InterimTranscriptionFrame:
		if s.useInterim {
			return s.check(fr.Text)
		}
	}
	return Continue
}

func (s *MinWordsStart) check(text string) ProcessFrameResult {
	threshold := 1
	if s.botSpeaking {
		threshold = s.minWords
	}
	if len(strings.Fields(text)) >= threshold {
		s.TriggerStarted()
		return Stop
	}
	s.TriggerResetAggregation()
	return Continue
}

// WakePhraseStartConfig configures a WakePhraseStart strategy.
type WakePhraseStartConfig struct {
	// Phrases are the wake phrases (case-insensitive, whitespace-flexible).
	Phrases []string
	// Timeout is how long the session stays awake without activity; 0 uses 10s.
	// In timeout mode the timer resets on activity (user or bot speech). In
	// single-activation mode it acts as a keepalive window: the strategy stays
	// awake for this long after the phrase is detected, which is what lets the
	// turn it opened run to completion before it sleeps again.
	Timeout time.Duration
	// SingleActivation requires the phrase again for every turn: the strategy
	// returns to sleep once the keepalive window closes.
	SingleActivation bool
	// OnWakePhraseDetected is called with the phrase that matched. It runs with
	// the turn lock held, so it must not block or push frames.
	OnWakePhraseDetected func(phrase string)
	// OnWakePhraseTimeout is called when the inactivity timeout expires and the
	// strategy goes back to sleep. Same rules as OnWakePhraseDetected.
	OnWakePhraseTimeout func()
}

// WakePhraseStart gates a turn behind a spoken wake phrase. Place it first in
// the start chain: while asleep it blocks the other start strategies; once awake
// it lets them run until an inactivity timeout puts it back to sleep.
//
// Use SingleActivation to require the phrase before every turn.
type WakePhraseStart struct {
	StartStrategyBase
	phrases  []string
	patterns []*regexp.Regexp
	timeout  time.Duration
	single   bool
	detected func(phrase string)
	timedOut func()

	awake       bool
	accum       string
	cancelTimer func()
}

// NewWakePhraseStart builds a wake-phrase start strategy.
func NewWakePhraseStart(cfg WakePhraseStartConfig) *WakePhraseStart {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = wakePhraseDefaultTimeout
	}
	phrases, patterns := compileWakePatterns(cfg.Phrases)
	s := &WakePhraseStart{
		phrases:  phrases,
		patterns: patterns,
		timeout:  timeout,
		single:   cfg.SingleActivation,
		detected: cfg.OnWakePhraseDetected,
		timedOut: cfg.OnWakePhraseTimeout,
	}
	s.EnableInterruptions = true
	s.EnableUserSpeakingFrames = true
	return s
}

// compileWakePatterns builds a case-insensitive, whitespace-flexible regex per
// phrase, alongside the phrases themselves so a match can name the one it hit.
func compileWakePatterns(phrases []string) ([]string, []*regexp.Regexp) {
	var kept []string
	var out []*regexp.Regexp
	for _, p := range phrases {
		words := strings.Fields(p)
		if len(words) == 0 {
			continue
		}
		quoted := make([]string, len(words))
		for i, w := range words {
			quoted[i] = regexp.QuoteMeta(w)
		}
		kept = append(kept, p)
		out = append(out, regexp.MustCompile(`(?i)\b`+strings.Join(quoted, `\s*`)+`\b`))
	}
	return kept, out
}

// wakePunctuation is everything punctuation strips: anything that is neither a
// letter, a digit, an underscore, nor whitespace.
//
//nolint:gochecknoglobals // compiled once
var wakePunctuation = regexp.MustCompile(`[^\p{L}\p{N}_\s]`)

// Process matches the wake phrase while asleep and keeps the session alive while
// awake.
func (s *WakePhraseStart) Process(f frames.Frame) ProcessFrameResult {
	if !s.awake {
		return s.processIdle(f)
	}
	return s.processAwake(f)
}

// processIdle handles a frame while asleep. Only a final transcript is checked
// for the phrase; one that does not match has its text dropped, so speech before
// the wake phrase never reaches the conversation. Every frame returns Stop,
// which is what blocks the rest of the start chain.
func (s *WakePhraseStart) processIdle(f frames.Frame) ProcessFrameResult {
	if fr, ok := f.(*frames.TranscriptionFrame); ok {
		if s.checkWakePhrase(fr.Text) {
			s.TriggerStarted()
			return Stop
		}
		s.TriggerResetAggregation()
	}
	return Stop
}

// processAwake handles a frame while awake: it refreshes the inactivity timeout
// on activity and lets the rest of the start chain run.
//
// Single activation does not refresh. Its timeout is a keepalive window opened
// when the phrase was detected, not an inactivity timer, so activity must not
// extend it.
func (s *WakePhraseStart) processAwake(f frames.Frame) ProcessFrameResult {
	if !s.single {
		switch f.(type) {
		case *frames.UserSpeakingFrame, *frames.BotSpeakingFrame,
			*frames.TranscriptionFrame, *frames.VADUserStartedSpeakingFrame:
			s.refresh()
		}
	}
	return Continue
}

// checkWakePhrase folds text into the accumulator and reports whether a wake
// phrase is now in it.
//
// Punctuation is stripped before matching, so transcription output like
// "Hey, Jargo!" still matches the phrase "hey jargo".
func (s *WakePhraseStart) checkWakePhrase(text string) bool {
	s.accum += " " + wakePunctuation.ReplaceAllString(text, "")
	// Cap the accumulator so it cannot grow without bound. Counted in
	// characters, so a multi-byte one is never cut in half.
	if r := []rune(s.accum); len(r) > wakeAccumLimit {
		s.accum = string(r[len(r)-wakeAccumLimit:])
	}
	for i, p := range s.patterns {
		if p.MatchString(s.accum) {
			slog.Debug("turns: wake phrase detected", "phrase", s.phrases[i])
			s.transitionToAwake(s.phrases[i])
			return true
		}
	}
	return false
}

// transitionToAwake opens the awake window.
func (s *WakePhraseStart) transitionToAwake(phrase string) {
	s.awake = true
	s.accum = ""
	s.refresh()
	if s.detected != nil {
		s.detected(phrase)
	}
}

// transitionToIdle closes it again.
func (s *WakePhraseStart) transitionToIdle() {
	slog.Debug("turns: wake phrase timeout, going back to sleep")
	s.awake = false
	s.accum = ""
	if s.timedOut != nil {
		s.timedOut()
	}
}

// refresh restarts the inactivity timer.
func (s *WakePhraseStart) refresh() {
	if s.cancelTimer != nil {
		s.cancelTimer()
	}
	s.cancelTimer = s.after(s.timeout, func() {
		s.cancelTimer = nil
		if s.awake {
			s.transitionToIdle()
		}
	})
}

// TurnStarted readies the strategy for a new turn.
//
// In timeout mode it keeps the state and refreshes the timeout: a turn starting
// is the activity that keeps the strategy awake. In single-activation mode it
// does nothing, because the keepalive window opened when the phrase was detected
// is what puts the strategy back to sleep, and cutting it short here would block
// the very turn the phrase opened.
func (s *WakePhraseStart) TurnStarted() {
	if s.awake && !s.single {
		s.refresh()
	}
}

// Cleanup stops the inactivity timer.
func (s *WakePhraseStart) Cleanup() {
	if s.cancelTimer != nil {
		s.cancelTimer()
		s.cancelTimer = nil
	}
}

// ExternalStartConfig configures an ExternalStart strategy.
type ExternalStartConfig struct {
	// EnableInterruptions broadcasts an interruption when a proposal opens a
	// turn; nil defaults to true. It is ignored on the adopt path, where the
	// emitter has already broadcast one.
	EnableInterruptions *bool
}

// ExternalStart takes its cue for the start of a turn from another processor
// rather than detecting it. It understands two signals, which differ in how much
// the emitter has already done:
//
//   - A ProposedUserStartedSpeakingFrame is a service with its own turn
//     detection proposing a turn boundary. This strategy makes the decision,
//     emitting the UserStartedSpeakingFrame and broadcasting the interruption
//     itself. Embed it to adjust when, or whether, a proposal opens a turn.
//   - A UserStartedSpeakingFrame means the turn was already decided and
//     announced elsewhere, typically by a shared turn processor fanning turns
//     out to several aggregators. This strategy adopts that decision and emits
//     nothing, so the turn is not announced twice.
//
// A service that emits turn frames directly lands on the adopt path and keeps
// working, but it owns the interruption logic itself. Emitting proposals instead
// hands that job back to the pipeline.
type ExternalStart struct {
	StartStrategyBase
}

// NewExternalStart builds an external start strategy.
func NewExternalStart(cfg ExternalStartConfig) *ExternalStart {
	s := &ExternalStart{}
	s.EnableInterruptions = boolOr(cfg.EnableInterruptions, true)
	s.EnableUserSpeakingFrames = true
	return s
}

// ResolvesProposedTurnStartFrames reports that this strategy resolves proposals
// into turn starts.
func (s *ExternalStart) ResolvesProposedTurnStartFrames() bool { return true }

// Process resolves a proposal, or adopts a turn start decided elsewhere.
func (s *ExternalStart) Process(f frames.Frame) ProcessFrameResult {
	switch f.(type) {
	case *frames.ProposedUserStartedSpeakingFrame:
		slog.Debug("turns: resolving a proposed user turn start")
		s.TriggerStarted()
		return Stop
	case *frames.UserStartedSpeakingFrame:
		// Already announced elsewhere: adopt the decision without repeating it.
		slog.Debug("turns: adopting a user turn start decided elsewhere")
		off := false
		s.TriggerStartedOverriding(StartedOverrides{
			EnableInterruptions:      &off,
			EnableUserSpeakingFrames: &off,
		})
		return Stop
	}
	return Continue
}
