package turns

import (
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
	Timeout time.Duration
	// SingleActivation requires the phrase again for every turn.
	SingleActivation bool
}

// WakePhraseStart gates a turn behind a spoken wake phrase. Place it first in
// the start chain: while asleep it blocks the other start strategies; once awake
// it lets them run until an inactivity timeout puts it back to sleep.
type WakePhraseStart struct {
	StartStrategyBase
	patterns []*regexp.Regexp
	timeout  time.Duration
	single   bool

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
	s := &WakePhraseStart{patterns: compileWakePatterns(cfg.Phrases), timeout: timeout, single: cfg.SingleActivation}
	s.EnableInterruptions = true
	s.EnableUserSpeakingFrames = true
	return s
}

// compileWakePatterns builds a case-insensitive, whitespace-flexible regex per
// phrase.
func compileWakePatterns(phrases []string) []*regexp.Regexp {
	var out []*regexp.Regexp
	for _, p := range phrases {
		words := strings.Fields(p)
		if len(words) == 0 {
			continue
		}
		for i, w := range words {
			words[i] = regexp.QuoteMeta(w)
		}
		out = append(out, regexp.MustCompile(`(?i)\b`+strings.Join(words, `\s+`)+`\b`))
	}
	return out
}

// Process matches the wake phrase while asleep and keeps the session alive while
// awake.
func (s *WakePhraseStart) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		return s.onTranscript(fr.Text)
	case *frames.UserSpeakingFrame, *frames.VADUserStartedSpeakingFrame, *frames.BotSpeakingFrame:
		if s.awake {
			s.refresh()
		}
	}
	if s.awake {
		return Continue
	}
	return Stop // asleep: block the other start strategies
}

func (s *WakePhraseStart) onTranscript(text string) ProcessFrameResult {
	if s.awake {
		s.refresh()
		return Continue
	}
	s.accum += " " + text
	if len(s.accum) > wakeAccumLimit {
		s.accum = s.accum[len(s.accum)-wakeAccumLimit:]
	}
	if s.matches(s.accum) {
		s.accum = ""
		s.awake = true
		s.TriggerStarted()
		s.refresh()
		return Stop
	}
	s.TriggerResetAggregation() // drop pre-wake speech
	return Stop
}

func (s *WakePhraseStart) matches(text string) bool {
	for _, p := range s.patterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}

// refresh restarts the inactivity timer.
func (s *WakePhraseStart) refresh() {
	if s.cancelTimer != nil {
		s.cancelTimer()
	}
	s.cancelTimer = s.after(s.timeout, func() {
		s.awake = false
		s.cancelTimer = nil
	})
}

// Reset re-requires the wake phrase for the next turn when single-activation.
func (s *WakePhraseStart) Reset() {
	if s.single {
		s.awake = false
		if s.cancelTimer != nil {
			s.cancelTimer()
			s.cancelTimer = nil
		}
	}
}

// Cleanup stops the inactivity timer.
func (s *WakePhraseStart) Cleanup() {
	if s.cancelTimer != nil {
		s.cancelTimer()
		s.cancelTimer = nil
	}
}

// ExternalStart relays a turn-start decided by another processor (it triggers on
// an inbound UserStartedSpeakingFrame) without re-emitting speaking frames or
// interruptions.
type ExternalStart struct {
	StartStrategyBase
}

// NewExternalStart builds an external start strategy.
func NewExternalStart() *ExternalStart {
	s := &ExternalStart{}
	s.EnableInterruptions = false
	s.EnableUserSpeakingFrames = false
	return s
}

// Process triggers on an inbound user-started frame.
func (s *ExternalStart) Process(f frames.Frame) ProcessFrameResult {
	if _, ok := f.(*frames.UserStartedSpeakingFrame); ok {
		s.TriggerStarted()
		return Stop
	}
	return Continue
}
