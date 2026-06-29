package turns

import "github.com/gojargo/jargo/frames"

// MuteStrategy decides whether user input should be suppressed right now.
// ShouldMute is called for every frame (so the strategy can track state) and
// returns the muted state as of that frame. The UserTurnProcessor OR-reduces all
// strategies and, while muted, drops the user-input frames before they reach
// turn detection — so the user can neither barge in nor pollute the context at
// the wrong moment. Strategies are driven only from the processor (under its
// mute mutex) and need no locking of their own.
type MuteStrategy interface {
	ShouldMute(f frames.Frame) bool
}

// AlwaysUserMute mutes the user whenever the bot is speaking.
type AlwaysUserMute struct {
	botSpeaking bool
}

// NewAlwaysUserMute builds an always-while-bot-speaking mute strategy.
func NewAlwaysUserMute() *AlwaysUserMute { return &AlwaysUserMute{} }

// ShouldMute reports muted while the bot speaks.
func (s *AlwaysUserMute) ShouldMute(f frames.Frame) bool {
	switch f.(type) {
	case *frames.BotStartedSpeakingFrame:
		s.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		s.botSpeaking = false
	}
	return s.botSpeaking
}

// FirstSpeechUserMute mutes the user only during the bot's first speaking turn,
// allowing pre-speech input and never muting afterward.
type FirstSpeechUserMute struct {
	botSpeaking  bool
	firstHandled bool
}

// NewFirstSpeechUserMute builds a first-speech mute strategy.
func NewFirstSpeechUserMute() *FirstSpeechUserMute { return &FirstSpeechUserMute{} }

// ShouldMute reports muted only during the bot's first speech.
func (s *FirstSpeechUserMute) ShouldMute(f frames.Frame) bool {
	switch f.(type) {
	case *frames.BotStartedSpeakingFrame:
		s.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		s.botSpeaking = false
		s.firstHandled = true
	}
	return s.botSpeaking && !s.firstHandled
}

// MuteUntilFirstBotComplete mutes the user from the start of the session until
// the bot finishes its first speech.
type MuteUntilFirstBotComplete struct {
	firstHandled bool
}

// NewMuteUntilFirstBotComplete builds a mute-until-first-bot-complete strategy.
func NewMuteUntilFirstBotComplete() *MuteUntilFirstBotComplete {
	return &MuteUntilFirstBotComplete{}
}

// ShouldMute reports muted until the bot's first speech completes.
func (s *MuteUntilFirstBotComplete) ShouldMute(f frames.Frame) bool {
	if _, ok := f.(*frames.BotStoppedSpeakingFrame); ok {
		s.firstHandled = true
	}
	return !s.firstHandled
}

// FunctionCallUserMute mutes the user while any tool call is in flight.
type FunctionCallUserMute struct {
	active map[string]struct{}
}

// NewFunctionCallUserMute builds a function-call mute strategy.
func NewFunctionCallUserMute() *FunctionCallUserMute {
	return &FunctionCallUserMute{active: map[string]struct{}{}}
}

// ShouldMute reports muted while one or more tool calls are running.
func (s *FunctionCallUserMute) ShouldMute(f frames.Frame) bool {
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		for _, c := range fr.Calls {
			s.active[c.ID] = struct{}{}
		}
	case *frames.FunctionCallResultFrame:
		delete(s.active, fr.ToolCallID)
	case *frames.FunctionCallCancelFrame:
		delete(s.active, fr.ToolCallID)
	}
	return len(s.active) > 0
}
