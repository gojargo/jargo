package turns

import (
	"log/slog"

	"github.com/gojargo/jargo/frames"
)

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
//
// The mute is also released when the bot's first speaking turn fails before
// producing any audio, so that a failed opening leaves the user able to speak.
type MuteUntilFirstBotComplete struct {
	botStarted   bool
	firstHandled bool
}

// NewMuteUntilFirstBotComplete builds a mute-until-first-bot-complete strategy.
func NewMuteUntilFirstBotComplete() *MuteUntilFirstBotComplete {
	return &MuteUntilFirstBotComplete{}
}

// ShouldMute reports muted until the bot's first speech completes, or until that
// first turn fails before it starts.
func (s *MuteUntilFirstBotComplete) ShouldMute(f frames.Frame) bool {
	switch fr := f.(type) {
	case *frames.BotStartedSpeakingFrame:
		s.botStarted = true
	case *frames.BotStoppedSpeakingFrame:
		s.firstHandled = true
	case *frames.ErrorFrame:
		s.handleError(fr)
	}
	return !s.firstHandled
}

// handleError releases the mute when the first speaking turn fails before any
// audio. No audio means no stopped frame, and no later turn to wait for either,
// since a muted user can never prompt one. Only errors before the bot starts
// speaking count: after that the transport ends the turn on its own once the
// audio dries up.
func (s *MuteUntilFirstBotComplete) handleError(f *frames.ErrorFrame) {
	if s.botStarted || s.firstHandled {
		return
	}
	source := "the pipeline"
	if f.Source != nil {
		source = f.Source.Name()
	}
	slog.Warn("releasing the user mute without the bot having completed its first speech",
		"after_an_error_from", source, "err", f.Error)
	s.firstHandled = true
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
