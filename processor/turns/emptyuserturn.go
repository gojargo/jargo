package turns

// DefaultEmptyUserTurnInterruptedPrompt is the developer message an empty user
// turn that interrupted the bot is answered with, unless another is configured.
const DefaultEmptyUserTurnInterruptedPrompt = "The user may have said something while you were responding, " +
	"but it was not " +
	"recognized. Your response was cut off there, and the conversation only includes " +
	"the part they heard, which may be none of it. Briefly ask them to repeat what they said, and " +
	"repeat any question of yours they may have missed."

// defaultMaxConsecutiveEmptyUserTurnRecoveries is how many empty turns in a row
// are answered unless another bound is configured.
const defaultMaxConsecutiveEmptyUserTurnRecoveries = 1

// EmptyUserTurnConfig is how the user aggregator responds to a user turn with
// no transcript.
//
// A user turn can start on voice activity alone and then end with no
// transcript: a cough, background noise, or speech the STT could not
// recognize. Nothing is written to the context and the LLM does not run. With
// this config, the aggregator can append a developer message for such a turn
// and run the LLM once.
//
// Either case may be noise. They differ in what happens if the turn goes
// unanswered, so each has its own prompt:
//
//   - Interrupted: the turn started while the bot was thinking, speaking or
//     running a function call, and cut it off. Unanswered, the bot stays silent
//     mid-response, so by default the LLM runs and the bot asks the user to
//     repeat or picks up where it left off. A turn before the bot has first
//     finished speaking counts as interrupted too.
//   - Idle: the bot had finished and was waiting for the user. The conversation
//     is not stuck, and answering what may be noise would be intrusive, so by
//     default the turn gets no answer.
type EmptyUserTurnConfig struct {
	// InterruptedPrompt is the developer message for an empty turn that
	// interrupted the bot. Nil uses DefaultEmptyUserTurnInterruptedPrompt; an
	// empty string leaves such turns unanswered, and with IdlePrompt left empty
	// too, every empty turn.
	InterruptedPrompt *string
	// IdlePrompt is the developer message for an empty turn while the bot was
	// idle. Empty, the default, leaves such turns unanswered.
	IdlePrompt string
	// MaxConsecutiveRecoveries is how many empty turns in a row are answered;
	// zero uses 1. Further ones are left unanswered until the user says
	// something that is transcribed.
	MaxConsecutiveRecoveries int `validate:"min=0"`
}

// interruptedPrompt is the prompt for an empty turn that interrupted the bot,
// or "" when such turns go unanswered.
func (c EmptyUserTurnConfig) interruptedPrompt() string {
	if c.InterruptedPrompt == nil {
		return DefaultEmptyUserTurnInterruptedPrompt
	}
	return *c.InterruptedPrompt
}

// Prompt is the developer message an empty turn is answered with, "" when it
// goes unanswered. interrupted says whether the turn interrupted the bot.
func (c EmptyUserTurnConfig) Prompt(interrupted bool) string {
	if interrupted {
		return c.interruptedPrompt()
	}
	return c.IdlePrompt
}

// MaxRecoveries is how many empty turns in a row are answered.
func (c EmptyUserTurnConfig) MaxRecoveries() int {
	if c.MaxConsecutiveRecoveries <= 0 {
		return defaultMaxConsecutiveEmptyUserTurnRecoveries
	}
	return c.MaxConsecutiveRecoveries
}
