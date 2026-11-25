package turns

import (
	"context"
	"log/slog"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// The wait-reassurance instruction and prompt are what the model reads: the
// instruction once, in its system instruction, and the prompt each time the
// conversation has gone quiet with background work in progress. A small change
// in wording can change what a model says, or whether it stays silent, in ways
// no unit test catches, so change either only with the model's replies checked
// before and after.

// WaitReassuranceSilentMark is what the model answers with to say nothing.
const WaitReassuranceSilentMark = "∅"

// WaitReassuranceInstruction is the standing guidance on the waiting message,
// composed into the LLM's system instruction: what the message is, how to
// answer it with silence, and what to say otherwise. {mark} is the silent mark.
const WaitReassuranceInstruction = "While background work you started on the user's behalf is in flight and the " +
	"conversation has gone quiet, a developer message beginning \"The user is waiting on background work you " +
	"started on their behalf\" will say so. That message asks one thing: to reassure the user, if needed, that the " +
	"work is still underway. Answer that message with exactly {mark} to respond with silence (after any marker " +
	"your instructions require). Otherwise, say one short sentence, in your own voice and with wording that " +
	"varies from previous reassurances about the same work, that makes clear the work has been going on since " +
	"the user asked and is continuing: a \"still\", not a start. Keep it general, naming at most the kind of step " +
	"under way, unless the user asked for detail: then name the step, drawing on the notes the work has " +
	"produced. Never report a result or a finding, never say the work is done, and ask no questions and make no " +
	"offers or promises, not even to keep the user posted. {mark} is only ever the answer to that message. " +
	"Whatever the user says, answer them, and when the work reports back, tell the user, however often you have " +
	"chosen silence before."

// WaitReassurancePrompt is the developer message appended to the conversation
// each time the user has waited quietly on background work: the facts of the
// moment, the ask, and what to weigh in answering it. {quiet_secs} is how long
// nothing has been said. Its opening is what the instruction names it by.
const WaitReassurancePrompt = "The user is waiting on background work you started on their behalf, and nothing " +
	"has been said for about {quiet_secs} seconds. Reassure them, if needed, that the work is still underway. " +
	"When deciding what to say (or in some cases, whether to say anything at all), consider the user's " +
	"preferences (for example, they may have asked for only the final result of the work and no progress " +
	"updates along the way) as well as what would now sound natural and reassuring in the conversation."

// DefaultWaitReassuranceInterval is how long the conversation stays quiet, with
// background work in progress, before the model is asked whether to say
// something, when WaitReassuranceConfig.Interval is zero.
const DefaultWaitReassuranceInterval = 10 * time.Second

// WaitReassuranceConfig configures a WaitReassuranceLLMExtension. The zero value
// is the defaults.
type WaitReassuranceConfig struct {
	// Interval is how long the conversation stays quiet, with background work
	// in progress, before the model is asked whether to say something; zero
	// uses DefaultWaitReassuranceInterval.
	Interval time.Duration
	// Instruction is the standing guidance added to the LLM's system
	// instruction, a template with {mark} (so any other brace in it is
	// doubled); empty uses WaitReassuranceInstruction.
	Instruction string
	// Prompt is the developer message appended each time, a template with
	// {quiet_secs} and, if it wants it, {mark}; empty uses
	// WaitReassurancePrompt.
	Prompt string
	// SilentMark is what the model answers with to say nothing; empty uses
	// WaitReassuranceSilentMark.
	SilentMark string
}

// WaitReassuranceLLMExtension has the LLM say a brief word, in its own words,
// while the user waits on background work.
//
// Background work is what a BackgroundWorkStartedFrame announced: an
// asynchronous function call, or a task delegated to an LLMWithBackend backend.
// Once nothing has been said for the interval with such work in progress, the
// user aggregator broadcasts a UserWaitingFrame, on which the extension appends
// the prompt to the conversation as a developer message and runs the LLM. The
// model decides what to say, with the user's wishes and what it knows of the
// work in view, or that nothing should be said, by answering with the silent
// mark.
//
// The extension's convention with the model has two halves. The instruction
// goes into the service's system instruction and explains the prompt and the
// mark once. A reply opening with the mark is dropped unspoken, reaching
// neither the TTS nor the context, and the decision is broadcast as an
// LLMMarkerResponseFrame of kind frames.LLMMarkerKindSilent, on which the user
// aggregator starts the interval over. A spoken remark starts it over through
// the bot's speech, so a long wait gets a fresh decision every interval.
//
// At start, the extension sets the user aggregator's waiting timeout to the
// interval with a UserWaitingTimeoutUpdateFrame, so the interval is configured
// here alone. The extension belongs on a text LLM: a speech-to-speech service's
// audio carries no mark to read. An application that wants to answer the wait
// in its own way (a canned phrase, music) uses the user aggregator's waiting
// timeout and its on_user_waiting event instead.
//
//	if err := model.AddExtension(turns.NewWaitReassuranceLLMExtension(turns.WaitReassuranceConfig{})); err != nil {
//		return err
//	}
type WaitReassuranceLLMExtension struct {
	llm.BaseExtension

	interval    time.Duration
	instruction string
	prompt      string
	silentMark  string

	mu sync.Mutex
	// The reply being read for the silent mark: its text so far, held until
	// the first character decides it, and whether the rest is dropped.
	replyOpening string
	replyDecided bool
	replySilent  bool
}

// NewWaitReassuranceLLMExtension builds the extension with cfg.
func NewWaitReassuranceLLMExtension(cfg WaitReassuranceConfig) *WaitReassuranceLLMExtension {
	e := &WaitReassuranceLLMExtension{
		interval:    cfg.Interval,
		instruction: cfg.Instruction,
		prompt:      cfg.Prompt,
		silentMark:  cfg.SilentMark,
	}
	if e.interval == 0 {
		e.interval = DefaultWaitReassuranceInterval
	}
	if e.instruction == "" {
		e.instruction = WaitReassuranceInstruction
	}
	if e.prompt == "" {
		e.prompt = WaitReassurancePrompt
	}
	if e.silentMark == "" {
		e.silentMark = WaitReassuranceSilentMark
	}
	return e
}

// SystemInstruction implements llm.Extension: the standing guidance, with
// the silent mark in place.
func (e *WaitReassuranceLLMExtension) SystemInstruction() string {
	return formatTemplate(e.instruction, map[string]string{"mark": e.silentMark})
}

// ProcessFrame implements llm.Extension: it configures the wait at start,
// and asks the model once the user has waited.
func (e *WaitReassuranceLLMExtension) ProcessFrame(ctx context.Context, f frames.Frame, _ processor.Direction) error {
	switch fr := f.(type) {
	case *frames.StartFrame:
		return e.PushFrame(ctx, frames.NewUserWaitingTimeoutUpdateFrame(e.interval), processor.Upstream)
	case *frames.UserWaitingFrame:
		quietSecs := strconv.FormatFloat(math.RoundToEven(fr.QuietSecs), 'f', 0, 64)
		slog.Debug("asking the model whether to reassure the user", "quiet_secs", quietSecs)
		ask := frames.NewLLMMessagesAppendFrame([]frames.Message{{
			Role: frames.RoleDeveloper,
			Text: formatTemplate(e.prompt, map[string]string{"quiet_secs": quietSecs, "mark": e.silentMark}),
		}})
		ask.RunLLM = true
		return e.PushFrame(ctx, ask, processor.Downstream)
	case *frames.InterruptionFrame:
		e.resetReading()
	}
	return nil
}

// FilterPushFrame implements llm.Extension: it drops a reply that opens with
// the silent mark, and reports it. It returns the reply's text once its opening
// shows it is not the mark, and nothing while the opening is held or the reply
// is being dropped.
func (e *WaitReassuranceLLMExtension) FilterPushFrame(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) ([]frames.Frame, error) {
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		e.resetReading()
	case *frames.LLMTextFrame:
		if dir == processor.Downstream {
			return e.readOpening(ctx, fr)
		}
	}
	return []frames.Frame{f}, nil
}

// readOpening holds a reply's opening until its first character shows whether
// the reply is the mark.
//
// A silent reply is reported as soon as the mark is read, and the rest of it is
// dropped unread. The turn-completion extension sits "closer" to the service,
// so a reply of a turn marker followed by the mark is read as silent too.
func (e *WaitReassuranceLLMExtension) readOpening(ctx context.Context, f *frames.LLMTextFrame) ([]frames.Frame, error) {
	e.mu.Lock()
	if e.replySilent {
		e.mu.Unlock()
		return nil, nil
	}
	if e.replyDecided {
		e.mu.Unlock()
		return []frames.Frame{f}, nil
	}
	held := e.replyOpening
	e.replyOpening += f.Text
	opening := e.replyOpening
	text := strings.TrimLeftFunc(opening, unicode.IsSpace)
	switch {
	case text == "":
		e.mu.Unlock()
		return nil, nil
	case strings.HasPrefix(text, e.silentMark):
		e.replyDecided = true
		e.replySilent = true
		e.mu.Unlock()
		slog.Debug("the model chose to say nothing")
		mark := e.silentMark
		err := e.BroadcastFrame(ctx, func() frames.Frame {
			return frames.NewLLMMarkerResponseFrame(opening, mark, frames.LLMMarkerKindSilent, []string{mark})
		})
		return nil, err
	case strings.HasPrefix(e.silentMark, text):
		// The mark so far, with more of it possibly to come.
		e.mu.Unlock()
		return nil, nil
	}
	e.replyDecided = true
	e.mu.Unlock()
	if held == "" {
		return []frames.Frame{f}, nil
	}
	return []frames.Frame{frames.NewLLMTextFrame(opening)}, nil
}

func (e *WaitReassuranceLLMExtension) resetReading() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.replyOpening = ""
	e.replyDecided = false
	e.replySilent = false
}

// formatTemplate fills a template's {name} fields from values, and turns a
// doubled brace into a single one.
func formatTemplate(template string, values map[string]string) string {
	pairs := []string{"{{", "{", "}}", "}"}
	for name, value := range values {
		pairs = append(pairs, "{"+name+"}", value)
	}
	return strings.NewReplacer(pairs...).Replace(template)
}
