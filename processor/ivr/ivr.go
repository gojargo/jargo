// Package ivr navigates automated phone menus (IVR systems).
//
// A Navigator is a pipeline of an LLM and a Processor. It starts by asking the
// LLM whether the far end is an automated system or a person: on an automated
// system it switches the LLM to a navigation prompt built around a goal, and
// the voice-activity detector to timing that does not cut a menu into pieces;
// on a person it raises EventConversationDetected, with the conversation so
// far, for the application to take over. While navigating, the LLM answers with
// control tags: <dtmf>N</dtmf> presses a key, and <ivr>status</ivr> reports
// progress, raised as EventIVRStatusChanged. The tags are stripped from what is
// spoken and recorded in the conversation, so the LLM knows what it already
// pressed.
package ivr

import (
	"context"
	"log/slog"
	"strings"
	"sync"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/text"
)

// Status is the IVR navigation status the LLM reports through an <ivr> tag.
type Status string

const (
	// StatusDetected means an IVR system was detected and navigation begins.
	StatusDetected Status = "detected"
	// StatusCompleted means navigation reached its goal.
	StatusCompleted Status = "completed"
	// StatusStuck means the menu could not be navigated.
	StatusStuck Status = "stuck"
	// StatusWait means the LLM needs more of the menu before it can act.
	StatusWait Status = "wait"
)

// valid reports whether s is one of the statuses.
func (s Status) valid() bool {
	switch s {
	case StatusDetected, StatusCompleted, StatusStuck, StatusWait:
		return true
	default:
		return false
	}
}

// The events a Processor raises.
const (
	// EventConversationDetected fires when the classifier finds a person on the
	// line, with the conversation so far ([]frames.Message, its system message
	// left out).
	EventConversationDetected = "on_conversation_detected"
	// EventIVRStatusChanged fires with the Status when IVR navigation is
	// detected, completed or stuck.
	EventIVRStatusChanged = "on_ivr_status_changed"
)

// The pattern types the control tags are aggregated under.
const (
	tagDTMF frames.TextType = "dtmf"
	tagMode frames.TextType = "mode"
	tagIVR  frames.TextType = "ivr"
)

// DefaultIVRStopSecs is how long the caller must be quiet before a turn ends
// while navigating an IVR, when Config.IVRVADParams is nil: long enough that a
// pause between menu options does not end the turn.
const DefaultIVRStopSecs = 2.0

// Config configures a Processor.
type Config struct {
	// ClassifierPrompt is the system prompt that asks the LLM whether the far
	// end is an IVR system or a person. Required.
	ClassifierPrompt string `validate:"required"`
	// IVRPrompt is the system prompt for navigating an IVR system. Required.
	IVRPrompt string `validate:"required"`
	// IVRVADParams are the voice-activity parameters while navigating. Nil uses
	// a stop of DefaultIVRStopSecs and the other defaults.
	IVRVADParams *vad.Params
}

// Processor acts on the control tags in the LLM's streamed response: it presses
// the keys a <dtmf> tag names, switches modes on a <mode> tag, and reports an
// <ivr> status, passing the rest of the text on as aggregated sentences.
type Processor struct {
	*processor.Base
	cfg       Config
	vadParams vad.Params

	mu sync.Mutex
	// saved is the conversation the navigator last saw, carried over when the
	// mode switches.
	saved []frames.Message
	agg   *text.PatternPairAggregator
	// ctx is the context of the frame being processed, which the handlers the
	// aggregator calls push with.
	ctx context.Context //nolint:containedctx // set only for the duration of one frame
}

// New builds an IVR Processor.
func New(cfg Config) *Processor {
	p := &Processor{cfg: cfg}
	p.vadParams = vad.DefaultParams()
	p.vadParams.StopSecs = DefaultIVRStopSecs
	if cfg.IVRVADParams != nil {
		p.vadParams = *cfg.IVRVADParams
	}
	p.Base = processor.New("IVRProcessor", p)
	p.agg = text.NewPatternPairAggregator(frames.TextTypeSentence, "")
	for _, tag := range []frames.TextType{tagDTMF, tagMode, tagIVR} {
		_ = p.agg.AddPattern(tag, "<"+string(tag)+">", "</"+string(tag)+">", text.MatchRemove)
	}
	p.agg.OnPatternMatch(tagDTMF, func(m text.Aggregation) { p.handleDTMF(m.Text) })
	p.agg.OnPatternMatch(tagMode, func(m text.Aggregation) { p.handleMode(m.Text) })
	p.agg.OnPatternMatch(tagIVR, func(m text.Aggregation) { p.handleIVR(m.Text) })
	p.Events().Register(EventConversationDetected, false)
	p.Events().Register(EventIVRStatusChanged, false)
	return p
}

// UpdateSavedMessages sets the conversation carried over when the mode
// switches.
func (p *Processor) UpdateSavedMessages(messages []frames.Message) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.saved = messages
}

// conversationHistory is the saved conversation without its first, system,
// message.
func (p *Processor) conversationHistory() []frames.Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.saved) == 0 {
		return []frames.Message{}
	}
	return append([]frames.Message(nil), p.saved[1:]...)
}

// ProcessFrame aggregates the LLM's text, acting on the control tags in it.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		if err := p.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		// The classifier prompt goes first, so the LLM starts by deciding who
		// answered.
		update := frames.NewLLMMessagesUpdateFrame([]frames.Message{
			{Role: frames.RoleDeveloper, Text: p.cfg.ClassifierPrompt},
		})
		return p.PushFrame(ctx, update, processor.Upstream)
	case *frames.InterruptionFrame:
		// The text of the interrupted response is dropped, a tag left open too.
		p.agg.HandleInterruption()
		return p.PushFrame(ctx, f, dir)
	case *frames.LLMTextFrame:
		p.ctx = ctx
		results := p.agg.Aggregate(fr.Text)
		p.ctx = nil
		for _, r := range results {
			if err := p.PushFrame(ctx, frames.NewAggregatedTextFrame(r.Text, r.Type), dir); err != nil {
				return err
			}
		}
		return nil
	case *frames.LLMFullResponseEndFrame, *frames.EndFrame:
		if rest, ok := p.agg.Flush(); ok {
			if err := p.PushFrame(ctx, frames.NewAggregatedTextFrame(rest.Text, rest.Type), dir); err != nil {
				return err
			}
		}
		return p.PushFrame(ctx, f, dir)
	default:
		return p.PushFrame(ctx, f, dir)
	}
}

// handleDTMF presses the key a <dtmf> tag names, at once, and records the tag
// in the conversation. A value that is not one keypad key is refused.
func (p *Processor) handleDTMF(value string) {
	ctx := p.ctx
	key := frames.KeypadEntry(value)
	if !key.Valid() {
		slog.Warn("ivr: invalid DTMF value, must be 0-9, * or #", "value", value)
		return
	}
	_ = p.PushFrame(ctx, frames.NewOutputDTMFUrgentFrame(key), processor.Downstream)
	_ = p.PushFrame(ctx, recorded("<dtmf>"+value+"</dtmf>"), processor.Downstream)
}

// handleIVR acts on an <ivr> status and records it in the conversation. An
// unknown status is refused.
func (p *Processor) handleIVR(value string) {
	ctx := p.ctx
	status := Status(value)
	if !status.valid() {
		slog.Warn("ivr: unknown IVR status", "status", value)
		return
	}
	switch status {
	case StatusDetected:
		p.ivrDetected(ctx)
	case StatusCompleted, StatusStuck:
		p.Events().Call(ctx, EventIVRStatusChanged, p, status)
	case StatusWait:
		// The LLM needs more of the menu, which keeps coming.
		slog.Debug("ivr: waiting for a more complete transcription")
	}
	_ = p.PushFrame(ctx, recorded("<ivr>"+value+"</ivr>"), processor.Downstream)
}

// handleMode switches on the classifier's verdict. Nothing is recorded: the
// classification exchange is dropped with its prompt once the mode is known.
func (p *Processor) handleMode(value string) {
	ctx := p.ctx
	switch value {
	case "conversation":
		slog.Debug("ivr: conversation detected")
		p.Events().Call(ctx, EventConversationDetected, p, p.conversationHistory())
	case "ivr":
		p.ivrDetected(ctx)
	}
}

// ivrDetected switches to navigating: the navigation prompt with the
// conversation so far, run at once, and the voice-activity timing for menus.
func (p *Processor) ivrDetected(ctx context.Context) {
	slog.Debug("ivr: IVR detected, switching to navigation")
	messages := append([]frames.Message{{Role: frames.RoleDeveloper, Text: p.cfg.IVRPrompt}},
		p.conversationHistory()...)
	update := frames.NewLLMMessagesUpdateFrame(messages)
	update.RunLLM = true
	_ = p.PushFrame(ctx, update, processor.Upstream)
	_ = p.PushFrame(ctx, frames.NewVADParamsUpdateFrame(p.vadParams), processor.Upstream)
	p.Events().Call(ctx, EventIVRStatusChanged, p, StatusDetected)
}

// recorded is a text frame the conversation records and the synthesizer skips.
func recorded(s string) *frames.TextFrame {
	f := frames.NewTextFrame(s)
	f.SkipTTS = new(true)
	return f
}

// NavigatorConfig configures a Navigator.
type NavigatorConfig struct {
	// LLM is the service that classifies the far end and navigates. Required.
	LLM processor.Processor `validate:"required"`
	// Goal is what the navigation is for ("reach the billing department"),
	// built into the navigation prompt. Required.
	Goal string `validate:"required"`
	// IVRVADParams are the voice-activity parameters while navigating. Nil uses
	// a stop of DefaultIVRStopSecs.
	IVRVADParams *vad.Params
}

// Navigator is a pipeline of the LLM and a Processor that classifies the far
// end, navigates an IVR system towards a goal, and hands a person over to the
// application. Place it where the LLM would go.
type Navigator struct {
	*pipeline.Pipeline
	proc *Processor
}

// NewNavigator builds a Navigator.
func NewNavigator(cfg NavigatorConfig) *Navigator {
	proc := New(Config{
		ClassifierPrompt: ClassifierPrompt,
		IVRPrompt:        strings.ReplaceAll(IVRNavigationBase, "{goal}", cfg.Goal),
		IVRVADParams:     cfg.IVRVADParams,
	})
	n := &Navigator{proc: proc}
	// The conversation reaching the LLM is what is carried over when the mode
	// switches, so it is saved on its way in.
	n.Pipeline = pipeline.New(newContextSaver(proc), cfg.LLM, proc)
	return n
}

// Processor returns the navigator's IVR processor, whose events
// (EventConversationDetected, EventIVRStatusChanged) report what the
// navigation finds.
func (n *Navigator) Processor() *Processor { return n.proc }

// contextSaver hands the processor the messages of each conversation on its
// way to the LLM.
type contextSaver struct {
	*processor.Base
	proc *Processor
}

func newContextSaver(proc *Processor) *contextSaver {
	s := &contextSaver{proc: proc}
	s.Base = processor.New("IVRContextSaver", s)
	return s
}

func (s *contextSaver) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if fr, ok := f.(*frames.LLMContextFrame); ok {
		s.proc.UpdateSavedMessages(fr.Context.Messages())
	}
	return s.PushFrame(ctx, f, dir)
}
