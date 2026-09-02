package pipeline

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// LLMMember is what an LLM switcher needs of each language model service it
// manages. The LLM service base satisfies it, so any service built on that base
// can be a member.
type LLMMember interface {
	processor.Processor
	// SyncToolHandlers brings the service's tool registry into line with the
	// toolset a conversation advertises.
	SyncToolHandlers(ctx context.Context, tools []frames.Tool)
	// RegisterFunction records a handler for a tool the model may call.
	RegisterFunction(name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption)
}

// LLMSwitcher routes the conversation through one of several language model
// services at a time, so a call can move to a stronger model when it gets hard,
// or to another provider when one stops answering.
//
// It is a ServiceSwitcher that also keeps the members in step on the things a
// conversation carries rather than the pipeline: the tools it advertises and the
// handlers registered for them. An inactive model sits behind a closed filter
// and would otherwise never see them, and would then be out of step with what
// the model is told it can call the moment it becomes active.
type LLMSwitcher struct {
	*ServiceSwitcher
	llms []LLMMember
}

// LLMService marks the switcher as a language-model service.
//
// It is what the rest of the pipeline talks to in place of one, and frames from
// the active service leave through the switcher's own PushFrame, so without this
// nothing arriving from an LLM looks like it came from an LLM. The observer that
// logs what a model was given and what it produced was silent for every pipeline
// built on a switcher, which is every pipeline with a failover chain.
func (s *LLMSwitcher) LLMService() {}

// NewLLMSwitcher builds a switcher over llms, the first of which starts active,
// choosing between them with the strategy newStrategy builds. A nil newStrategy
// switches only when asked.
func NewLLMSwitcher(llms []LLMMember, newStrategy StrategyFunc) (*LLMSwitcher, error) {
	s := &LLMSwitcher{llms: llms}

	services := make([]processor.Processor, 0, len(llms))
	for _, l := range llms {
		services = append(services, l)
	}

	sw, err := newServiceSwitcherAs(s, "LLMSwitcher", services, newStrategy)
	if err != nil {
		return nil, err
	}
	s.ServiceSwitcher = sw
	return s, nil
}

// LLMs are the language model services the switcher manages.
func (s *LLMSwitcher) LLMs() []LLMMember { return s.llms }

// ActiveLLM is the language model service currently in use.
func (s *LLMSwitcher) ActiveLLM() LLMMember {
	active := s.ActiveService()
	for _, l := range s.llms {
		if l == active {
			return l
		}
	}
	return nil
}

// RunInference answers a conversation once on the active service, off to the
// side of the pipeline. It reports false if the active service cannot answer
// off-pipeline.
func (s *LLMSwitcher) RunInference(
	ctx context.Context, convo *frames.LLMContext, opts llm.InferenceOptions,
) (string, bool, error) {
	inf, ok := s.ActiveLLM().(llm.Inferencer)
	if !ok {
		return "", false, nil
	}
	out, err := inf.RunInference(ctx, convo, opts)
	return out, true, err
}

// RegisterFunction records a handler on every member, active or not, so a tool
// keeps working across a switch.
func (s *LLMSwitcher) RegisterFunction(
	name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption,
) {
	for _, l := range s.llms {
		l.RegisterFunction(name, h, opts...)
	}
}

// ProcessFrame keeps every member's tool handlers in step with the conversation.
//
// Only the active service receives the conversation, so it alone would pick up
// the handlers the advertised tools carry. The others are synced here, from
// outside the branches, so whichever becomes active next already knows how to
// answer what the model may call.
func (s *LLMSwitcher) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.ServiceSwitcher.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if cf, ok := f.(*frames.LLMContextFrame); ok && cf.Context != nil {
		for _, l := range s.llms {
			l.SyncToolHandlers(ctx, cf.Context.Tools())
		}
	}
	return nil
}
