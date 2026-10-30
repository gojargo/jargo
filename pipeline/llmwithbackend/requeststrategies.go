package llmwithbackend

import (
	"context"
	"encoding/json"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers/llmworker"
)

// Request strategies: how a delegate call becomes the backend's request.

// requestArgument is the explicit strategy's one argument, the request as the
// model words it.
const requestArgument = "request"

// explicitRequestParameter describes the explicit strategy's one argument.
const explicitRequestParameter = "What the user is asking for now, self-contained: their goal, the details " +
	"they gave (places, dates, names) and their latest correction, or what they " +
	"said about work already under way, such as to stop it or change it. Work " +
	"already handed over stays with the backend; a new request joins it, so do " +
	"not restate earlier requests."

// BackendRequestStrategy defines the delegate tool the frontend calls, and turns
// a call into the request the backend receives.
//
// The tool's parameters are the strategy's to declare, so the connector's
// handler has one signature and reads whatever was declared back from the call's
// arguments in ComposeRequest.
//
// Two strategies are provided, chosen by frontend kind when the app gives none.
// TranscriptBackendRequestStrategy hands over the conversation since the
// previous delegation: the default for a text frontend, whose context is current
// when the tool runs. ExplicitBackendRequestStrategy has the frontend model word
// the request itself: the default for a speech-to-speech frontend, which
// responds to audio before the transcript of that audio reaches its context, so
// the turns that prompted a handoff may not be in the context yet, and a
// transcript taken then would miss them.
type BackendRequestStrategy interface {
	// ToolParameters are the JSON-schema properties of the delegate tool, empty
	// when it takes none.
	ToolParameters() map[string]any
	// ToolRequired are which of the properties are required.
	ToolRequired() []string
	// FrontendInstruction is the strategy's guidance to the frontend model, if
	// any: how the request is worded. It goes where the pairing instruction's
	// PairingInstructionStrategyPlaceholder stands.
	FrontendInstruction() string
	// ToolDescription is how the delegate tool is described to the frontend
	// model.
	ToolDescription() string
	// RequiresContext reports whether the strategy reads the conversation from
	// the frontend's context, so it needs that context to be current when the
	// delegate tool runs. A speech-to-speech frontend's context lags the audio,
	// so the connector refuses such a strategy there.
	RequiresContext() bool
	// DelegateNeedsArguments reports whether a delegate call has to carry
	// arguments for the request to be built. Predictive delegation makes its
	// delegate call with none, so it works only with a strategy that reports
	// false.
	DelegateNeedsArguments() bool
	// ComposeRequest turns a delegate call into the text put to the backend. It
	// reports false when the call carries nothing new for the backend, in which
	// case nothing is sent.
	ComposeRequest(ctx context.Context, params llm.FunctionCallParams) (string, bool)
}

// StrategyTexts are the texts a request strategy puts to the models. A field
// left empty takes the strategy's default.
type StrategyTexts struct {
	// FrontendInstruction is the strategy's guidance to the frontend model,
	// placed in the pairing instruction.
	FrontendInstruction string
	// NoFrontendInstruction leaves the strategy without guidance of its own.
	NoFrontendInstruction bool
	// ToolDescription is how the delegate tool is described.
	ToolDescription string
	// RequestInstruction is what the backend should do with the request, placed
	// after it.
	RequestInstruction string
}

// frontendInstruction is the guidance the texts give, or def.
func (t StrategyTexts) frontendInstruction(def string) string {
	switch {
	case t.NoFrontendInstruction:
		return ""
	case t.FrontendInstruction != "":
		return t.FrontendInstruction
	default:
		return def
	}
}

// orDefault is s, or def when s is empty.
func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// TranscriptBackendRequestStrategy hands the backend the conversation and lets
// it work out the request.
//
// The frontend model calls the tool with no arguments; the request is the
// transcript of what was said since the previous delegation (the whole
// conversation the first time), rendered by llmworker.RenderTranscriptRequest.
// The backend's own messages in the conversation are left out: the backend
// already has what it said. The default for a text frontend, whose context is
// current when the tool runs.
//
// The cursor assumes the conversation accrues. A rewritten context or a failed
// delegation's turns are not re-sent; the user's next request carries what
// matters. A call that finds no new turn since the previous delegation sends
// nothing.
type TranscriptBackendRequestStrategy struct {
	frontendInstruction string
	toolDescription     string
	requestInstruction  string

	mu               sync.Mutex
	delegatedThrough int
}

// NewTranscriptBackendRequestStrategy builds a transcript strategy.
func NewTranscriptBackendRequestStrategy(texts StrategyTexts) *TranscriptBackendRequestStrategy {
	return &TranscriptBackendRequestStrategy{
		frontendInstruction: texts.frontendInstruction(TranscriptFrontendInstruction),
		toolDescription:     orDefault(texts.ToolDescription, TranscriptToolDescription),
		requestInstruction:  orDefault(texts.RequestInstruction, llmworker.TranscriptRequestInstruction),
	}
}

// ToolParameters implements BackendRequestStrategy: the tool takes none.
func (*TranscriptBackendRequestStrategy) ToolParameters() map[string]any { return map[string]any{} }

// ToolRequired implements BackendRequestStrategy.
func (*TranscriptBackendRequestStrategy) ToolRequired() []string { return nil }

// FrontendInstruction implements BackendRequestStrategy.
func (s *TranscriptBackendRequestStrategy) FrontendInstruction() string { return s.frontendInstruction }

// ToolDescription implements BackendRequestStrategy.
func (s *TranscriptBackendRequestStrategy) ToolDescription() string { return s.toolDescription }

// RequiresContext implements BackendRequestStrategy: the transcript is read
// from the frontend's context.
func (*TranscriptBackendRequestStrategy) RequiresContext() bool { return true }

// DelegateNeedsArguments implements BackendRequestStrategy: the request is the
// transcript, so a call with no arguments builds it.
func (*TranscriptBackendRequestStrategy) DelegateNeedsArguments() bool { return false }

// ComposeRequest renders the turns since the previous delegation as the
// request.
func (s *TranscriptBackendRequestStrategy) ComposeRequest(
	_ context.Context, params llm.FunctionCallParams,
) (string, bool) {
	messages := params.Context.Messages()
	s.mu.Lock()
	if s.delegatedThrough > len(messages) {
		// The context holds fewer messages than the previous delegation read: it
		// was reset since, so the transcript starts from the top.
		s.delegatedThrough = 0
	}
	// A message in a service's own format is not a user or assistant turn, so
	// the transcript leaves it out.
	var conversation []frames.Message
	for _, m := range messages[s.delegatedThrough:] {
		if !m.IsLLMSpecific() && !isBackendMessage(m) {
			conversation = append(conversation, m)
		}
	}
	first := s.delegatedThrough == 0
	s.delegatedThrough = len(messages)
	s.mu.Unlock()

	hasTurn := false
	for _, m := range conversation {
		hasTurn = hasTurn || isTurn(m)
	}
	if !hasTurn {
		return "", false
	}
	return llmworker.RenderTranscriptRequest(conversation, s.requestInstruction, first), true
}

// isTurn reports whether a message is a user or assistant turn. A tool's result
// is written with the user's role but is not the user speaking.
func isTurn(m frames.Message) bool {
	switch m.Role {
	case frames.RoleAssistant:
		return true
	case frames.RoleUser:
		return len(m.ToolResults) == 0
	default:
		return false
	}
}

// ExplicitBackendRequestStrategy has the frontend model word the request itself.
//
// The tool takes a request argument, which is sent to the backend as it stands.
// The default for a speech-to-speech frontend, which responds to audio before
// the transcript of that audio reaches its context, so the turns that prompted a
// handoff may not be recorded when the tool runs.
type ExplicitBackendRequestStrategy struct {
	frontendInstruction string
	toolDescription     string
	requestInstruction  string
}

// NewExplicitBackendRequestStrategy builds an explicit strategy.
func NewExplicitBackendRequestStrategy(texts StrategyTexts) *ExplicitBackendRequestStrategy {
	return &ExplicitBackendRequestStrategy{
		frontendInstruction: texts.frontendInstruction(ExplicitFrontendInstruction),
		toolDescription:     orDefault(texts.ToolDescription, ExplicitToolDescription),
		requestInstruction:  orDefault(texts.RequestInstruction, llmworker.ExplicitRequestInstruction),
	}
}

// ToolParameters implements BackendRequestStrategy: the request, as the model
// words it.
func (*ExplicitBackendRequestStrategy) ToolParameters() map[string]any {
	return map[string]any{
		requestArgument: map[string]any{"type": "string", "description": explicitRequestParameter},
	}
}

// ToolRequired implements BackendRequestStrategy.
func (*ExplicitBackendRequestStrategy) ToolRequired() []string { return []string{requestArgument} }

// FrontendInstruction implements BackendRequestStrategy.
func (s *ExplicitBackendRequestStrategy) FrontendInstruction() string { return s.frontendInstruction }

// ToolDescription implements BackendRequestStrategy.
func (s *ExplicitBackendRequestStrategy) ToolDescription() string { return s.toolDescription }

// RequiresContext implements BackendRequestStrategy: the request comes from the
// call's arguments.
func (*ExplicitBackendRequestStrategy) RequiresContext() bool { return false }

// DelegateNeedsArguments implements BackendRequestStrategy: the request is the
// call's request argument.
func (*ExplicitBackendRequestStrategy) DelegateNeedsArguments() bool { return true }

// ComposeRequest sends the model's request as it stands, with the reporting
// expectation after it.
func (s *ExplicitBackendRequestStrategy) ComposeRequest(
	_ context.Context, params llm.FunctionCallParams,
) (string, bool) {
	var args struct {
		Request string `json:"request"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	request := strings.TrimSpace(args.Request)
	if request == "" {
		return "", false
	}
	return llmworker.RenderExplicitRequest(request, s.requestInstruction), true
}
