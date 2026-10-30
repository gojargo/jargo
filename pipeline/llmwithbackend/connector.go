// Package llmwithbackend pairs a conversational frontend LLM with a backend that
// does the work it hands off.
//
// The frontend is any LLM service, text or speech-to-speech, and holds the
// conversation. The backend is an llmworker.BackendWorker running a heavier
// model with the tools, and does the work the frontend hands off. LLMWithBackend
// wraps the frontend so the pair drops into a pipeline where an LLM goes, and
// installs the tool that joins them.
//
// The two exchange messages. The frontend's delegate tool puts a message to the
// backend and returns at once; what the backend has to say comes back as a
// stream of outputs, each appended to the frontend's conversation as a message
// marked "Backend:", spoken or silent as the backend's flag says. A second
// request joins the first, and a correction changes or stops it. How a request
// is worded is the BackendRequestStrategy's business, and the BackendConnector
// owns the tool, the session with the backend, and how each output is rendered
// into the conversation.
//
// The package is alpha: its API is likely to change between releases.
package llmwithbackend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers/llmworker"
)

// DelegateToolName is the name of the tool the frontend calls to delegate.
const DelegateToolName = "delegate"

// How a backend's messages are marked in the frontend's conversation.
const (
	// BackendMessagePrefix marks a backend output.
	BackendMessagePrefix = "Backend: "
	// BackendThoughtPrefix marks a backend reasoning summary.
	BackendThoughtPrefix = "Backend (thinking): "
	// BackendNotePrefix marks a backend note: text its model wrote without the
	// mark that asks for the user to be told it.
	BackendNotePrefix = "Backend (note): "
	// BackendWorkingPrefix marks a function call the backend is making.
	BackendWorkingPrefix = "Backend (working): "
)

// backendPrefixes is every mark a rendered backend message can open with.
//
//nolint:gochecknoglobals // a fixed list, read only
var backendPrefixes = []string{
	BackendMessagePrefix, BackendThoughtPrefix, BackendNotePrefix, BackendWorkingPrefix,
}

// RelayNote is carried on each spoken backend message, so the frontend relays it
// on whatever turn it lands on, a user's turn included, where the standing
// instruction alone leaves it unsaid.
const RelayNote = "(Relay this to the user now, in your own words, once, even if the conversation has " +
	"moved on. If the user has just said something you have not answered yet, answer that " +
	"first and relay this at the end of the same reply.)"

// StrategyPlaceholder is where the request strategy's own guidance goes in
// FrontendPairingInstruction.
const StrategyPlaceholder = "{strategy}"

// The prompts that follow are read by the frontend's model. A small change in
// wording can change what a model does in ways no unit test catches: gpt-4.1
// stops delegating a "never mind", or starts delegating questions about how the
// work is going.

// FrontendPairingInstruction is how the frontend's model works with a backend,
// appended to its system instruction: what the backend is and when to
// delegate, that to the user there is one assistant, what to do after
// delegating, and what to do with the backend's messages. The request
// strategy's guidance (how the request is worded) goes where
// StrategyPlaceholder stands, at the end of the first paragraph. The frontend's
// own system instruction says what the backend is for.
// ConnectorConfig.FrontendPairingInstruction replaces it.
//
// The instruction ends by restating its one rule, that a message about
// delegated work, a stop included, gets a delegate call. A stop is where a
// model goes wrong: it answers "never mind" with "I won't make any changes" and
// no call, and the backend carries on. gpt-4.1 weighs the end of the
// instruction most, so the restatement is what makes it delegate a stop; the
// first paragraph alone, with the rule in it, does not, nor does the rule as a
// list of cases.
const FrontendPairingInstruction = "THE BACKEND: You have a backend for anything that needs a " +
	"backend tool or careful reasoning. It sees nothing of the conversation but what you delegate, and nothing you " +
	"say reaches it: to start, change or stop its work, call the delegate tool, at once, in " +
	"the same reply, and do the rest of the reply yourself around it. That holds for every " +
	"message that concerns its work, whether a new request, a yes to something that was " +
	"offered, a correction or a \"never mind\", and even while earlier work is still running: " +
	"a stop you only speak leaves the work running. Do not delegate what you can do " +
	"yourself: small talk, a question that a result you already have answers, or a question " +
	"about how the work is going, which you answer from the backend's latest notes or, with " +
	"none yet, by saying there is no update yet. Never guess a result while waiting. " +
	StrategyPlaceholder + "\n\n" +
	"ONE ASSISTANT: To the user there is one assistant, you: never mention the backend, " +
	"delegation, a handoff or tools. Say what is being done as your own doing, such as \"I'm " +
	"checking\", never \"the backend is checking\", and never that you passed anything along.\n\n" +
	"AFTER DELEGATING: Acknowledge in one short sentence of your own that fits the request, " +
	"naming the task briefly if that reads naturally. For example: " +
	"\"Let me look into that\", \"Sure, I'll get on that now\" or \"Okay, pulling that up now\". An " +
	"acknowledgement is never a substitute for the delegate call. In the acknowledgement, avoid " +
	"restating the request in full, or describing in detail what you are going to do, or " +
	"following up on your own with promises, offers or questions: you've fired off the delegate " +
	"call, so let it be. " +
	"Then answer whatever else the user asked that you can. Never say work is done, or give a " +
	"result, that the backend has not given you; if the user asks whether work is under way and " +
	"you never delegated it, say so and delegate it then.\n\n" +
	"BACKEND MESSAGES: The backend works on its own after you delegate and may be doing " +
	"several things at once. What it has to say arrives as messages in the conversation " +
	"marked \"Backend:\": a result, a question for the user, or news " +
	"worth an update, on whatever turn happens to be in progress by then. Such a message is " +
	"owed to the user, whatever the conversation has moved on to: small talk, other " +
	"questions or a long wait do not make it unwanted. If the user has said something you " +
	"have not answered yet, answer that first, then relay the message at the end of that " +
	"same reply, in your own words. If you have already answered everything the user said, " +
	"relay just the message; do not repeat or rephrase your earlier reply. Relay each " +
	//nolint:misspell // the prompt as upstream words it
	"message once. Only when the user has cancelled or changed what they asked for does a " +
	"message go unsaid, and then say only what still helps. Messages marked " +
	"\"Backend (working):\", \"Backend (note):\" and " +
	"\"Backend (thinking):\" are what the backend is doing and thinking. They " +
	"carry no results: what a call found reaches you only in a message marked " +
	"\"Backend:\", so never describe or guess what a call found. Never " +
	"relay them on their own, but when the user asks how the work is going, answer from the " +
	"latest of them, concretely: name the step, such as which file it is reading or that it " +
	"is running the tests, rather than saying only that it is still working. Never state a " +
	"result you have not received from the backend, and never say work is done that the " +
	"backend has not said is done.\n\n" +
	"Whatever else a reply does, a user message about work you delegated, whether to add to " +
	"it, change it or stop it, a never mind included, gets a delegate call in that reply: " +
	"nothing else stops or changes that work."

// TranscriptFrontendInstruction is the transcript strategy's guidance, in the
// pairing instruction's delegating paragraph.
const TranscriptFrontendInstruction = "A delegation hands over the whole conversation, not one item: the " +
	"backend reads it and does everything in it that is its job, so delegate once per reply however " +
	"many things the user asked for, and do not word the request."

// TranscriptToolDescription is how the transcript strategy describes the
// delegate tool.
const TranscriptToolDescription = "The only way to start, change or stop the backend's work. Hands " +
	"the conversation over to the backend, which reads it and does everything in it that needs a " +
	"backend tool or careful reasoning, and stops or changes what the user has called off or " +
	"changed. Call it as soon as any part of what the user says concerns that work, even while " +
	"the backend is still working, and do the rest yourself. Not for a question about how the " +
	"work is going, which you answer from the backend's notes. One handoff per reply, however " +
	"many things the user asked for. It takes no arguments. Keep talking with the user while " +
	"it works."

// ExplicitFrontendInstruction is the explicit strategy's guidance, in the
// pairing instruction's delegating paragraph.
const ExplicitFrontendInstruction = "Word the request so it stands on its own: the user's goal, the exact " +
	"details they gave and their latest correction, with everything new they asked for in " +
	"the one request, so you delegate once per reply however many things that is. Work " +
	"already handed over stays with the backend: a new request joins it, so do not restate " +
	"earlier requests; a correction names what changes, and a stop names what to stop."

// ExplicitToolDescription is how the explicit strategy describes the delegate
// tool.
const ExplicitToolDescription = "The only way to start, change or stop the backend's work. Hands " +
	"a request to the backend, which does everything in it that needs a backend tool or careful " +
	"reasoning, and stops or changes what the user has called off or changed. Call it as soon " +
	"as any part of what the user says concerns that work, even while the backend is still " +
	"working, with the request worded to stand on its own, and do the rest yourself. Not for a " +
	"question about how the work is going, which you answer from the backend's notes. One call " +
	"per reply, however many things the user asked for. Work already handed over stays with the " +
	"backend, so send only what is new, including anything the user says about that work, such " +
	"as to stop or change it. Keep talking with the user while it works."

// requestArgument is the explicit strategy's one argument, the request as the
// model words it.
const requestArgument = "request"

// traceEvent is the field of a traced exchange that says what it was.
const traceEvent = "event"

// explicitRequestParameter describes the explicit strategy's one argument.
const explicitRequestParameter = "What the user is asking for now, self-contained: their goal, the details " +
	"they gave (places, dates, names) and their latest correction, or what they " +
	"said about work already under way, such as to stop it or change it. Work " +
	"already handed over stays with the backend; a new request joins it, so do " +
	"not restate earlier requests."

// ConnectorContext is what a connector needs to know about the layers it joins.
type ConnectorContext struct {
	// BackendName is the name of the backend worker, local or registered
	// elsewhere.
	BackendName string
	// FrontendIsRealtime reports whether the frontend is a speech-to-speech
	// service, whose context can lag the audio, which changes how a delegation
	// words its request.
	FrontendIsRealtime bool
}

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
	// StrategyPlaceholder stands.
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

// isBackendMessage reports whether a context message is one the connector
// rendered from a backend output.
func isBackendMessage(m frames.Message) bool {
	if m.Role != frames.RoleDeveloper {
		return false
	}
	for _, prefix := range backendPrefixes {
		if strings.HasPrefix(m.Text, prefix) {
			return true
		}
	}
	return false
}

// backendSession is what the connector needs of the session with the backend.
type backendSession interface {
	Send(ctx context.Context, request string) (string, error)
}

// frontendFrames is what the connector needs of the frontend to deliver what
// the backend produced: somewhere to queue a message and to push a report.
type frontendFrames interface {
	QueueFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error
	PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error
}

// ErrStrategyNeedsContext is reported when a speech-to-speech frontend is given
// a request strategy that reads the frontend's context, which lags the audio
// there.
var ErrStrategyNeedsContext = errors.New("the request strategy reads the frontend's context, " +
	"which lags the audio on a speech-to-speech frontend")

// errNotAttached is reported by a delegation when the session with the backend
// does not open in time.
var errNotAttached = errors.New("the backend is not attached")

// ConnectorConfig configures a BackendConnector. The zero value is the default
// connector.
type ConnectorConfig struct {
	// RequestStrategy is how a delegate call becomes the backend's request.
	// Picked by frontend kind when nil.
	RequestStrategy BackendRequestStrategy
	// FrontendPairingInstruction is how the model works with a backend,
	// appended to its system instruction: FrontendPairingInstruction when empty.
	// One of its own has to say what that one says: when and how to delegate,
	// that to the user there is one assistant, what to do after delegating, and
	// what to do with the backend's messages. The request strategy's guidance
	// goes where StrategyPlaceholder stands, or after the text when it has none.
	FrontendPairingInstruction string
	// RespondOnDelegate reports whether the frontend runs again as soon as a
	// delegation is sent, which is where its acknowledgement comes from: a text
	// model's tool-call turn carries no prose. Nil means true; set it false for a
	// frontend that announces a handoff itself.
	RespondOnDelegate *bool
	// Timeout is how long the backend may take to acknowledge a message,
	// including the wait for it to become ready. Zero means 30 seconds;
	// negative waits as long as it takes.
	Timeout time.Duration
	// ClientTrace sends every exchange with the backend to the client as an RTVI
	// server message (type "llm-with-backend"): each request, output, tool-call
	// phase, error and idle. For debugging; a client's event log shows them.
	ClientTrace bool
	// RenderOutput renders a backend output as the message the frontend's
	// conversation takes in, or reports false to deliver nothing for it.
	// DefaultRenderOutput when nil.
	RenderOutput func(output llmworker.BackendOutput) (frames.Message, bool)
	// RenderToolCall renders a phase of a backend function call as a message the
	// frontend's conversation takes in silently, or reports false to record
	// nothing for it. DefaultRenderToolCall when nil.
	RenderToolCall func(call llmworker.BackendToolCall) (frames.Message, bool)
	// BuildTools builds the tools to install on the frontend, each carrying its
	// handler, for tools of another shape entirely. The connector's delegate
	// tool when nil; the service reads nothing else from the connector but its
	// frontend instruction.
	BuildTools func(c *BackendConnector) []frames.Tool
}

// defaultTimeout is how long the backend may take to acknowledge a message
// when the configuration does not say.
const defaultTimeout = 30 * time.Second

// BackendConnector joins the two layers: the tools the frontend calls, the
// session with the backend, and what each output becomes.
//
// A request strategy defines the delegate tool and composes the request. It may
// be given, or left to the connector to pick by frontend kind once bound:
// TranscriptBackendRequestStrategy for a text frontend,
// ExplicitBackendRequestStrategy for a speech-to-speech one, whose context lags
// the audio. A given strategy that reads the frontend's context is refused for a
// speech-to-speech frontend.
//
// The connector holds the session with the backend for the frontend's life, and
// turns each output the backend sends into a message appended to the frontend's
// conversation, run or not as the output's PrefersSpoken flag says. The function
// calls the backend makes on the way are reported in the frontend's pipeline for
// clients to show; nothing else in the pipeline sees them.
type BackendConnector struct {
	cfg             ConnectorConfig
	requestStrategy BackendRequestStrategy
	timeout         time.Duration

	mu          sync.Mutex
	bound       *ConnectorContext
	tools       []frames.Tool
	session     backendSession
	sessionOpen chan struct{}
}

// NewBackendConnector builds a connector.
func NewBackendConnector(cfg ConnectorConfig) *BackendConnector {
	timeout := cfg.Timeout
	switch {
	case timeout == 0:
		timeout = defaultTimeout
	case timeout < 0:
		timeout = 0
	}
	return &BackendConnector{
		cfg:             cfg,
		requestStrategy: cfg.RequestStrategy,
		timeout:         timeout,
		sessionOpen:     make(chan struct{}),
	}
}

// RequestStrategy is the request strategy in use, once bound.
func (c *BackendConnector) RequestStrategy() BackendRequestStrategy { return c.requestStrategy }

// Tools are the tools to install on the frontend, once bound.
func (c *BackendConnector) Tools() []frames.Tool { return c.tools }

// Tool is the delegate tool, once bound.
func (c *BackendConnector) Tool() frames.Tool { return c.tools[0] }

// FrontendInstruction is what is appended to the frontend's system
// instruction: the pairing instruction with the strategy's guidance in place,
// once bound.
func (c *BackendConnector) FrontendInstruction() string {
	pairing := orDefault(c.cfg.FrontendPairingInstruction, FrontendPairingInstruction)
	guidance := c.requestStrategy.FrontendInstruction()
	if strings.Contains(pairing, StrategyPlaceholder) {
		placed := strings.ReplaceAll(pairing, StrategyPlaceholder, guidance)
		return strings.TrimSpace(strings.ReplaceAll(placed, " \n", "\n"))
	}
	var parts []string
	for _, p := range []string{pairing, guidance} {
		if p != "" {
			parts = append(parts, p)
		}
	}
	return strings.Join(parts, "\n\n")
}

// Bind settles the strategy for the layers being joined and builds the tools.
// It reports an error when the frontend is speech-to-speech and the strategy
// reads the frontend's context, which lags the audio there.
func (c *BackendConnector) Bind(bound ConnectorContext) error {
	c.bound = &bound
	if c.requestStrategy == nil {
		if bound.FrontendIsRealtime {
			c.requestStrategy = NewExplicitBackendRequestStrategy(StrategyTexts{})
		} else {
			c.requestStrategy = NewTranscriptBackendRequestStrategy(StrategyTexts{})
		}
	}
	if bound.FrontendIsRealtime && c.requestStrategy.RequiresContext() {
		return fmt.Errorf("%w: %T; use a request strategy that does not read the context, such as "+
			"ExplicitBackendRequestStrategy", ErrStrategyNeedsContext, c.requestStrategy)
	}
	if c.cfg.BuildTools != nil {
		c.tools = c.cfg.BuildTools(c)
	} else {
		c.tools = c.BuildTools()
	}
	return nil
}

// BuildTools builds the tools to install on the frontend: delegate.
func (c *BackendConnector) BuildTools() []frames.Tool {
	schema := map[string]any{"type": "object", "properties": c.requestStrategy.ToolParameters()}
	if required := c.requestStrategy.ToolRequired(); len(required) > 0 {
		schema["required"] = required
	}
	params, err := json.Marshal(schema)
	if err != nil {
		params = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return []frames.Tool{{
		Name:        DelegateToolName,
		Description: c.requestStrategy.ToolDescription(),
		Parameters:  params,
		Handler:     llm.FunctionCallHandler(c.Delegate),
	}}
}

// RunSession holds the session with the backend, delivering what it produces
// to the frontend, until ctx ends. requester is the pipeline worker the
// frontend runs in.
func (c *BackendConnector) RunSession(ctx context.Context, requester jobcontext.Requester, frontend frontendFrames) {
	name := c.bound.BackendName
	session, err := llmworker.OpenBackendSession(ctx, requester, name, c.timeout)
	if err != nil {
		if ctx.Err() == nil {
			slog.ErrorContext(ctx, "the session with the backend failed", "backend", name, "err", err)
		}
		return
	}
	defer session.Close(ctx)
	c.setSession(session)
	defer c.setSession(nil)

	for {
		event, err := session.Next(ctx)
		switch {
		case err == nil:
			c.Deliver(ctx, frontend, event)
			continue
		case ctx.Err() != nil:
		case errors.Is(err, io.EOF):
			if !session.Detached() {
				slog.WarnContext(ctx, "the backend went away", "backend", name)
			}
		default:
			slog.ErrorContext(ctx, "the session with the backend failed", "backend", name, "err", err)
		}
		return
	}
}

// setSession records the session with the backend, or that there is none.
func (c *BackendConnector) setSession(session backendSession) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if session != nil {
		c.session = session
		close(c.sessionOpen)
		return
	}
	c.session = nil
	c.sessionOpen = make(chan struct{})
}

// openSession is the session, once it is open; it waits for it up to the
// timeout.
func (c *BackendConnector) openSession(ctx context.Context) (backendSession, error) {
	c.mu.Lock()
	session, open := c.session, c.sessionOpen
	c.mu.Unlock()
	if session != nil {
		return session, nil
	}
	var expired <-chan time.Time
	if c.timeout > 0 {
		timer := time.NewTimer(c.timeout)
		defer timer.Stop()
		expired = timer.C
	}
	select {
	case <-open:
	case <-expired:
		return nil, errNotAttached
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == nil {
		return nil, errNotAttached
	}
	return c.session, nil
}

// Delegate puts the request a delegate call carries to the backend, and settles
// the call at once.
func (c *BackendConnector) Delegate(ctx context.Context, params llm.FunctionCallParams) error {
	request, ok := c.requestStrategy.ComposeRequest(ctx, params)
	if !ok {
		// No turns since the previous delegation: the backend already has
		// everything the transcript would carry, so nothing is sent.
		slog.DebugContext(ctx, "delegate call carries nothing new; not sent", "tool_call_id", params.ToolCallID)
		return params.Result(ctx, resultJSON(map[string]any{"status": "already_delegated"}), runLLM(false))
	}
	slog.DebugContext(ctx, "delegating", "backend", c.bound.BackendName, "request", request)
	session, err := c.openSession(ctx)
	if err != nil {
		return err
	}
	status, err := session.Send(ctx, request)
	if err != nil {
		return err
	}
	if params.LLM != nil {
		c.trace(ctx, params.LLM, map[string]any{traceEvent: "request", "text": request, "backend": status})
	}
	respond := c.cfg.RespondOnDelegate == nil || *c.cfg.RespondOnDelegate
	return params.Result(ctx, resultJSON(map[string]any{"status": "delegated", "backend": status}), runLLM(respond))
}

// runLLM is result properties that say whether the result runs the model.
func runLLM(run bool) *frames.FunctionCallResultProperties {
	return &frames.FunctionCallResultProperties{RunLLM: &run}
}

// resultJSON renders a tool result.
func resultJSON(v map[string]any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// Deliver delivers one thing the backend produced to the frontend.
//
// An output becomes a message appended to the frontend's conversation, which
// runs the frontend when the output asks to be spoken. A function-call phase is
// reported as the ExternalFunctionCall frame for it, which the RTVI observer
// turns into function-call events. An error becomes a spoken message, so the
// user hears the work stopped. The appended messages are uninterruptible: one
// queued while the bot speaks waits behind that speech, and an interruption
// clears the speech but must leave the message to be recorded.
func (c *BackendConnector) Deliver(ctx context.Context, frontend frontendFrames, event llmworker.BackendEvent) {
	switch e := event.(type) {
	case llmworker.BackendOutput:
		c.trace(ctx, frontend, map[string]any{
			traceEvent: "output", "text": e.Text, "spoken": e.PrefersSpoken, "thought": e.IsThought,
		})
		slog.DebugContext(ctx, "backend output", "spoken", e.PrefersSpoken, "thought", e.IsThought, "text", e.Text)
		render := c.cfg.RenderOutput
		if render == nil {
			render = DefaultRenderOutput
		}
		if message, ok := render(e); ok {
			c.appendMessage(ctx, frontend, message, e.PrefersSpoken)
		}
	case llmworker.BackendToolCall:
		c.trace(ctx, frontend, map[string]any{
			traceEvent: "tool_call", "phase": string(e.Phase), "name": e.FunctionName,
			"arguments": e.ToPayload()["arguments"], "result": e.Result,
		})
		if err := frontend.PushFrame(ctx, e.ToFrame(), processor.Downstream); err != nil {
			slog.DebugContext(ctx, "could not report a backend call", "err", err)
		}
		render := c.cfg.RenderToolCall
		if render == nil {
			render = DefaultRenderToolCall
		}
		if message, ok := render(e); ok {
			c.appendMessage(ctx, frontend, message, false)
		}
	case llmworker.BackendError:
		c.trace(ctx, frontend, map[string]any{traceEvent: "error", "error": e.Error})
		slog.WarnContext(ctx, "backend error", "error", e.Error)
		c.appendMessage(ctx, frontend, frames.Message{
			Role: frames.RoleDeveloper,
			Text: BackendMessagePrefix + "The work could not be completed.\n\n" + RelayNote,
		}, true)
	case llmworker.BackendIdle:
		c.trace(ctx, frontend, map[string]any{traceEvent: "idle"})
		slog.DebugContext(ctx, "backend is idle")
	}
}

// trace sends one exchange with the backend to the client, when tracing is on.
func (c *BackendConnector) trace(ctx context.Context, frontend frontendFrames, data map[string]any) {
	if !c.cfg.ClientTrace {
		return
	}
	data["type"] = "llm-with-backend"
	if err := frontend.PushFrame(ctx, rtvi.NewServerMessageFrame(data), processor.Downstream); err != nil {
		slog.DebugContext(ctx, "could not trace a backend exchange", "err", err)
	}
}

// appendMessage queues a message into the frontend's conversation, past any
// interruption on the way.
func (c *BackendConnector) appendMessage(ctx context.Context, frontend frontendFrames, m frames.Message, run bool) {
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{m})
	f.RunLLM = run
	f.SetInterruptible(false)
	if err := frontend.QueueFrame(ctx, f, processor.Downstream); err != nil {
		slog.DebugContext(ctx, "could not deliver a backend message", "err", err)
	}
}

// DefaultRenderOutput renders a backend output as the message the frontend's
// conversation takes in. It marks the message so the frontend can tell it from
// the user's and its own, and so the transcript strategy can leave it out of
// what it sends the backend: a spoken output as "Backend:" with RelayNote after
// it, a silent one as "Backend (note):", a reasoning summary as
// "Backend (thinking):". An output with no text delivers nothing.
func DefaultRenderOutput(output llmworker.BackendOutput) (frames.Message, bool) {
	switch {
	case output.Text == "":
		return frames.Message{}, false
	case output.IsThought:
		return frames.Message{Role: frames.RoleDeveloper, Text: BackendThoughtPrefix + output.Text}, true
	case !output.PrefersSpoken:
		return frames.Message{Role: frames.RoleDeveloper, Text: BackendNotePrefix + output.Text}, true
	default:
		return frames.Message{
			Role: frames.RoleDeveloper, Text: BackendMessagePrefix + output.Text + "\n\n" + RelayNote,
		}, true
	}
}

// DefaultRenderToolCall renders a phase of a backend function call as a message
// the frontend's conversation takes in, silently. It records each call as it
// starts, with its arguments, so the frontend can say what the backend is doing
// if asked; the other phases are not recorded.
func DefaultRenderToolCall(call llmworker.BackendToolCall) (frames.Message, bool) {
	if call.Phase != llmworker.BackendToolCallInProgress {
		return frames.Message{}, false
	}
	return frames.Message{
		Role: frames.RoleDeveloper,
		Text: BackendWorkingPrefix + call.FunctionName + "(" + renderArguments(call.Arguments) + ")",
	}, true
}

// renderArguments renders a call's arguments as name=value pairs, in the order
// the model gave them, each value written the way the backend's own logs and
// prompts write one.
func renderArguments(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return ""
	}
	var parts []string
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			break
		}
		var value any
		if err := dec.Decode(&value); err != nil {
			break
		}
		parts = append(parts, fmt.Sprint(key)+"="+reprValue(value))
	}
	return strings.Join(parts, ", ")
}

// reprValue writes a JSON value as a literal: strings quoted with single
// quotes, true and false capitalized, null as None, as the conversation already
// shows values elsewhere.
func reprValue(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case json.Number:
		return x.String()
	case string:
		return "'" + strings.ReplaceAll(strings.ReplaceAll(x, `\`, `\\`), "'", `\'`) + "'"
	case []any:
		items := make([]string, 0, len(x))
		for _, item := range x {
			items = append(items, reprValue(item))
		}
		return "[" + strings.Join(items, ", ") + "]"
	case map[string]any:
		items := make([]string, 0, len(x))
		for _, k := range slices.Sorted(maps.Keys(x)) {
			items = append(items, reprValue(k)+": "+reprValue(x[k]))
		}
		return "{" + strings.Join(items, ", ") + "}"
	default:
		return strconv.Quote(fmt.Sprint(x))
	}
}
