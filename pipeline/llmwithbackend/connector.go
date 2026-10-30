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

// The connector, which joins a frontend to its backend: it owns the tools the
// frontend calls, the session with the backend, and how each backend output is
// rendered into the frontend's conversation.

// DelegateToolName is the name of the tool the frontend calls to delegate.
const DelegateToolName = "delegate"

// traceEvent is the field of a traced exchange that says what it was.
const traceEvent = "event"

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
	// goes where PairingInstructionStrategyPlaceholder stands, or after the text
	// when it has none.
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
	if strings.Contains(pairing, PairingInstructionStrategyPlaceholder) {
		placed := strings.ReplaceAll(pairing, PairingInstructionStrategyPlaceholder, guidance)
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
			Text: BackendMessagePrefix + "The work could not be completed.\n\n" + BackendMessageRelayInstruction,
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
// what it sends the backend: a spoken output as "Backend:" with
// BackendMessageRelayInstruction after it, a silent one as "Backend (note):", a
// reasoning summary as "Backend (thinking):". An output with no text delivers
// nothing.
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
			Role: frames.RoleDeveloper, Text: BackendMessagePrefix + output.Text + "\n\n" + BackendMessageRelayInstruction,
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
