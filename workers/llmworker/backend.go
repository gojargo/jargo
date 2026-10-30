package llmworker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"

	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// A backend worker does the work a frontend hands off. The frontend holds the
// conversation and hands off requests that need tools or careful reasoning.
// What the frontend is does not matter to this contract: a speech-to-speech
// model delegating on its own, or a pipeline calling a tool. A BackendWorker runs
// any LLM service, with its own context and multi-step tool calling, to do that
// work.
//
// The frontend and the backend exchange messages. A frontend attaches to the
// worker once, for as long as it lives, and from then on hears everything the
// backend produces as a stream: each of its model's turns and reasoning
// summaries as a BackendOutput, each phase of the function calls it makes as a
// BackendToolCall, and when it has run out of things to do, that it is idle.
// Requests go the other way as messages appended to the backend's conversation,
// each answered at once; a message that arrives while the backend is working
// joins that work, and the backend's model sorts out what it means.
// BackendSession is the caller side of that contract.

// The names of the jobs a BackendWorker answers.
const (
	// AttachJobName is the job that attaches a frontend to a BackendWorker.
	AttachJobName = "attach"
	// MessageJobName is the job that puts a message to an attached
	// BackendWorker.
	MessageJobName = "message"
)

// The types of the updates an attach job's stream carries.
const (
	// OutputUpdateType is the type of an update that carries a BackendOutput.
	OutputUpdateType = "output"
	// ToolCallUpdateType is the type of an update that carries a
	// BackendToolCall.
	ToolCallUpdateType = "tool_call"
	// AttachedUpdateType is the type of the update that opens an attach job's
	// stream.
	AttachedUpdateType = "attached"
	// IdleUpdateType is the type of the update that says the backend has
	// nothing left to do.
	IdleUpdateType = "idle"
	// ErrorUpdateType is the type of the update that says the backend could
	// not go on.
	ErrorUpdateType = "error"
)

// The keys of an update payload that every update type shares.
const (
	payloadType  = "type"
	payloadError = "error"
)

// ErrRoleDeclined is reported by NewBackend when the LLM service cannot be the
// backend; the error names the service's own reason.
var ErrRoleDeclined = errors.New("the LLM service declines the backend role")

// SpokenMark is the mark the backend's model puts at the start of a message it
// wants the user told. The message is sent with PrefersSpoken set, mark
// stripped; a message without it is a note. Strict prompt, flexible parsing:
// the model is told to put the mark first, but sometimes writes a note paragraph
// and then a marked one, so a mark at the start of a later line splits the
// message, a note before it and a spoken message from it on, rather than losing
// the result.
const SpokenMark = ">>"

// spokenMarkLine finds where a message's spoken part starts: the first line
// that begins with the mark.
var spokenMarkLine = regexp.MustCompile(`(?m)^[ \t]*` + regexp.QuoteMeta(SpokenMark) + `[ \t]*`)

// BackendPairingInstruction is how the backend's model works with the
// assistant, appended to its system instruction: who it is, which of what it
// writes the user hears, and how it takes requests. The backend's own system
// instruction says what it does; this says how what it writes reaches the user.
// BackendConfig.PairingInstruction replaces it.
const BackendPairingInstruction = "You are the backend of a voice assistant. The assistant talks with " +
	"the user and sends you requests: the conversation it is having, or a request it worded for you. " +
	"Do the parts that need your tools or careful reasoning; the assistant handles the rest of the " +
	"conversation itself, such as small talk, jokes and stories, so leave those to it.\n\n" +
	"WHAT THE USER HEARS: Begin a message with " + SpokenMark + " and the whole of it is told to " +
	"the user, in the assistant's own words, tool calls or not. Anything you write without " +
	"the mark is not told to the user: the assistant keeps it as notes on your work, and " +
	"draws on them only if the user asks how the work is going. " +
	"Mark a message to give a result, to ask a question you cannot proceed without, or for " +
	"news the user should have now. A result you do not mark is " +
	"lost: the message that gives it begins with " + SpokenMark + ". Never say again what you " +
	"have already marked, and never mark that you have stopped or that there is nothing to " +
	"do: the assistant has already told the user. Write a marked message as plain spoken " +
	"sentences the assistant can say as they are: no headings, lists, Markdown or raw " +
	"JSON.\n\n" +
	"REQUESTS: A request may arrive while you are working on an earlier one. It may add " +
	"work, change it, or stop some or all of it: follow the latest, cancel the tools whose " +
	"results are no longer wanted, and do not repeat work already done. With several " +
	"requests in hand, do the newest first unless it depends on an earlier one, and give " +
	"each one's result as soon as you have it, not when everything is done."

// TranscriptRequestInstruction is what RenderTranscriptRequest tells the backend
// to do with a transcript.
const TranscriptRequestInstruction = "Act on the user's most recent request in the conversation above. " +
	"If it asks for something, tell the user the result as soon as you have it, before going on with " +
	"other work, in a message that begins with " + SpokenMark + ". If it only stops or changes work " +
	"already under way, tell them nothing: the assistant already has."

// ExplicitRequestInstruction is appended to a request the frontend worded
// itself, so the backend tells the user its result on its own rather than with
// whatever else it is doing.
const ExplicitRequestInstruction = "If this request asks for something, tell the user the result as " +
	"soon as you have it, before going on with other work, in a message that begins with " +
	SpokenMark + ". If it only stops or changes work already under way, tell them nothing: the " +
	"assistant already has."

// BackendEvent is what a BackendSession yields: a BackendOutput, a
// BackendToolCall, a BackendIdle or a BackendError.
type BackendEvent interface {
	isBackendEvent()
}

// BackendOutput is one piece of output from a backend, on its way to the
// frontend.
type BackendOutput struct {
	// Text is the text the backend produced.
	Text string
	// IsThought reports whether this is a reasoning summary rather than a
	// response.
	IsThought bool
	// PrefersSpoken reports whether the backend would like the user to hear
	// this. It is a hint the frontend may follow or not. A literal leaves it
	// false; NewBackendOutput sets it, as an output the app sends is meant to be
	// heard unless it says otherwise.
	PrefersSpoken bool
}

// NewBackendOutput builds an output the backend would like the user to hear.
func NewBackendOutput(text string) BackendOutput {
	return BackendOutput{Text: text, PrefersSpoken: true}
}

func (BackendOutput) isBackendEvent() {}

// ToPayload renders the output as a job update payload.
func (o BackendOutput) ToPayload() map[string]any {
	return map[string]any{
		payloadType:      OutputUpdateType,
		"text":           o.Text,
		"is_thought":     o.IsThought,
		"prefers_spoken": o.PrefersSpoken,
	}
}

// BackendOutputFromPayload rebuilds an output from a job update payload.
//
// A flag the payload omits is left to its default, so a sender that predates a
// flag does not decide its value: an output is spoken and not a thought unless
// the payload says otherwise. A flag it carries is coerced, since a payload can
// arrive from another process.
func BackendOutputFromPayload(payload map[string]any) BackendOutput {
	out := NewBackendOutput(payloadString(payload, "text"))
	if v, ok := payload["is_thought"]; ok {
		out.IsThought = truthy(v)
	}
	if v, ok := payload["prefers_spoken"]; ok {
		out.PrefersSpoken = truthy(v)
	}
	return out
}

// BackendToolCallPhase is where a function call the backend made has got to.
type BackendToolCallPhase string

// The phases of a backend function call, which mirror the pipeline's own
// function-call frames.
const (
	BackendToolCallStarted    BackendToolCallPhase = "started"
	BackendToolCallInProgress BackendToolCallPhase = "in_progress"
	BackendToolCallResult     BackendToolCallPhase = "result"
	BackendToolCallCanceled   BackendToolCallPhase = "cancelled" //nolint:misspell // the value on the wire
)

// BackendToolCall is one phase of a function call the backend made while
// working, on its way to the frontend.
//
// The call ran in the backend's own pipeline; the frontend reports it to
// clients, as the ExternalFunctionCall frame ToFrame builds, and does nothing
// else with it.
type BackendToolCall struct {
	// Phase is where the call has got to.
	Phase BackendToolCallPhase
	// FunctionName is the name of the function called.
	FunctionName string
	// ToolCallID identifies the call.
	ToolCallID string
	// Arguments is the raw JSON arguments passed to the function, once known.
	Arguments json.RawMessage
	// Result is the result, for the result phase.
	Result string
	// IsFinal reports, for the result phase, whether the result completes the
	// call rather than being one of a stream of intermediate results.
	IsFinal bool
}

func (BackendToolCall) isBackendEvent() {}

// ToPayload renders the call phase as a job update payload. The arguments
// travel as the JSON object they are, so a payload crossing a bus carries them
// as data rather than as encoded text.
func (c BackendToolCall) ToPayload() map[string]any {
	var arguments any
	if len(c.Arguments) > 0 {
		if err := json.Unmarshal(c.Arguments, &arguments); err != nil {
			arguments = string(c.Arguments)
		}
	}
	return map[string]any{
		payloadType:     ToolCallUpdateType,
		"phase":         string(c.Phase),
		"function_name": c.FunctionName,
		"tool_call_id":  c.ToolCallID,
		"arguments":     arguments,
		"result":        c.Result,
		"is_final":      c.IsFinal,
	}
}

// BackendToolCallFromPayload rebuilds a call phase from a job update payload.
func BackendToolCallFromPayload(payload map[string]any) BackendToolCall {
	phase := BackendToolCallPhase(payloadString(payload, "phase"))
	if phase == "" {
		phase = BackendToolCallInProgress
	}
	call := BackendToolCall{
		Phase:        phase,
		FunctionName: payloadString(payload, "function_name"),
		ToolCallID:   payloadString(payload, "tool_call_id"),
		Result:       payloadString(payload, "result"),
		IsFinal:      true,
	}
	switch args := payload["arguments"].(type) {
	case nil:
	case string:
		call.Arguments = json.RawMessage(args)
	default:
		if raw, err := json.Marshal(args); err == nil {
			call.Arguments = raw
		}
	}
	if v, ok := payload["is_final"]; ok {
		call.IsFinal = truthy(v)
	}
	return call
}

// ToFrame builds the frame that reports this phase in the frontend's pipeline.
func (c BackendToolCall) ToFrame() frames.Frame {
	switch c.Phase {
	case BackendToolCallStarted:
		// The frame can announce several calls at once, as the pipeline's own
		// FunctionCallsStartedFrame does; a backend reports its calls one at a
		// time, so here it carries one.
		return frames.NewExternalFunctionCallsStartedFrame([]frames.ExternalFunctionCall{
			{ToolName: c.FunctionName, ToolCallID: c.ToolCallID},
		})
	case BackendToolCallInProgress:
		return frames.NewExternalFunctionCallInProgressFrame(c.ToolCallID, c.FunctionName, c.Arguments)
	case BackendToolCallResult:
		f := frames.NewExternalFunctionCallResultFrame(c.ToolCallID, c.FunctionName, c.Arguments, c.Result)
		f.IsFinal = c.IsFinal
		return f
	default:
		return frames.NewExternalFunctionCallCancelFrame(c.ToolCallID, c.FunctionName)
	}
}

// BackendIdle says the backend has nothing left to do: no run outstanding, no
// call in flight.
type BackendIdle struct{}

func (BackendIdle) isBackendEvent() {}

// BackendError says the backend could not go on with what it was doing.
type BackendError struct {
	// Error is what went wrong.
	Error string
}

func (BackendError) isBackendEvent() {}

// BackendOutputTransform shapes one of the backend model's outputs before it is
// sent. It returns the output to send in its place, or nil to send nothing.
type BackendOutputTransform func(ctx context.Context, output BackendOutput) (*BackendOutput, error)

// RenderExplicitRequest renders a request the frontend worded itself, with what
// the backend should do with it after. An empty instruction uses
// ExplicitRequestInstruction.
func RenderExplicitRequest(request, instruction string) string {
	if instruction == "" {
		instruction = ExplicitRequestInstruction
	}
	return request + "\n\n" + instruction
}

// splitSpoken splits a message the model wrote into its note and its spoken
// part, by the mark: the note is what precedes the first line that begins with
// the mark, or the whole message when no line does, and the spoken part is what
// follows the mark, with any later mark removed, or empty. Both are trimmed.
func splitSpoken(text string) (note, spoken string) {
	loc := spokenMarkLine.FindStringIndex(text)
	if loc == nil {
		return text, ""
	}
	// From the first marked line on, the whole of it is spoken; a mark the
	// model put on a later line is not meant for the user's ear.
	rest := spokenMarkLine.ReplaceAllString(text[loc[1]:], "")
	return strings.TrimSpace(text[:loc[0]]), strings.TrimSpace(rest)
}

// RenderTranscriptRequest renders a conversation as a labeled transcript for
// the backend to act on.
//
// Flattening the conversation into one message keeps the two conversations
// apart: the backend's context holds only what the backend itself said as
// assistant messages, so it never mistakes the frontend's speech for its own.
//
// Only user and assistant text crosses: tool results, other content and
// messages in a service's own format are skipped, and say so at debug level.
// Carrying more is a question of how to render it.
//
// instruction is what the backend should do with the transcript, placed after
// it; empty uses TranscriptRequestInstruction. first reports whether this is
// the backend's first request in the conversation, which decides how the
// transcript is introduced: as the conversation so far, or as what has been
// said since the previous delegation.
func RenderTranscriptRequest(conversation []frames.Message, instruction string, first bool) string {
	if instruction == "" {
		instruction = TranscriptRequestInstruction
	}
	var lines []string
	if len(conversation) > 0 {
		if first {
			lines = append(lines, "Voice conversation so far:")
		} else {
			lines = append(lines, "Voice conversation since the previous delegation:")
		}
		for _, m := range conversation {
			if line, ok := transcriptLine(m); ok {
				lines = append(lines, line)
			}
		}
		lines = append(lines, "")
	}
	lines = append(lines, instruction)
	return strings.Join(lines, "\n")
}

// transcriptLine is a message's line in a transcript, and whether it has one:
// only user and assistant text does.
func transcriptLine(m frames.Message) (string, bool) {
	if m.IsLLMSpecific() {
		slog.Debug("skipping delegated message in a service's own format", "llm", m.LLM)
		return "", false
	}
	text := strings.TrimSpace(m.Text)
	if (m.Role != frames.RoleUser && m.Role != frames.RoleAssistant) || text == "" {
		slog.Debug("skipping delegated message with no transcript form", "role", m.Role)
		return "", false
	}
	return strings.ToUpper(string(m.Role)) + ": " + text, true
}

// BackendLLM is what a BackendWorker needs of the LLM service it runs. The LLM
// service base satisfies it.
type BackendLLM interface {
	ToolRegistrar
	// Events is the registry the service's frame events are raised on.
	Events() *events.Registry
	// HasQueuedFrame reports whether a frame matching is waiting to be handled.
	HasQueuedFrame(match func(frames.Frame) bool) bool
	// AppendSystemInstruction adds text to the model's system instruction.
	AppendSystemInstruction(instruction string)
	// SetUseProviderThinkingDefault has a model with no thinking configured run
	// at its provider's default.
	SetUseProviderThinkingDefault(on bool)
	// CancelFunctionCalls cancels every function call in flight.
	CancelFunctionCalls(ctx context.Context, reason string) []frames.ToolCall
	llm.LLMWithBackendRoleObjector
}

// BackendConfig configures a BackendWorker.
type BackendConfig struct {
	// Name is what other workers address this one by. Empty generates one;
	// give one when the backend runs in another process, where the frontend
	// addresses it by name.
	Name string
	// LLM is the backend LLM service.
	LLM BackendLLM
	// Context is the backend's conversation, typically carrying its tools. Nil
	// starts an empty one.
	Context *frames.LLMContext
	// Tools are tools the model may call, registered on the service as an LLM
	// worker registers them, beside any the context carries.
	Tools []frames.Tool
	// TransformOutput is called with each BackendOutput the model produces
	// before it is sent, to adjust its text or whether the user may hear it, or
	// to return nil and send nothing. Outputs the app sends itself are not
	// passed through it unless SendOutput asks.
	TransformOutput BackendOutputTransform
	// PairingInstruction is how the model works with the assistant, appended
	// to the LLM's system instruction: BackendPairingInstruction when empty. One
	// of its own has to say what that one says: that the assistant sends it
	// requests, that the user hears what it marks with SpokenMark and nothing
	// else, and how to take requests that arrive while it works.
	PairingInstruction string
	// AggregatorOptions configure the aggregator pair the worker builds. The
	// user aggregator takes external turn strategies unless these choose
	// others: the backend has no audio, so the default strategies, and any model
	// they load, have nothing to do here.
	AggregatorOptions []aggregators.Option
	// WorkerConfig is the rest of the pipeline worker's configuration.
	WorkerConfig pipeline.WorkerConfig
}

// attachedFrontend is the frontend attached to the worker, for as long as its
// attach job lives.
type attachedFrontend struct {
	jobID string
}

// BackendWorker is a worker that runs an LLM service as the backend a frontend
// delegates to.
//
// The worker owns the backend's conversation, a context plus the aggregator
// pair, so multi-step tool calling works as it does in any pipeline. A frontend
// attaches once, for as long as it lives, and from then on hears everything the
// backend produces; requests it sends are appended to that conversation as user
// messages and run the model. BackendSession is the caller side.
//
// The worker's LLM does the reasoning its frontend hands off, so it runs at its
// provider's default thinking unless its configuration sets thinking (see
// llm.Base.SetUseProviderThinkingDefault).
//
// Job contract:
//
//   - attach: no payload. Its first update is {"type": "attached",
//     "capabilities": {...}}; after that, a BackendOutput payload for each of
//     the model's turns and reasoning summaries and each output the app sends
//     with SendOutput, a BackendToolCall payload for each phase of each function
//     call the backend makes, {"type": "idle"} when the backend has nothing left
//     to do, and {"type": "error", "error": ...} when it could not go on. The job
//     lives until the requester cancels it, which stops the backend's work.
//     Refused while another frontend is attached.
//   - message: {"request": string}, appended as a user message; answers at once
//     with {"backend": "idle" | "working"}, what the backend was doing when the
//     message arrived. A message that arrives mid-run is taken up after the
//     current step, with everything that came before it in view.
//
// The model decides what the user hears: a message it begins with SpokenMark
// asks to be spoken, whether or not the turn also calls tools, and the mark is
// stripped; anything else it writes, reasoning summaries included, is a note
// that the frontend keeps but does not speak. A mark at the start of a later
// line splits the message into a note and a spoken part, since the model
// sometimes writes it that way. The backend's own system instruction says what
// it does and, in plain words, what the user should be told; the pairing
// instruction the worker appends is the only place the mark is named.
// TransformOutput can change the flag, or the text, or drop the output.
type BackendWorker struct {
	*ContextWorker

	llm             BackendLLM
	transformOutput BackendOutputTransform

	mu       sync.Mutex
	attached *attachedFrontend
	// Whether the backend is working is read off its pipeline: requests queued
	// but not yet taken up, LLM runs picked up but not yet ended, function calls
	// in flight, and context frames queued for the LLM. A request takes one or
	// more runs: the first for the request itself, then one per round of tool
	// results (the assistant aggregator pushes a context frame back to the LLM
	// after each). Runs are counted as the LLM picks them up and as their turns
	// end; an interruption cancels whatever was outstanding.
	requestsPending int
	runsRequested   int
	runsCompleted   int
	// idleAnnounced reports whether the idle update has been sent for the
	// current lull, so a call that settles after the last turn stopped does not
	// send it twice.
	idleAnnounced bool
	// requestAwaitingRun is the index in the context of a request appended
	// while the model was busy, as an append that does not run the model, or -1.
	// The run that follows the current step takes it up, and the model sees the
	// request alongside whatever that step produced. Running it separately
	// would run the model twice on the same context, the second time with
	// nothing new to react to. Cleared once a context frame that holds the
	// request reaches the LLM.
	requestAwaitingRun int

	// serial holds the worker's own handlers waiting to run, one at a time, in
	// the order they were asked for, and draining reports whether a goroutine is
	// running them. See runSerially.
	serial   []func()
	draining bool
}

// NewBackend builds a backend worker. It reports an error when the LLM service
// declines the backend role.
func NewBackend(cfg BackendConfig) (*BackendWorker, error) {
	if objection := cfg.LLM.LLMWithBackendRoleObjection(llm.RoleBackend); objection != "" {
		return nil, fmt.Errorf("%w: %s", ErrRoleDeclined, objection)
	}
	cfg.LLM.SetUseProviderThinkingDefault(true)

	external := aggregators.WithTurns(turns.Config{
		Strategies: turns.ExternalStrategies(turns.ExternalStrategiesConfig{}),
	})
	active := true
	b := &BackendWorker{
		llm:                cfg.LLM,
		transformOutput:    cfg.TransformOutput,
		requestAwaitingRun: -1,
	}
	b.ContextWorker = NewContext(ContextConfig{
		Name:              cfg.Name,
		LLM:               cfg.LLM,
		Tools:             cfg.Tools,
		Active:            &active,
		Context:           cfg.Context,
		AggregatorOptions: append([]aggregators.Option{external}, cfg.AggregatorOptions...),
		WorkerConfig:      cfg.WorkerConfig,
	})
	// The base reaches the hooks through the worker it is bound to, so this one
	// binds itself for its OnJobCancelled to be called.
	b.Bind(b)

	pairing := cfg.PairingInstruction
	if pairing == "" {
		pairing = BackendPairingInstruction
	}
	cfg.LLM.AppendSystemInstruction(pairing)

	b.HandleJob(AttachJobName, workers.JobOptions{}, b.attachFrontend)
	b.HandleJob(MessageJobName, workers.JobOptions{}, b.handleMessage)
	b.watchPipeline()
	return b, nil
}

// watchPipeline attaches the handlers that follow the backend's work as it
// passes through the pipeline.
func (b *BackendWorker) watchPipeline() {
	events.On(b.llm.Events(), processor.EventBeforeProcessFrame, b.onBeforeLLMFrame)
	events.On(b.UserAggregator().Events(), processor.EventBeforeProcessFrame, b.onBeforeUserAggregatorFrame)
	// The backend's function calls are relayed as they pass the assistant
	// aggregator, which every phase of a call reaches.
	events.On(b.AssistantAggregator().Events(), processor.EventBeforeProcessFrame, b.onFunctionCallFrame)
	events.On(b.AssistantAggregator().Events(), processor.EventAfterProcessFrame, b.onAfterAssistantAggregatorFrame)
	// These three are handled one at a time, in order, with the idle checks.
	events.On(b.AssistantAggregator().Events(), aggregators.EventAssistantTurnStopped,
		func(ctx context.Context, m aggregators.AssistantTurnStopped) {
			b.runSerially(func() { b.onAssistantTurnStopped(ctx, m) })
		})
	events.On(b.AssistantAggregator().Events(), aggregators.EventAssistantThought,
		func(ctx context.Context, m aggregators.AssistantThought) {
			b.runSerially(func() { b.onAssistantThought(ctx, m) })
		})
	events.On(&b.Registry, pipeline.EventPipelineError, func(ctx context.Context, ef *frames.ErrorFrame) {
		b.runSerially(func() { b.onPipelineError(ctx, ef) })
	})
}

// runSerially runs fn after the worker's own handlers asked for before it, and
// before those asked for after, one at a time.
//
// The worker reads whether the backend is working off counts its handlers keep,
// and the handlers that end a turn, report a thought or an error, and check for
// idle each assume no other is halfway through: a turn's end counts its run done
// and then sends what it said, and an idle check taken between the two would
// tell the frontend the backend is idle before it has heard the last of it.
// These handlers are raised from different goroutines, so they are put through
// here to run in turn.
func (b *BackendWorker) runSerially(fn func()) {
	b.mu.Lock()
	b.serial = append(b.serial, fn)
	start := !b.draining
	b.draining = true
	b.mu.Unlock()
	if start {
		go b.drainSerial()
	}
}

// drainSerial runs the queued handlers, one at a time, until none are left.
func (b *BackendWorker) drainSerial() {
	for {
		b.mu.Lock()
		if len(b.serial) == 0 {
			b.draining = false
			b.mu.Unlock()
			return
		}
		fn := b.serial[0]
		b.serial[0] = nil
		b.serial = b.serial[1:]
		b.mu.Unlock()
		fn()
	}
}

func (b *BackendWorker) onBeforeLLMFrame(_ context.Context, f frames.Frame) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch fr := f.(type) {
	case *frames.LLMContextFrame:
		b.runsRequested++
		if b.requestAwaitingRun >= 0 && len(fr.Context.Messages()) > b.requestAwaitingRun {
			b.requestAwaitingRun = -1
		}
	case *frames.InterruptionFrame:
		// The run in progress, if any, is canceled by this frame and ends as an
		// interrupted turn, which is not counted as completed; a run queued
		// behind it is dropped. A run that starts after the frame has passed is a
		// new one, so the counters are reset here, as the LLM sees the
		// interruption, and not later at the assistant aggregator, where a reset
		// could erase a run that had just started.
		b.runsRequested, b.runsCompleted = 0, 0
	}
}

func (b *BackendWorker) onBeforeUserAggregatorFrame(ctx context.Context, f frames.Frame) {
	fr, ok := f.(*frames.LLMMessagesAppendFrame)
	if !ok {
		return
	}
	b.mu.Lock()
	if b.requestsPending == 0 {
		b.mu.Unlock()
		return
	}
	b.requestsPending--
	held := !fr.RunLLM
	if held {
		b.requestAwaitingRun = len(b.Context().Messages())
	}
	b.mu.Unlock()
	if held {
		// The synchronous call the request was held for may have settled by
		// the time the request lands in the context, as when the model canceled
		// it with its cancel tool meanwhile, so check again whether the request
		// can run now.
		b.runAwaitingRequest(ctx)
	}
}

func (b *BackendWorker) onAfterAssistantAggregatorFrame(ctx context.Context, f frames.Frame) {
	// A result that asks for no run may have settled the last synchronous call a
	// waiting request was held for, so the request runs now; one that asks for a
	// run brings the run itself.
	switch fr := f.(type) {
	case *frames.FunctionCallResultFrame:
		if !resultDeclinesRun(fr) {
			return
		}
	case *frames.FunctionCallCancelFrame:
	default:
		return
	}
	b.runAwaitingRequest(ctx)
	// A call that settles (its result or cancellation arrives) after the last
	// turn stopped may have been the last thing the backend had in flight, so
	// the idle update is sent from here as well as from the end of a turn.
	b.announceIdleIfDone(ctx)
}

// resultDeclinesRun reports whether a result says outright that it asks for no
// run: its properties, or failing them the frame, set the run to false.
func resultDeclinesRun(f *frames.FunctionCallResultFrame) bool {
	if f.Properties != nil && f.Properties.RunLLM != nil {
		return !*f.Properties.RunLLM
	}
	return f.RunLLM != nil && !*f.RunLLM
}

// Working reports whether the backend has work in hand: a request or run
// outstanding, or a call in flight.
func (b *BackendWorker) Working() bool {
	b.mu.Lock()
	pending := b.requestsPending > 0
	b.mu.Unlock()
	return pending || b.modelBusy() || b.AssistantAggregator().HasFunctionCallsInProgress()
}

// modelBusy reports whether a run is in progress or queued for the model.
func (b *BackendWorker) modelBusy() bool {
	b.mu.Lock()
	running := b.runsRequested > b.runsCompleted
	b.mu.Unlock()
	return running || b.llm.HasQueuedFrame(func(f frames.Frame) bool {
		_, ok := f.(*frames.LLMContextFrame)
		return ok
	})
}

// synchronousCallInFlight reports whether a synchronous call is in flight,
// whose result will bring the next run.
func (b *BackendWorker) synchronousCallInFlight() bool {
	return b.AssistantAggregator().HasSynchronousFunctionCallsInProgress()
}

// BackendCapabilities is what an attached frontend can count on: all of it, for
// a backend that is an LLM.
func (b *BackendWorker) BackendCapabilities() map[string]bool {
	return map[string]bool{"steering": true, "progress": true}
}

// attachFrontend attaches a frontend, streaming everything the backend
// produces until the job is canceled.
func (b *BackendWorker) attachFrontend(ctx context.Context, m *bus.JobRequestMessage) {
	b.mu.Lock()
	if b.attached != nil {
		b.mu.Unlock()
		b.refuse(ctx, m.JobID, "a frontend is already attached")
		return
	}
	attached := &attachedFrontend{jobID: m.JobID}
	b.attached = attached
	b.mu.Unlock()

	if err := b.SendJobUpdate(ctx, m.JobID, map[string]any{
		payloadType: AttachedUpdateType, "capabilities": b.BackendCapabilities(),
	}, false); err != nil {
		slog.WarnContext(ctx, "could not open the attach stream", "worker", b.Name(), "err", err)
	}

	// The attach job ends only when the frontend cancels it, which is how the
	// frontend detaches.
	<-ctx.Done()
	b.mu.Lock()
	if b.attached == attached {
		b.attached = nil
	}
	b.mu.Unlock()
}

// handleMessage puts a message to the backend, and says what it was doing when
// the message arrived.
func (b *BackendWorker) handleMessage(ctx context.Context, m *bus.JobRequestMessage) {
	b.mu.Lock()
	attached := b.attached != nil
	b.mu.Unlock()
	if !attached {
		b.refuse(ctx, m.JobID, "no frontend is attached; open an attach job first")
		return
	}
	request := strings.TrimSpace(payloadString(m.Payload, "request"))
	if request == "" {
		b.refuse(ctx, m.JobID, "the message has no request")
		return
	}
	status := "idle"
	if b.Working() {
		status = "working"
	}
	b.queueRequest(ctx, request)
	response := map[string]any{"backend": status}
	if err := b.SendJobResponse(ctx, m.JobID, response, workers.JobResponseOptions{}); err != nil {
		slog.WarnContext(ctx, "could not answer a message", "worker", b.Name(), "err", err)
	}
}

func (b *BackendWorker) refuse(ctx context.Context, jobID, reason string) {
	slog.WarnContext(ctx, "refusing job", "worker", b.Name(), "job", jobID, "reason", reason)
	if err := b.SendJobResponse(ctx, jobID, map[string]any{payloadError: reason}, workers.JobResponseOptions{
		Status: jobcontext.JobError,
	}); err != nil {
		slog.WarnContext(ctx, "could not refuse a job", "worker", b.Name(), "job", jobID, "err", err)
	}
}

// queueRequest appends a request to the backend's conversation, and runs the
// model on it if it is idle.
//
// While a run is in progress or queued, or a synchronous call is in flight, the
// request asks for no run of its own: the run that follows the current step
// takes it up (see requestAwaitingRun). A run while a synchronous call is in
// flight would see the call without its result, which the adapter cannot send
// as it is.
func (b *BackendWorker) queueRequest(ctx context.Context, request string) {
	b.mu.Lock()
	b.requestsPending++
	b.idleAnnounced = false
	b.mu.Unlock()
	busy, syncCall := b.modelBusy(), b.synchronousCallInFlight()
	runLLM := !busy && !syncCall
	if runLLM {
		slog.DebugContext(ctx, "running the model on the request", "worker", b.Name())
	} else {
		slog.DebugContext(ctx, "holding the request for the current step", "worker", b.Name(),
			"model_busy", busy, "synchronous_call_in_flight", syncCall)
	}
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: request}})
	f.RunLLM = runLLM
	b.QueueFrame(ctx, f)
}

// runAwaitingRequest runs the model on a request that was waiting for the
// current step, if nothing else will.
func (b *BackendWorker) runAwaitingRequest(ctx context.Context) {
	b.mu.Lock()
	waiting := b.requestAwaitingRun >= 0
	b.mu.Unlock()
	if waiting && !b.modelBusy() && !b.synchronousCallInFlight() {
		slog.DebugContext(ctx, "running the model on the request that was held", "worker", b.Name())
		b.QueueFrame(ctx, frames.NewLLMRunFrame())
	}
}

// stopWork interrupts the pipeline and cancels every function call in flight.
//
// An interruption stops the run in progress and the calls that opt into
// cancellation on interruption; the asynchronous ones, which an interruption
// leaves running by design, are canceled on their own. A request not yet taken
// up is dropped with the interruption, and one held for the current step is let
// go of, so that neither runs the model on work the frontend has abandoned: the
// cancellations that follow would otherwise run a held request as they settle.
func (b *BackendWorker) stopWork(ctx context.Context, reason string) {
	b.mu.Lock()
	b.requestsPending = 0
	b.requestAwaitingRun = -1
	b.mu.Unlock()
	b.QueueFrame(ctx, frames.NewInterruptionFrame())
	b.llm.CancelFunctionCalls(ctx, reason)
}

// SendOutput sends one output of the app's own to the frontend.
//
// It reaches the frontend as the model's outputs do, but as given unless asked:
// TransformOutput is not applied to it unless applyTransform is set, so an app
// can silence the model's outputs with a transform that drops them and still
// send outputs of its own, such as from a tool the model calls to talk to the
// user. With no frontend attached there is nowhere for it to go, and it is
// dropped with a warning.
func (b *BackendWorker) SendOutput(ctx context.Context, output BackendOutput, applyTransform bool) {
	b.mu.Lock()
	attached := b.attached != nil
	b.mu.Unlock()
	if !attached {
		slog.WarnContext(ctx, "no frontend to send output to", "worker", b.Name())
		return
	}
	if err := b.emit(ctx, output, applyTransform); err != nil {
		slog.ErrorContext(ctx, "transform_output failed", "worker", b.Name(), "err", err)
	}
}

func (b *BackendWorker) onAssistantTurnStopped(ctx context.Context, message aggregators.AssistantTurnStopped) {
	if message.Interrupted {
		// Text from an interrupted turn is not sent, and the run counters were
		// reset as the LLM saw the InterruptionFrame.
		return
	}
	b.mu.Lock()
	b.runsCompleted++
	attached := b.attached != nil
	b.mu.Unlock()
	if !attached {
		return
	}
	note, spoken := splitSpoken(strings.TrimSpace(message.Content))
	err := func() error {
		if note != "" {
			if err := b.emit(ctx, BackendOutput{Text: note}, true); err != nil {
				return err
			}
		}
		if spoken != "" {
			return b.emit(ctx, NewBackendOutput(spoken), true)
		}
		return nil
	}()
	if err != nil {
		slog.ErrorContext(ctx, "transform_output failed", "worker", b.Name(), "err", err)
		b.sendUpdate(ctx, map[string]any{payloadType: ErrorUpdateType, payloadError: err.Error()})
	}
	b.runAwaitingRequest(ctx)
	b.announceIdleIfDone(ctx)
}

// announceIdleIfDone sends the idle update once the backend has nothing left to
// do, once per lull.
//
// The check runs with the worker's other handlers (see runSerially), and reads
// the backend once the pipeline has settled. Whether the backend is working is
// read off frames on their way through the pipeline, and the handlers asking
// for this check run beside the processors moving those frames rather than
// between them: a check made at once could catch a result handled but its run
// not yet asked for, or a run asked for but not yet picked up, and find the
// backend idle with its next step already under way. Letting the pipeline
// settle first leaves nothing half done in sight, which is the state every
// check is meant to read.
//
// The check belongs to the frontend attached when it was asked for. Waiting for
// the pipeline to settle can outlast that frontend detaching and another
// attaching, and the lull it was asked for is not the new frontend's to hear
// about: a backend a frontend attaches to while idle says nothing until it has
// worked for it.
func (b *BackendWorker) announceIdleIfDone(ctx context.Context) {
	ctx = context.WithoutCancel(ctx)
	b.mu.Lock()
	attached := b.attached
	b.mu.Unlock()
	if attached == nil {
		return
	}
	b.runSerially(func() {
		if err := b.Flush(ctx); err != nil {
			// A pipeline that will not settle is either stopping or still busy
			// with a run, whose end asks for another check.
			slog.DebugContext(ctx, "the pipeline did not settle, skipping the idle check",
				"worker", b.Name(), "err", err)
			return
		}
		b.checkIdle(ctx, attached)
	})
}

// checkIdle sends the idle update to the attached frontend if it is still the
// one the check was asked for, the backend has nothing left to do and it has
// not said so already for this lull.
func (b *BackendWorker) checkIdle(ctx context.Context, attached *attachedFrontend) {
	b.mu.Lock()
	skip := b.attached != attached || b.idleAnnounced
	b.mu.Unlock()
	if skip || b.Working() {
		return
	}
	b.mu.Lock()
	b.idleAnnounced = true
	b.mu.Unlock()
	b.sendUpdate(ctx, map[string]any{payloadType: IdleUpdateType})
}

// onFunctionCallFrame relays a phase of one of the backend's own function calls
// as a job update.
func (b *BackendWorker) onFunctionCallFrame(ctx context.Context, f frames.Frame) {
	var calls []BackendToolCall
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		for _, call := range fr.Calls {
			calls = append(calls, BackendToolCall{
				Phase: BackendToolCallStarted, FunctionName: call.Name, ToolCallID: call.ID, IsFinal: true,
			})
		}
	case *frames.FunctionCallInProgressFrame:
		calls = append(calls, BackendToolCall{
			Phase: BackendToolCallInProgress, FunctionName: fr.ToolName, ToolCallID: fr.ToolCallID,
			Arguments: fr.Args, IsFinal: true,
		})
	case *frames.FunctionCallResultFrame:
		calls = append(calls, BackendToolCall{
			Phase: BackendToolCallResult, FunctionName: fr.ToolName, ToolCallID: fr.ToolCallID,
			Arguments: fr.Args, Result: fr.Result, IsFinal: fr.Properties.Final(),
		})
	case *frames.FunctionCallCancelFrame:
		calls = append(calls, BackendToolCall{
			Phase: BackendToolCallCanceled, FunctionName: fr.ToolName, ToolCallID: fr.ToolCallID, IsFinal: true,
		})
	}
	for _, call := range calls {
		b.sendUpdate(ctx, call.ToPayload())
	}
}

// OnJobCancelled stops the model when the frontend that was attached is gone.
//
// The attach job's handler is already called off by the time this runs, and has
// let go of the attachment; the work in flight stops too, so none of it is
// reported after the frontend has let go of it.
func (b *BackendWorker) OnJobCancelled(ctx context.Context, m *bus.JobCancelMessage) {
	b.ContextWorker.OnJobCancelled(ctx, m)
	if request, ok := b.ActiveJobs()[m.JobID]; ok && request.JobName == AttachJobName {
		b.stopWork(ctx, "frontend detached")
	}
}

// onPipelineError tells the frontend the backend cannot go on with what it was
// doing.
//
// A tool handler that fails is the exception. The LLM service reports that as
// an application error, settles the call with an error result and carries on,
// so the model still has something coming.
func (b *BackendWorker) onPipelineError(ctx context.Context, ef *frames.ErrorFrame) {
	if ef.Category == errs.Application {
		return
	}
	b.sendUpdate(ctx, map[string]any{payloadType: ErrorUpdateType, payloadError: ef.Error})
}

func (b *BackendWorker) onAssistantThought(ctx context.Context, message aggregators.AssistantThought) {
	text := strings.TrimSpace(message.Content)
	b.mu.Lock()
	attached := b.attached != nil
	b.mu.Unlock()
	if text == "" || !attached {
		return
	}
	if err := b.emit(ctx, BackendOutput{Text: text, IsThought: true}, true); err != nil {
		slog.ErrorContext(ctx, "transform_output failed", "worker", b.Name(), "err", err)
	}
}

// emit sends one output as a job update, after the transform when asked.
func (b *BackendWorker) emit(ctx context.Context, output BackendOutput, applyTransform bool) error {
	if applyTransform && b.transformOutput != nil {
		shaped, err := b.transformOutput(ctx, output)
		if err != nil {
			return err
		}
		if shaped == nil {
			return nil
		}
		output = *shaped
	}
	b.sendUpdate(ctx, output.ToPayload())
	return nil
}

// sendUpdate sends a job update to the attached frontend, if there is one.
func (b *BackendWorker) sendUpdate(ctx context.Context, payload map[string]any) {
	b.mu.Lock()
	attached := b.attached
	b.mu.Unlock()
	if attached == nil {
		return
	}
	if err := b.SendJobUpdate(ctx, attached.jobID, payload, false); err != nil {
		slog.DebugContext(ctx, "could not send a job update", "worker", b.Name(), "err", err)
	}
}

// payloadString reads a payload field as a string, "" when it is missing or
// not one.
func payloadString(payload map[string]any, key string) string {
	switch v := payload[key].(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

// truthy reads a flag from a payload the way a value that crossed a bus may
// carry it: a bool, a number, or a string.
func truthy(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case nil:
		return false
	case string:
		return x != ""
	case float64:
		return x != 0
	case int:
		return x != 0
	default:
		return true
	}
}
