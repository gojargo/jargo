package frames

import (
	"context"
	"encoding/json"
	"fmt"
)

// FunctionCallsStartedFrame announces that the model requested one or more tool
// calls in the current assistant turn. The assistant context aggregator records
// the calls as awaiting results; it writes nothing to the context, because each
// call's own FunctionCallInProgressFrame writes the assistant tool-call message
// and its placeholder result. It is a system frame so it reaches the aggregator
// ahead of the queued frames of the turn it belongs to.
type FunctionCallsStartedFrame struct {
	BaseSystemFrame
	// Calls are the tool calls the model requested this turn.
	Calls []ToolCall
}

// NewFunctionCallsStartedFrame builds a FunctionCallsStartedFrame.
func NewFunctionCallsStartedFrame(calls []ToolCall) *FunctionCallsStartedFrame {
	return &FunctionCallsStartedFrame{
		BaseSystemFrame: NewBaseSystemFrame("FunctionCallsStartedFrame"),
		Calls:           calls,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallsStartedFrame) String() string {
	return fmt.Sprintf("%s(calls: %d)", f.Name(), len(f.Calls))
}

// FunctionCallInProgressFrame reports that a specific tool call has started
// executing. The assistant context aggregator writes the assistant tool-call
// message and a placeholder tool result from it, which is what keeps the pair
// balanced for every inference that follows. It is uninterruptible because the
// context must be updated even when the turn is cut off.
type FunctionCallInProgressFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// ToolCallID is the id of the call that is executing.
	ToolCallID string
	// ToolName is the tool's name.
	ToolName string
	// Args is the raw JSON arguments the model produced for the call.
	Args json.RawMessage
	// CancelOnInterruption reports whether the call is canceled when the turn is
	// interrupted. A call registered with it false is asynchronous: the
	// conversation is not held while it runs, and a result that arrives after the
	// conversation has moved on is delivered as a developer message (see
	// AsyncToolMessage). A result that arrives before then settles in place, like
	// the result of a call the model waited on.
	CancelOnInterruption bool
	// GroupID is shared by every call the model requested in one response, so the
	// aggregator can tell when the last of them completes.
	GroupID string
}

// NewFunctionCallInProgressFrame builds a FunctionCallInProgressFrame.
func NewFunctionCallInProgressFrame(
	toolCallID, name string, args json.RawMessage, cancelOnInterruption bool, groupID string,
) *FunctionCallInProgressFrame {
	return &FunctionCallInProgressFrame{
		BaseControlFrame:     NewBaseControlFrame("FunctionCallInProgressFrame"),
		ToolCallID:           toolCallID,
		ToolName:             name,
		Args:                 args,
		CancelOnInterruption: cancelOnInterruption,
		GroupID:              groupID,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallInProgressFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// FunctionCallResultProperties tunes how a tool result is applied. A handler
// passes it to its result callback; a nil value means the defaults, which are to
// treat the result as final and to let the aggregator decide whether to
// re-generate.
type FunctionCallResultProperties struct {
	// RunLLM forces whether generation re-runs after this result. Nil leaves the
	// decision to the aggregator, which re-runs once the last call in the group
	// completes.
	RunLLM *bool
	// OnContextUpdated runs once the result has been written to the context. It
	// runs on its own goroutine so it never blocks the pipeline.
	OnContextUpdated func(ctx context.Context) error
	// IsFinal reports whether this is the call's final result. Nil means final.
	// A false value marks an intermediate update, which is only meaningful for an
	// asynchronous call (one registered with CancelOnInterruption false). A
	// speech-to-speech service takes one output per call, so it puts an
	// intermediate result to its model some other way (a conversation item, a
	// text event, a response that keeps the call open), and RunLLM says whether
	// the model should answer it or just take it in. A service that cannot do
	// that at all says so with AcceptsIntermediateFunctionCallResults and drops
	// intermediate results with a warning.
	IsFinal *bool
}

// Final reports whether the properties describe a final result. Nil properties,
// and properties that leave IsFinal unset, both mean final.
func (p *FunctionCallResultProperties) Final() bool {
	return p == nil || p.IsFinal == nil || *p.IsFinal
}

// FunctionCallResultFrame carries the result of one tool call. The assistant
// context aggregator updates the call's placeholder result in place, so the
// tool-use and tool-result pair stays adjacent and balanced. It is
// uninterruptible because a result that was produced must always reach the
// context.
type FunctionCallResultFrame struct {
	BaseDataFrame
	UninterruptibleMixin
	// ToolCallID pairs the result to its call.
	ToolCallID string
	// ToolName is the tool's name.
	ToolName string
	// Args is the raw JSON arguments the call was made with.
	Args json.RawMessage
	// Result is the tool-result content.
	Result string
	// Error is what went wrong, on a call whose handler failed, and "" on one
	// that returned. Result still carries what the model is told, which names
	// the function and nothing else: the failure itself is for whoever is
	// watching the conversation rather than for the model.
	Error string
	// RunLLM forces whether generation re-runs after this result. Nil leaves the
	// decision to the aggregator. Properties, when it sets RunLLM, wins over it.
	RunLLM *bool
	// Properties tunes how the result is applied; nil means the defaults.
	Properties *FunctionCallResultProperties
}

// NewFunctionCallResultFrame builds a FunctionCallResultFrame.
func NewFunctionCallResultFrame(toolCallID, name string, args json.RawMessage, result string) *FunctionCallResultFrame {
	return &FunctionCallResultFrame{
		BaseDataFrame: NewBaseDataFrame("FunctionCallResultFrame"),
		ToolCallID:    toolCallID,
		ToolName:      name,
		Args:          args,
		Result:        result,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallResultFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// FunctionCallCancelFrame reports that a tool call was canceled, which happens
// when a barge-in interrupts a call registered to be canceled on interruption.
// The assistant context aggregator marks the call's placeholder result canceled
// so the pair stays balanced. It is a system frame because cancellation must
// overtake the queued frames of the turn being abandoned.
type FunctionCallCancelFrame struct {
	BaseSystemFrame
	// ToolCallID identifies the canceled call.
	ToolCallID string
	// ToolName is the tool's name.
	ToolName string
	// RunLLM asks for inference once the cancellation is settled in the
	// conversation. Only a call canceled by its own deadline sets it: an
	// interruption must not trigger inference, and a cancellation the model
	// asked for already runs inference through the result of the tool that
	// requested it.
	RunLLM bool
}

// NewFunctionCallCancelFrame builds a FunctionCallCancelFrame.
func NewFunctionCallCancelFrame(toolCallID, name string) *FunctionCallCancelFrame {
	return &FunctionCallCancelFrame{
		BaseSystemFrame: NewBaseSystemFrame("FunctionCallCancelFrame"),
		ToolCallID:      toolCallID,
		ToolName:        name,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallCancelFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// ExternalFunctionCall is a function call made outside this pipeline, as
// announced to it: what an ExternalFunctionCallsStartedFrame carries, one per
// call.
type ExternalFunctionCall struct {
	// ToolName is the name of the function called.
	ToolName string
	// ToolCallID identifies the call.
	ToolCallID string
}

// ExternalFunctionCallsStartedFrame reports function calls made outside this
// pipeline: FunctionCallsStartedFrame's counterpart.
//
// Nothing in the pipeline acts on one; observers report it as they report the
// pipeline's own. The calls ran elsewhere, in a backend worker's pipeline on
// behalf of a tool here say, and belong to that pipeline's conversation, not this
// one. Like its counterpart it carries every call announced together, which may
// be one. Each call's later phases come as an ExternalFunctionCallInProgressFrame,
// an ExternalFunctionCallResultFrame and an ExternalFunctionCallCancelFrame.
type ExternalFunctionCallsStartedFrame struct {
	BaseSystemFrame
	// Calls are the calls made.
	Calls []ExternalFunctionCall
}

// NewExternalFunctionCallsStartedFrame builds an
// ExternalFunctionCallsStartedFrame.
func NewExternalFunctionCallsStartedFrame(calls []ExternalFunctionCall) *ExternalFunctionCallsStartedFrame {
	return &ExternalFunctionCallsStartedFrame{
		BaseSystemFrame: NewBaseSystemFrame("ExternalFunctionCallsStartedFrame"),
		Calls:           calls,
	}
}

// String implements fmt.Stringer.
func (f *ExternalFunctionCallsStartedFrame) String() string {
	return fmt.Sprintf("%s(calls: %d)", f.Name(), len(f.Calls))
}

// ExternalFunctionCallInProgressFrame reports an external function call is
// running: FunctionCallInProgressFrame's counterpart.
//
// Nothing in the pipeline acts on one; observers report it as they report the
// pipeline's own. The call ran elsewhere and belongs to that pipeline's
// conversation, not this one.
type ExternalFunctionCallInProgressFrame struct {
	BaseSystemFrame
	// ToolCallID identifies the call.
	ToolCallID string
	// ToolName is the name of the function called.
	ToolName string
	// Args is the raw JSON arguments passed to the function.
	Args json.RawMessage
}

// NewExternalFunctionCallInProgressFrame builds an
// ExternalFunctionCallInProgressFrame.
func NewExternalFunctionCallInProgressFrame(
	toolCallID, name string, args json.RawMessage,
) *ExternalFunctionCallInProgressFrame {
	return &ExternalFunctionCallInProgressFrame{
		BaseSystemFrame: NewBaseSystemFrame("ExternalFunctionCallInProgressFrame"),
		ToolCallID:      toolCallID,
		ToolName:        name,
		Args:            args,
	}
}

// String implements fmt.Stringer.
func (f *ExternalFunctionCallInProgressFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// ExternalFunctionCallResultFrame reports an external function call produced a
// result: FunctionCallResultFrame's counterpart.
//
// Nothing in the pipeline acts on one; observers report it as they report the
// pipeline's own.
type ExternalFunctionCallResultFrame struct {
	BaseSystemFrame
	// ToolCallID identifies the call.
	ToolCallID string
	// ToolName is the name of the function called.
	ToolName string
	// Args is the raw JSON arguments passed to the function.
	Args json.RawMessage
	// Result is the result.
	Result string
	// IsFinal reports whether this result completes the call, rather than being
	// one of a stream of intermediate results before the final one.
	IsFinal bool
}

// NewExternalFunctionCallResultFrame builds an ExternalFunctionCallResultFrame
// for a final result.
func NewExternalFunctionCallResultFrame(
	toolCallID, name string, args json.RawMessage, result string,
) *ExternalFunctionCallResultFrame {
	return &ExternalFunctionCallResultFrame{
		BaseSystemFrame: NewBaseSystemFrame("ExternalFunctionCallResultFrame"),
		ToolCallID:      toolCallID,
		ToolName:        name,
		Args:            args,
		Result:          result,
		IsFinal:         true,
	}
}

// String implements fmt.Stringer.
func (f *ExternalFunctionCallResultFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// ExternalFunctionCallCancelFrame reports an external function call was
// canceled: FunctionCallCancelFrame's counterpart.
//
// Nothing in the pipeline acts on one; observers report it as they report the
// pipeline's own.
type ExternalFunctionCallCancelFrame struct {
	BaseSystemFrame
	// ToolCallID identifies the call.
	ToolCallID string
	// ToolName is the name of the function called.
	ToolName string
}

// NewExternalFunctionCallCancelFrame builds an ExternalFunctionCallCancelFrame.
func NewExternalFunctionCallCancelFrame(toolCallID, name string) *ExternalFunctionCallCancelFrame {
	return &ExternalFunctionCallCancelFrame{
		BaseSystemFrame: NewBaseSystemFrame("ExternalFunctionCallCancelFrame"),
		ToolCallID:      toolCallID,
		ToolName:        name,
	}
}

// String implements fmt.Stringer.
func (f *ExternalFunctionCallCancelFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// Compile-time interface checks.
var (
	_ SystemFrame     = (*FunctionCallsStartedFrame)(nil)
	_ ControlFrame    = (*FunctionCallInProgressFrame)(nil)
	_ Uninterruptible = (*FunctionCallInProgressFrame)(nil)
	_ DataFrame       = (*FunctionCallResultFrame)(nil)
	_ Uninterruptible = (*FunctionCallResultFrame)(nil)
	_ SystemFrame     = (*FunctionCallCancelFrame)(nil)
	_ SystemFrame     = (*ExternalFunctionCallsStartedFrame)(nil)
	_ SystemFrame     = (*ExternalFunctionCallInProgressFrame)(nil)
	_ SystemFrame     = (*ExternalFunctionCallResultFrame)(nil)
	_ SystemFrame     = (*ExternalFunctionCallCancelFrame)(nil)
)
