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
	// interrupted. A call registered with it false is asynchronous: the model
	// carries on without waiting, and the result arrives later as a developer
	// message (see AsyncToolMessage).
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
	// asynchronous call (one registered with CancelOnInterruption false).
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

// Compile-time interface checks.
var (
	_ SystemFrame     = (*FunctionCallsStartedFrame)(nil)
	_ ControlFrame    = (*FunctionCallInProgressFrame)(nil)
	_ Uninterruptible = (*FunctionCallInProgressFrame)(nil)
	_ DataFrame       = (*FunctionCallResultFrame)(nil)
	_ Uninterruptible = (*FunctionCallResultFrame)(nil)
	_ SystemFrame     = (*FunctionCallCancelFrame)(nil)
)
