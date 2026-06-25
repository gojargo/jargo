package frames

import "fmt"

// FunctionCallsStartedFrame announces that the model requested one or more tool
// calls in the current assistant turn. PreambleText is any text the model spoke
// before the calls. The assistant context aggregator writes the assistant turn
// (preamble text plus the tool-use blocks) from this frame. It is a control
// frame so it stays ordered with the response's text frames.
type FunctionCallsStartedFrame struct {
	BaseControlFrame
	// PreambleText is text the model produced alongside the calls, if any.
	PreambleText string
	// Calls are the tool calls the model requested this turn.
	Calls []ToolCall
}

// NewFunctionCallsStartedFrame builds a FunctionCallsStartedFrame.
func NewFunctionCallsStartedFrame(preamble string, calls []ToolCall) *FunctionCallsStartedFrame {
	return &FunctionCallsStartedFrame{
		BaseControlFrame: NewBaseControlFrame("FunctionCallsStartedFrame"),
		PreambleText:     preamble,
		Calls:            calls,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallsStartedFrame) String() string {
	return fmt.Sprintf("%s(calls: %d)", f.Name(), len(f.Calls))
}

// FunctionCallInProgressFrame reports that a specific tool call has started
// executing. It is useful for observability and for clients that surface an
// "in progress" indication. It is a control frame.
type FunctionCallInProgressFrame struct {
	BaseControlFrame
	// ToolCallID is the id of the call that is executing.
	ToolCallID string
	// ToolName is the tool's name.
	ToolName string
}

// NewFunctionCallInProgressFrame builds a FunctionCallInProgressFrame.
func NewFunctionCallInProgressFrame(toolCallID, name string) *FunctionCallInProgressFrame {
	return &FunctionCallInProgressFrame{
		BaseControlFrame: NewBaseControlFrame("FunctionCallInProgressFrame"),
		ToolCallID:       toolCallID,
		ToolName:         name,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallInProgressFrame) String() string {
	return fmt.Sprintf("%s(%s)", f.Name(), f.ToolName)
}

// FunctionCallResultFrame carries the result of one tool call. The assistant
// context aggregator collects results into a single tool-result message. It is
// a control frame.
type FunctionCallResultFrame struct {
	BaseControlFrame
	// ToolCallID pairs the result to its call.
	ToolCallID string
	// ToolName is the tool's name.
	ToolName string
	// Result is the tool-result content.
	Result string
	// IsError reports whether the tool failed.
	IsError bool
}

// NewFunctionCallResultFrame builds a FunctionCallResultFrame.
func NewFunctionCallResultFrame(toolCallID, name, result string, isError bool) *FunctionCallResultFrame {
	return &FunctionCallResultFrame{
		BaseControlFrame: NewBaseControlFrame("FunctionCallResultFrame"),
		ToolCallID:       toolCallID,
		ToolName:         name,
		Result:           result,
		IsError:          isError,
	}
}

// String implements fmt.Stringer.
func (f *FunctionCallResultFrame) String() string {
	return fmt.Sprintf("%s(%s, error: %t)", f.Name(), f.ToolName, f.IsError)
}

// Compile-time interface checks.
var (
	_ ControlFrame = (*FunctionCallsStartedFrame)(nil)
	_ ControlFrame = (*FunctionCallInProgressFrame)(nil)
	_ ControlFrame = (*FunctionCallResultFrame)(nil)
)
