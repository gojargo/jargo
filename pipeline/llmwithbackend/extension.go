package llmwithbackend

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// BackendPairingLLMExtension pairs the frontend of an LLMWithBackend with its
// backend.
//
// It is the frontend's half of the pairing: the instruction that says when and
// how to delegate and what to do with what the backend sends back, and the
// tools delegation goes through, delegate first of all. The composite builds
// both from its BackendConnector and installs the extension on the frontend;
// the connector runs the session the tools feed.
type BackendPairingLLMExtension struct {
	llm.BaseLLMExtension

	instruction string
	tools       []llm.LLMExtensionTool
}

// NewBackendPairingLLMExtension builds the pairing extension. instruction is
// what the pairing adds to the frontend's system instruction, or "" for
// nothing. tools are the tools delegation goes through, each carrying its
// handler. They run as asynchronous tools: a delegation returns at once, and
// the backend's answer arrives later.
func NewBackendPairingLLMExtension(instruction string, tools []frames.Tool) *BackendPairingLLMExtension {
	async := false
	e := &BackendPairingLLMExtension{instruction: instruction}
	for _, tool := range tools {
		// A delegation is announced when it starts, as a call of the
		// application's own tools is: the application may speak on it, and the
		// pipeline counts it among the calls in progress.
		e.tools = append(e.tools, llm.LLMExtensionTool{
			Schema:               tool,
			CancelOnInterruption: &async,
			AnnounceCallStart:    true,
		})
	}
	return e
}

// SystemInstruction implements llm.LLMExtension: the pairing instruction.
func (e *BackendPairingLLMExtension) SystemInstruction() string { return e.instruction }

// BuiltinTools implements llm.LLMExtension: the delegation tools, whatever is
// registered.
func (e *BackendPairingLLMExtension) BuiltinTools(map[string]llm.FunctionCallRegistryItem) []llm.LLMExtensionTool {
	return e.tools
}
