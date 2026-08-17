package llm

import (
	"log/slog"
	"strings"
)

// The system instruction a service sends is composed rather than stored whole:
// the prompt the application gave is one part of it, and the framework adds
// others as features that need the model told about them are turned on. This
// file holds that composition.
//
// The base prompt is the single source of truth. Every rebuild starts from it,
// so composing twice produces the same instruction rather than compounding the
// additions.

// SystemInstruction returns the system instruction this service sends: the base
// prompt the application set, followed by everything composed onto it.
//
// A provider reads it when it builds a request, and passes it to its adapter as
// the instruction for the call.
func (b *Base) SystemInstruction() string {
	b.systemMu.Lock()
	defer b.systemMu.Unlock()
	return b.systemInstruction
}

// SetSystemInstruction replaces the base system prompt and rebuilds the
// composed instruction from it. It is what an LLMUpdateSettingsFrame carrying a
// system instruction ends up calling.
func (b *Base) SetSystemInstruction(instruction string) {
	b.systemMu.Lock()
	b.baseSystemInstruction = instruction
	b.systemMu.Unlock()
	b.composeSystemInstruction()
}

// AppendSystemInstruction adds durable text to the end of the system
// instruction, keeping the application's own prompt.
//
// The text is composed onto the instruction rather than written into the
// conversation, so it survives a reset of the context messages and is sent on
// every generation. It is for a framework component that owns an LLM and has to
// tell the model something the application did not: how to use a wire format it
// is being asked to produce, say. Appended text is joined after the base prompt
// and before the instructions the built-in tools contribute.
func (b *Base) AppendSystemInstruction(instruction string) {
	b.systemMu.Lock()
	b.appendedSystemInstructions = append(b.appendedSystemInstructions, instruction)
	b.systemMu.Unlock()
	b.composeSystemInstruction()
}

// composeSystemInstruction rebuilds the instruction from the base prompt and
// every addition currently in force. It always rebuilds from the base, so it is
// safe to call repeatedly.
func (b *Base) composeSystemInstruction() {
	b.systemMu.Lock()
	parts := make([]string, 0, len(b.appendedSystemInstructions)+1)
	if b.baseSystemInstruction != "" {
		parts = append(parts, b.baseSystemInstruction)
	}
	for _, p := range b.appendedSystemInstructions {
		if p != "" {
			parts = append(parts, p)
		}
	}
	b.systemMu.Unlock()

	// The marker protocol is composed on rather than appended, so turning the
	// gating off takes it back out again.
	b.turnCompletion.mu.Lock()
	if b.turnCompletion.enabled {
		parts = append(parts, b.turnCompletion.config.CompletionInstructions())
	}
	b.turnCompletion.mu.Unlock()

	// Likewise the guidance on stopping work early: it comes and goes with the
	// cancel tools it describes, so a session with none never carries it.
	if len(b.cancelToolNames()) > 0 {
		parts = append(parts, asyncToolCancellationInstructions)
	}

	b.systemMu.Lock()
	composed := strings.Join(parts, "\n\n")
	b.systemInstruction = composed
	b.systemMu.Unlock()

	slog.Debug("composed system instruction", "service", b.Name(), "chars", len(composed))
}
