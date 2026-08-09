// Package adapter converts jargo's universal conversation context into the
// request each LLM provider takes.
//
// A provider's wire format is its own: one takes the system prompt as a leading
// message and another as a field beside the conversation, one names a tool
// result by an id and another by the call it answers. An adapter is where that
// translation lives, and it is a plain value the service owns and calls rather
// than a base the service inherits from. That keeps the conversion a pure
// function of the conversation, so it can be tested by comparing what went in
// with what came out, with no endpoint to stand up.
//
// Each provider adapter lives in its own package under this one and satisfies
// [LLMAdapter] for its parameter and tool types.
package adapter

import (
	"fmt"
	"log/slog"
	"sync/atomic"

	"github.com/gojargo/jargo/frames"
)

// ConversionError reports that a conversation could not be mapped into a
// provider's message format. An adapter returns it from its conversion, and the
// LLM service reports it as a generation failure, so a context the provider
// cannot represent is told apart from the provider itself failing.
type ConversionError struct {
	// Cause is the error the conversion failed with.
	Cause error
}

// Error implements error.
func (e *ConversionError) Error() string {
	return fmt.Sprintf("error mapping context messages to provider format: %v", e.Cause)
}

// Unwrap returns the underlying conversion failure.
func (e *ConversionError) Unwrap() error { return e.Cause }

// Options tunes one conversion. The zero value converts the conversation as the
// provider's own API takes it.
type Options struct {
	// SystemInstruction is the instruction this call was given, which stands
	// beside the conversation's own system prompt. It is what a one-shot
	// inference runs under (see llm.InferenceOptions). Empty leaves the
	// conversation's prompt to stand alone.
	SystemInstruction string
	// ConvertDeveloperToUser sends a developer message as a user message, for a
	// provider with no developer role. What the role carries, the late results of
	// an asynchronous tool, is worth more to the model than the role it arrives
	// under.
	ConvertDeveloperToUser bool
}

// LLMAdapter converts a universal conversation into one provider's invocation
// parameters. P is the parameter type that provider's API takes and T its tool
// type, both of which the adapter's own package defines.
type LLMAdapter[P, T any] interface {
	// IDForLLMSpecificMessages is the identifier this provider's messages are
	// held under in a universal context, so a message written in one provider's
	// native format reaches that provider and no other.
	IDForLLMSpecificMessages() string
	// LLMInvocationParams converts the conversation into what this provider's API
	// takes. It returns a [ConversionError] for a conversation the provider has
	// no representation for.
	LLMInvocationParams(convo *frames.LLMContext, opts Options) (P, error)
	// ToProviderToolsFormat renders the advertised tools in this provider's
	// format.
	ToProviderToolsFormat(tools []frames.Tool) []T
	// MessagesForLogging renders the conversation as this provider will see it,
	// for a log or a trace.
	MessagesForLogging(convo *frames.LLMContext) []map[string]any
}

// Base carries the state an adapter keeps between conversions, and the handling
// of a system prompt that every provider shares. Embed it in a provider adapter
// to inherit both.
//
// It is safe for concurrent use: a one-shot inference runs off to the side of
// the pipeline and so may convert a conversation while a generation is
// converting another.
type Base struct {
	// warnedSystemInstruction records that the conflict between an instruction
	// given for a call and the conversation's own prompt has been reported, so it
	// is reported once rather than on every generation of the session.
	warnedSystemInstruction atomic.Bool
}

// ExtractInitialSystem reports the system prompt a provider should send beside
// the conversation, and the messages to send with it.
//
// A conversation carrying nothing but a system prompt is the case this exists
// for. Sending the prompt on its own would leave the message list empty, which
// a provider that requires at least one non-system message rejects, so the
// prompt is sent as a user message instead and no separate instruction goes
// out. systemInstruction is only read to decide whether that displaced a prompt
// the caller also gave.
func (b *Base) ExtractInitialSystem(
	system, systemInstruction string, msgs []frames.Message,
) (string, []frames.Message) {
	if system == "" || len(msgs) > 0 {
		return system, msgs
	}
	if systemInstruction != "" {
		b.warnOnce("both an instruction for this call and a system prompt on the conversation" +
			" are set; using the instruction. The conversation's prompt is being sent as a user" +
			" message so the provider is not given an empty conversation")
	}
	return "", []frames.Message{{Role: frames.RoleUser, Text: system}}
}

// ResolveSystemInstruction settles which system prompt a provider is sent when
// the conversation carries one and the call was given another.
//
// discardContextSystem says what the provider does with the conversation's own
// prompt when both are set. A provider that takes the prompt beside the
// conversation has one field to put it in, so the instruction given for the call
// replaces it. A provider that takes it as a leading message can carry both, and
// keeping both is what lets an instruction supplement the conversation's prompt
// rather than silently replace it; there the returned prompt is empty, because
// the conversation's own is already in the messages.
func (b *Base) ResolveSystemInstruction(
	fromContext, systemInstruction string, discardContextSystem bool,
) string {
	if fromContext != "" && systemInstruction != "" {
		if discardContextSystem {
			b.warnOnce("both an instruction for this call and a system prompt on the" +
				" conversation are set; using the instruction")
		} else {
			b.warnOnce("both an instruction for this call and a system prompt on the" +
				" conversation are set, which may be unintended. Both are being sent; consider" +
				" giving system-level instructions on the conversation and keeping the call's" +
				" instruction for supplementary guidance")
		}
	}
	if systemInstruction != "" {
		return systemInstruction
	}
	if discardContextSystem {
		return fromContext
	}
	// The prompt is already in the messages; there is nothing to send beside them.
	return ""
}

// warnOnce reports msg the first time it is reached and stays quiet after that.
func (b *Base) warnOnce(msg string) {
	if b.warnedSystemInstruction.CompareAndSwap(false, true) {
		slog.Warn(msg)
	}
}
