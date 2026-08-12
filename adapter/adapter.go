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
	"slices"
	"strings"
	"sync"
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
	// EnablePromptCaching marks the conversation so the provider caches the
	// prompt and reads the cache back on the next turn.
	EnablePromptCaching bool
	// EnsureLastMessageIsUser appends a minimal user message when the converted
	// conversation ends on an assistant message. It is for a model without
	// assistant-prefill support, which rejects a request ending that way.
	EnsureLastMessageIsUser bool
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
	// ToProviderToolsFormat renders the advertised toolset in this provider's
	// format, including the custom tools written for it and leaving out those
	// written for another.
	ToProviderToolsFormat(schema frames.ToolsSchema) []T
	// MessagesForLogging renders the conversation as this provider will see it,
	// for a log or a trace.
	MessagesForLogging(convo *frames.LLMContext) []map[string]any
}

// Identifier is the part of an adapter that names the provider it converts for.
// Every [LLMAdapter] satisfies it.
type Identifier interface {
	IDForLLMSpecificMessages() string
}

// ToolsForLogging renders the advertised toolset in the provider's own format,
// boxed so that a caller which cannot name that provider's tool type can still
// carry it. It is for a log or a trace, where what matters is showing the
// toolset the model was actually sent rather than the universal form it was
// converted from.
//
// It is a function rather than a method because an adapter's tool type is a
// parameter of its interface, and a service reporting on itself is reached
// through an interface that cannot name it.
func ToolsForLogging[P, T any](a LLMAdapter[P, T], schema frames.ToolsSchema) []any {
	tools := a.ToProviderToolsFormat(schema)
	out := make([]any, len(tools))
	for i, t := range tools {
		out[i] = t
	}
	return out
}

// CreateLLMSpecificMessage builds a conversation message written in a's
// provider's own format, for something the universal conversation has no
// representation for. Only a's provider is sent it; every other adapter leaves
// it out.
//
// native must be the message type a's own package defines, which is what its
// conversion reads back. Anything else is reported as a conversion failure when
// the conversation is sent.
func CreateLLMSpecificMessage(a Identifier, native any) frames.Message {
	return frames.NewLLMSpecificMessage(a.IDForLLMSpecificMessages(), native)
}

// NativeMessage reads a provider-native message back as the type that
// provider's adapter defines, reporting a conversion failure when it holds
// something else. An adapter calls it on a message it has already established
// is its own.
func NativeMessage[T any](m frames.Message) (T, error) {
	native, ok := m.Native.(T)
	if !ok {
		var want T
		return want, &ConversionError{Cause: &nativeTypeError{
			llm: m.LLM, got: fmt.Sprintf("%T", m.Native), want: fmt.Sprintf("%T", want),
		}}
	}
	return native, nil
}

// nativeTypeError reports a provider-native message holding something its own
// adapter cannot read.
type nativeTypeError struct {
	llm  string
	got  string
	want string
}

// Error implements error.
func (e *nativeTypeError) Error() string {
	return fmt.Sprintf("message for %q holds %s, want %s", e.llm, e.got, e.want)
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

	// mu guards the built-in tools below, which the service adds and withdraws
	// as its registrations change while generations are converting.
	mu sync.RWMutex
	// builtins are the tools the service implements itself, sent on every
	// request without the application having advertised them, each with the
	// instructions that go with it. They are keyed by tool name.
	builtins map[string]Builtin
	// builtinOrder keeps the order they were added in, so a request carries them
	// the same way twice running and a cached prompt prefix stays byte-identical.
	builtinOrder []string
}

// Builtin is a tool the LLM service implements itself, and whatever the model
// has to be told to use it.
type Builtin struct {
	// Tool is the declaration advertised to the model.
	Tool frames.Tool
	// Instructions are appended to the system prompt while the tool is offered.
	// Empty adds nothing.
	Instructions string
}

// SetBuiltin adds a tool the service implements itself, replacing any already
// registered under the same name. It is sent on every request from now on,
// without the application having to advertise it.
//
// It lives here rather than on the conversation because it belongs to the
// service: a conversation is shared, and writing the tool into it would offer
// it to every other service reading that conversation, and edit a context the
// application owns.
func (b *Base) SetBuiltin(builtin Builtin) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.builtins == nil {
		b.builtins = make(map[string]Builtin)
	}
	if _, seen := b.builtins[builtin.Tool.Name]; !seen {
		b.builtinOrder = append(b.builtinOrder, builtin.Tool.Name)
	}
	b.builtins[builtin.Tool.Name] = builtin
}

// RemoveBuiltin withdraws the tool registered under name, reporting whether
// there was one.
func (b *Base) RemoveBuiltin(name string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.builtins[name]; !ok {
		return false
	}
	delete(b.builtins, name)
	b.builtinOrder = slices.DeleteFunc(b.builtinOrder, func(s string) bool { return s == name })
	return true
}

// WithBuiltins returns the toolset with the tools the service implements itself
// appended to the standard ones. An adapter renders the result, so a built-in
// tool reaches the model in the provider's own format rather than in one shape
// that has to suit every provider.
func (b *Base) WithBuiltins(schema frames.ToolsSchema) frames.ToolsSchema {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if len(b.builtinOrder) == 0 {
		return schema
	}
	out := schema
	out.Standard = make([]frames.Tool, 0, len(schema.Standard)+len(b.builtinOrder))
	out.Standard = append(out.Standard, schema.Standard...)
	for _, name := range b.builtinOrder {
		out.Standard = append(out.Standard, b.builtins[name].Tool)
	}
	return out
}

// CustomToolsFor reads the custom tools written for one format back as the tool
// type that adapter defines, reporting a conversion failure for anything else.
func CustomToolsFor[T any](schema frames.ToolsSchema, t frames.AdapterType) ([]T, error) {
	raw := schema.CustomFor(t)
	if len(raw) == 0 {
		return nil, nil
	}
	out := make([]T, 0, len(raw))
	for _, v := range raw {
		tool, ok := v.(T)
		if !ok {
			var want T
			return nil, &ConversionError{Cause: &customToolTypeError{
				adapter: string(t), got: fmt.Sprintf("%T", v), want: fmt.Sprintf("%T", want),
			}}
		}
		out = append(out, tool)
	}
	return out, nil
}

// customToolTypeError reports a custom tool holding something its adapter
// cannot read.
type customToolTypeError struct {
	adapter string
	got     string
	want    string
}

// Error implements error.
func (e *customToolTypeError) Error() string {
	return fmt.Sprintf("custom tool for %q holds %s, want %s", e.adapter, e.got, e.want)
}

// SystemWithBuiltins appends the instructions of every built-in tool currently
// offered to the system prompt, so the model is told how to use what it is being
// sent. It returns system unchanged when there are none.
func (b *Base) SystemWithBuiltins(system string) string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	parts := make([]string, 0, len(b.builtinOrder)+1)
	if system != "" {
		parts = append(parts, system)
	}
	for _, name := range b.builtinOrder {
		if text := b.builtins[name].Instructions; text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n\n")
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
