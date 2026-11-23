package llm

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// LLMExtensionTool is a tool an extension adds to the service's built-in tools.
//
// Built-in tools go to the model on every inference, beside the tools the
// conversation advertises. Once a tool is added its name is reserved: an
// application cannot register a tool of its own under it.
//
//nolint:revive // the name is the one the extension design uses throughout
type LLMExtensionTool struct {
	// Schema is the tool, carrying its handler in Handler, which must be a
	// FunctionCallHandler.
	Schema frames.Tool
	// CancelOnInterruption says whether a call of the tool is canceled when the
	// user interrupts. Nil means true; false makes it an asynchronous tool.
	CancelOnInterruption *bool
	// AnnounceCallStart says whether the service announces a call of the tool
	// when it starts, with OnFunctionCallsStarted and a
	// FunctionCallsStartedFrame, as it does for an application's own tools.
	// False, the default, suits a tool that is the extension's own mechanism,
	// which the application and the rest of the pipeline need not hear about.
	// Either way, the call's FunctionCallInProgressFrame and
	// FunctionCallResultFrame are pushed, and the call and its result go into
	// the conversation.
	AnnounceCallStart bool
}

// cancelOnInterruption resolves CancelOnInterruption, whose default is true.
func (t LLMExtensionTool) cancelOnInterruption() bool {
	return t.CancelOnInterruption == nil || *t.CancelOnInterruption
}

// LLMExtension is a pluggable feature or behavior of an LLM service. It can add
// to the service's system instruction, watch the frames the service processes,
// filter what the service pushes, and push frames of its own.
//
// An extension embeds BaseLLMExtension, which gives it the defaults for every
// method below and the PushFrame and BroadcastFrame it pushes its own frames
// with, and overrides what it needs.
//
// Extensions wrap the service the way middleware wraps a handler: as the layers
// of an onion, with the service at the center. Order places an extension among
// the layers: one with a lower order sits "closer" to the service, one with a
// higher order "farther" from it. An incoming frame, one the service
// processes, goes through the layers from the outside in. An outgoing frame,
// one pushed from the service, goes through them from the inside out. With
// extensions A, B and C, A "closest" to the service:
//
//	incoming:  C.ProcessFrame, B.ProcessFrame, A.ProcessFrame, the service
//	outgoing:  the service, A.FilterPushFrame, B.FilterPushFrame, C.FilterPushFrame
//
// To an extension, everything inside its layer is "the service": the service
// itself and the extensions "closer" to it. So an extension's author needs to
// know nothing about what else is attached, and has two guarantees:
//
//   - FilterPushFrame sees every frame pushed from inside the layer, whether
//     the service pushed it or a "closer" extension did.
//   - ProcessFrame sees an incoming frame before FilterPushFrame sees anything
//     produced in reaction to it.
//
// An extension is attached to a service by a component that has the service in
// hand, before the pipeline runs (Base.AddExtension), or by an
// LLMUpdateSettingsFrame whose delta lists it in Extensions. An extension
// belongs to one service. From then on the service:
//
//   - composes SystemInstruction onto its system instruction, after the
//     application's prompt;
//   - adds the tools BuiltinTools returns to its built-in tools, synced with
//     the registered tools before each inference;
//   - hands it every incoming frame, in ProcessFrame, before the extensions
//     "closer" to the service, and the service, handle the frame;
//   - passes every frame pushed from inside its layer through
//     FilterPushFrame, which can hold the frame back, drop it, or add frames;
//   - sends the frames it pushes with PushFrame and BroadcastFrame through the
//     extensions "farther" from the service than it, then out of the service.
//
// A service holds one extension of each type, which GetExtension finds. A
// developer who needs a parameterized extension twice should make a type of its
// own for the second, embedding the first.
//
// The service calls an extension from more than one goroutine: the frames it
// processes arrive on one, and the frames it pushes leave from the generation
// and from each tool call. An extension guards its own state.
//
//nolint:revive // the name is the one the extension design uses throughout
type LLMExtension interface {
	// Order is where the extension sits among the layers around a service: an
	// extension with a lower order is "closer" to the service. The "closest"
	// one filters an outgoing frame first and sees an incoming frame last.
	// Instructions compose lowest first. Among extensions of equal order, one
	// attached earlier sits "closer".
	Order() int
	// SystemInstruction is the standing instructions the extension
	// contributes, or "". It is composed onto the service's system instruction
	// after the application's prompt, in extension order, each time the service
	// composes it.
	SystemInstruction() string
	// BuiltinTools are the tools the extension adds, given the tools registered
	// with the service by name, the extensions' own included; the empty name
	// keys a catch-all handler. It is called whenever the service syncs its
	// registered tools, before an inference; the service advertises what is
	// returned and withdraws what no longer is. SystemInstruction is read after
	// it, so an instruction can describe the tools.
	BuiltinTools(registered map[string]FunctionCallRegistryItem) []LLMExtensionTool
	// Setup sets the extension up, as the service it extends is set up.
	Setup(ctx context.Context, s processor.Setup) error
	// Cleanup releases what the extension holds, as the service is cleaned up.
	Cleanup(ctx context.Context) error
	// ProcessFrame watches an incoming frame, before the service handles it.
	// The extensions "farther" from the service than this one have seen the
	// frame. The ones "closer" to it handle the frame next, then the service
	// does. Anything they produce in reaction reaches FilterPushFrame after
	// this returns.
	ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error
	// FilterPushFrame filters a frame pushed from inside this extension's
	// layer: the service pushed it, or an extension "closer" to it did. It
	// returns the frames to push in its place, in the same direction: the frame
	// alone to pass it on, none to hold it back or drop it, or more frames to
	// add some. The frames returned go on through the extensions "farther" from
	// the service than this one. A frame for the other direction is pushed with
	// PushFrame or BroadcastFrame.
	FilterPushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) ([]frames.Frame, error)

	// extensionBase is the embedded BaseLLMExtension, which the service wires
	// the extension's own pushes through.
	extensionBase() *BaseLLMExtension
}

// extensionPush is how a service takes the frames an extension pushes.
type extensionPush func(ctx context.Context, f frames.Frame, dir processor.Direction) error

// extensionBroadcast is how a service takes the frames an extension
// broadcasts.
type extensionBroadcast func(ctx context.Context, build func() frames.Frame) error

// BaseLLMExtension is embedded by every LLM extension. It gives the defaults: an
// order of zero, no instruction, no tools, nothing done with the frames the
// service processes, and every frame pushed passed on. It also holds what the
// service wired the extension with, which PushFrame and BroadcastFrame send the
// extension's own frames through.
//
//nolint:revive // the name is the one the extension design uses throughout
type BaseLLMExtension struct {
	mu sync.Mutex
	// service is the service the extension is wired to, nil until that service
	// is set up.
	service   *Base
	push      extensionPush
	broadcast extensionBroadcast
}

// extensionBase implements LLMExtension.
func (e *BaseLLMExtension) extensionBase() *BaseLLMExtension { return e }

// Order implements LLMExtension: zero by default.
func (e *BaseLLMExtension) Order() int { return 0 }

// SystemInstruction implements LLMExtension: none by default.
func (e *BaseLLMExtension) SystemInstruction() string { return "" }

// BuiltinTools implements LLMExtension: none by default.
func (e *BaseLLMExtension) BuiltinTools(map[string]FunctionCallRegistryItem) []LLMExtensionTool {
	return nil
}

// Setup implements LLMExtension: nothing to set up by default.
func (e *BaseLLMExtension) Setup(context.Context, processor.Setup) error { return nil }

// Cleanup implements LLMExtension: nothing to release by default.
func (e *BaseLLMExtension) Cleanup(context.Context) error { return nil }

// ProcessFrame implements LLMExtension: nothing is done with the frame by
// default.
func (e *BaseLLMExtension) ProcessFrame(context.Context, frames.Frame, processor.Direction) error {
	return nil
}

// FilterPushFrame implements LLMExtension: the frame is passed on by default.
func (e *BaseLLMExtension) FilterPushFrame(
	_ context.Context, f frames.Frame, _ processor.Direction,
) ([]frames.Frame, error) {
	return []frames.Frame{f}, nil
}

// AttachedService is the service the extension is attached to, known once that
// service is set up; nil before.
func (e *BaseLLMExtension) AttachedService() *Base {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.service
}

// PushFrame pushes a frame of the extension's own. The frame goes through the
// FilterPushFrame of each extension "farther" from the service than this one,
// then leaves the service. Before the service is set up there is nowhere for it
// to go, and it is dropped.
func (e *BaseLLMExtension) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	e.mu.Lock()
	push := e.push
	e.mu.Unlock()
	if push == nil {
		return nil
	}
	return push(ctx, f, dir)
}

// BroadcastFrame broadcasts a frame of the extension's own, upstream and
// downstream. build is called once per direction, and the two frames are paired
// as siblings. Each goes through the FilterPushFrame of each extension
// "farther" from the service than this one, then leaves the service.
func (e *BaseLLMExtension) BroadcastFrame(ctx context.Context, build func() frames.Frame) error {
	e.mu.Lock()
	broadcast := e.broadcast
	e.mu.Unlock()
	if broadcast == nil {
		return nil
	}
	return broadcast(ctx, build)
}

// wire takes the service's handlers for the extension's own frames.
func (e *BaseLLMExtension) wire(service *Base, push extensionPush, broadcast extensionBroadcast) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.service, e.push, e.broadcast = service, push, broadcast
}

// unwire drops the service's handlers.
func (e *BaseLLMExtension) unwire() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.service, e.push, e.broadcast = nil, nil, nil
}

// ExtensionHolder is implemented by a service that holds LLM extensions. *Base
// implements it.
type ExtensionHolder interface {
	// Extensions are the extensions attached, from the one "closest" to the
	// service to the one "farthest" from it.
	Extensions() []LLMExtension
}

// GetExtension is the extension of type T attached to the service, exactly:
// an extension of a type embedding T is an extension of its own. It reports
// false if none of that type is attached.
func GetExtension[T LLMExtension](service ExtensionHolder) (T, bool) {
	for _, extension := range service.Extensions() {
		if found, ok := extension.(T); ok {
			return found, true
		}
	}
	var zero T
	return zero, false
}
