package llm

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm/adapter"
	"github.com/gojargo/jargo/service/settings"
)

// The service side of the LLM extensions (see LLMExtension): attaching them,
// wiring them up as the service is set up, running every frame the service
// processes and pushes through their layers, and keeping their built-in tools
// in step with the registry.

// Errors reported when an extension cannot be attached.
//
//nolint:gochecknoglobals // sentinel errors
var (
	// ErrExtensionAfterSetup is returned by AddExtension on a service that is
	// set up already. A running service takes an extension over an
	// LLMUpdateSettingsFrame.
	ErrExtensionAfterSetup = errors.New("the service is set up; a running service takes an extension " +
		"over an LLMUpdateSettingsFrame")
	// ErrDuplicateExtension is returned when the service holds an extension of
	// the same type already. A service holds one of each type.
	ErrDuplicateExtension = errors.New("the service holds one extension of each type; if you need a " +
		"parameterized extension twice, make a type of its own for the second")
	// ErrExtensionAttachedElsewhere is returned when another service holds the
	// extension already. An extension belongs to one service.
	ErrExtensionAttachedElsewhere = errors.New("the extension belongs to another service already, and an " +
		"extension belongs to one service; a settings update that reaches several services should " +
		"carry a function that builds an extension for each")
)

// FunctionCallRegistryItem is a registered tool handler and the call options it
// was registered with, as an extension's BuiltinTools sees the registry.
type FunctionCallRegistryItem struct {
	// FunctionName is the name the handler is registered under; empty for the
	// catch-all.
	FunctionName string
	// Handler runs the tool's calls.
	Handler FunctionCallHandler
	// CancelOnInterruption says whether a call is canceled when the user
	// interrupts; false makes the tool asynchronous.
	CancelOnInterruption bool
	// Timeout bounds one call, overriding the service's own bound; zero means
	// the service's.
	Timeout time.Duration
	// CancellableByLLM says whether the model may cancel the tool's calls while
	// they run.
	CancellableByLLM bool
	// AutoRegistered marks a handler registered because an advertised tool
	// carried it, which is dropped again once the tool stops being advertised.
	AutoRegistered bool
}

// registryView is the registry entry as an extension sees it.
func registryView(item registryItem) FunctionCallRegistryItem {
	return FunctionCallRegistryItem{
		FunctionName:         item.name,
		Handler:              item.handler,
		CancelOnInterruption: item.cancelOnInterruption,
		Timeout:              item.timeout,
		CancellableByLLM:     item.cancellableByLLM,
		AutoRegistered:       item.fromToolset,
	}
}

// FunctionCallRunnerItem is one function call in flight: its name, its
// tool_call_id, its arguments and the registration it runs under.
type FunctionCallRunnerItem struct {
	// RegistryItem is the registration the call was dispatched under.
	RegistryItem FunctionCallRegistryItem
	// FunctionName is the name of the tool called.
	FunctionName string
	// ToolCallID identifies the call.
	ToolCallID string
	// Arguments is the raw JSON the model produced for the call.
	Arguments json.RawMessage
	// Context is the conversation the call was made in.
	Context *frames.LLMContext
	// GroupID is the batch the call belongs to, empty when calls are not
	// grouped.
	GroupID string
}

// Extensions implements ExtensionHolder: the extensions attached, from the one
// "closest" to the service to the one "farthest" from it.
func (b *Base) Extensions() []LLMExtension {
	b.extensionsMu.RLock()
	defer b.extensionsMu.RUnlock()
	return slices.Clone(b.extensions)
}

// AddExtension attaches an LLM extension to the service, before the service is
// set up. It is for a component that has the service in hand, such as a
// pairing that owns it, and for an application that builds its service before
// running it. A running service takes an extension over an
// LLMUpdateSettingsFrame with it in the delta's Extensions.
//
// It reports ErrExtensionAfterSetup on a service that is set up, and
// ErrDuplicateExtension when the service holds an extension of the same type.
func (b *Base) AddExtension(extension LLMExtension) error {
	b.extensionsMu.RLock()
	setUp := b.extensionSetup != nil
	b.extensionsMu.RUnlock()
	if setUp {
		return fmt.Errorf("%s: %w", b.Name(), ErrExtensionAfterSetup)
	}
	return b.attachExtension(extension)
}

// attachExtension adds an extension to the ones held, in its place by order,
// and its instruction to the composition. An extension held already is left as
// it is; one of a type held already is refused.
func (b *Base) attachExtension(extension LLMExtension) error {
	b.extensionsMu.Lock()
	if slices.Contains(b.attached, extension) {
		b.extensionsMu.Unlock()
		return nil
	}
	for _, held := range b.attached {
		if reflect.TypeOf(held) == reflect.TypeOf(extension) {
			b.extensionsMu.Unlock()
			return fmt.Errorf("%s already has a %T: %w", b.Name(), extension, ErrDuplicateExtension)
		}
	}
	b.attached = append(b.attached, extension)
	b.sortExtensionsLocked()
	attached := slices.Clone(b.attached)
	b.extensionsMu.Unlock()
	b.storeExtensions(attached)
	b.composeSystemInstruction()
	return nil
}

// detachExtension removes an extension from the ones held, and from the
// running service.
func (b *Base) detachExtension(ctx context.Context, extension LLMExtension) {
	if b.removeExtension(extension) {
		b.unwireExtension(ctx, extension)
	}
	b.composeSystemInstruction()
}

// removeExtension drops an extension from the ones held, reporting whether the
// service is set up, and so had it wired.
func (b *Base) removeExtension(extension LLMExtension) bool {
	b.extensionsMu.Lock()
	b.attached = slices.DeleteFunc(b.attached, func(e LLMExtension) bool { return e == extension })
	b.sortExtensionsLocked()
	attached := slices.Clone(b.attached)
	setUp := b.extensionSetup != nil
	b.extensionsMu.Unlock()
	b.storeExtensions(attached)
	return setUp
}

// sortExtensionsLocked orders the extensions held from the one "closest" to
// the service to the one "farthest" from it: by order, and among equal orders
// by when each was attached. Call it with extensionsMu held.
func (b *Base) sortExtensionsLocked() {
	b.extensions = slices.Clone(b.attached)
	slices.SortStableFunc(b.extensions, func(x, y LLMExtension) int { return cmp.Compare(x.Order(), y.Order()) })
}

// storeExtensions writes the extensions held into the settings, which is where
// a settings update reads the ones held from. A store with no field for them
// is left alone: the service holds them either way.
func (b *Base) storeExtensions(attached []LLMExtension) {
	entries := make(settings.Extensions, 0, len(attached))
	for _, extension := range attached {
		entries = append(entries, extension)
	}
	b.settingsMu.Lock()
	defer b.settingsMu.Unlock()
	_ = settings.SetNamed(b.settingsStore(), "extensions", entries)
}

// attachInitialExtensions attaches every service's async-tool extension, then
// the entries the settings were built with, which take the path a delta's do:
// a function builds an extension, and each attaches in turn.
func (b *Base) attachInitialExtensions() {
	var initial settings.Extensions
	if v, ok := settings.Get(b.settingsStore(), "extensions"); ok {
		initial, _ = v.(settings.Extensions)
	}
	b.storeExtensions(nil)
	// Every service has one: it contributes nothing until an async tool is
	// registered.
	if err := b.attachExtension(NewAsyncToolLLMExtension()); err != nil {
		slog.Error("attaching the async-tool extension failed", "service", b.Name(), "error", err)
	}
	for _, entry := range initial {
		extension, ok := buildLLMExtension(entry)
		if !ok {
			slog.Error("an extension in the settings is not an LLM extension; ignoring it",
				"service", b.Name(), "type", fmt.Sprintf("%T", entry))
			continue
		}
		if err := b.attachExtension(extension); err != nil {
			slog.Error("attaching an extension failed", "service", b.Name(), "error", err)
		}
	}
}

// buildLLMExtension is an extension entry of the settings as an extension:
// itself, or what the function it is builds.
func buildLLMExtension(entry any) (LLMExtension, bool) {
	switch e := entry.(type) {
	case LLMExtension:
		return e, true
	case func() LLMExtension:
		built := e()
		return built, built != nil
	default:
		return nil, false
	}
}

// wireExtension sets an extension up and takes its frames, then hands it the
// latest metadata of each kind the service has processed: a system frame such
// as a PerceptionMetadataFrame can pass before the settings update that
// attaches the extension that reads it.
func (b *Base) wireExtension(ctx context.Context, extension LLMExtension) error {
	base := extension.extensionBase()
	if other := base.AttachedService(); other != nil && other != b {
		return fmt.Errorf("%T belongs to %s, not %s: %w", extension, other.Name(), b.Name(),
			ErrExtensionAttachedElsewhere)
	}
	b.extensionsMu.RLock()
	setup := *b.extensionSetup
	metadata := make([]frames.Frame, 0, len(b.metadataFrames))
	for _, md := range b.metadataFrames {
		metadata = append(metadata, md)
	}
	b.extensionsMu.RUnlock()

	if err := extension.Setup(ctx, setup); err != nil {
		return err
	}
	base.wire(b,
		func(ctx context.Context, f frames.Frame, dir processor.Direction) error {
			return b.pushThroughExtensions(ctx, f, dir, b.extensionsOutside(extension))
		},
		func(ctx context.Context, build func() frames.Frame) error {
			return b.broadcastThroughExtensions(ctx, build, b.extensionsOutside(extension))
		},
	)
	for _, md := range metadata {
		if err := extension.ProcessFrame(ctx, md, processor.Downstream); err != nil {
			return err
		}
	}
	return nil
}

// unwireExtension drops an extension's frames and cleans it up.
func (b *Base) unwireExtension(ctx context.Context, extension LLMExtension) {
	extension.extensionBase().unwire()
	if err := extension.Cleanup(ctx); err != nil {
		slog.ErrorContext(ctx, "cleaning an extension up failed", "service", b.Name(),
			"extension", fmt.Sprintf("%T", extension), "error", err)
	}
}

// extensionsOutside are the extensions "farther" from the service than the
// given one, the "closest" first.
func (b *Base) extensionsOutside(extension LLMExtension) []LLMExtension {
	b.extensionsMu.RLock()
	defer b.extensionsMu.RUnlock()
	i := slices.Index(b.extensions, extension)
	if i < 0 {
		// Detached meanwhile: nothing is outside it any longer.
		return nil
	}
	return slices.Clone(b.extensions[i+1:])
}

// broadcastThroughExtensions pushes a frame each way through the given
// extensions, the two paired as siblings.
func (b *Base) broadcastThroughExtensions(
	ctx context.Context, build func() frames.Frame, extensions []LLMExtension,
) error {
	down, up := build(), build()
	down.Base().SetBroadcastSiblingID(up.ID())
	up.Base().SetBroadcastSiblingID(down.ID())
	if err := b.pushThroughExtensions(ctx, down, processor.Downstream, extensions); err != nil {
		return err
	}
	return b.pushThroughExtensions(ctx, up, processor.Upstream, extensions)
}

// pushThroughExtensions runs a frame through the given extensions and pushes
// what comes out. The extensions filter it in the order given, the one
// "closest" to the service first, and what one returns goes on through the
// rest, each frame through the rest of the chain and out before the next.
func (b *Base) pushThroughExtensions(
	ctx context.Context, f frames.Frame, dir processor.Direction, extensions []LLMExtension,
) error {
	if len(extensions) == 0 {
		return b.pushPastExtensions(ctx, f, dir)
	}
	first, rest := extensions[0], extensions[1:]
	filtered, err := first.FilterPushFrame(ctx, f, dir)
	if err != nil {
		return err
	}
	for _, out := range filtered {
		if err := b.pushThroughExtensions(ctx, out, dir, rest); err != nil {
			return err
		}
	}
	return nil
}

// pushPastExtensions pushes a frame out of the service: the last step of every
// push. No extension filters the frame from here; it has already gone through
// all that need to filter it.
//
// The frames of a response are stamped with the output configuration in effect,
// so a TTS service downstream knows whether to speak them. Nothing is stamped
// until something configures the output, which leaves the decision to whatever
// else in the pipeline may set it on a frame.
func (b *Base) pushPastExtensions(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if skip, ok := b.currentSkipTTS(); ok {
		// A copy per frame: the frames travel on their own from here, and one
		// must not be able to change what another says.
		switch fr := f.(type) {
		case *frames.LLMTextFrame:
			fr.SkipTTS = &skip
		case *frames.LLMFullResponseStartFrame:
			fr.SkipTTS = &skip
		case *frames.LLMFullResponseEndFrame:
			fr.SkipTTS = &skip
		}
	}

	// The speculation gate decides synchronously, so its verdict cannot be torn
	// by another goroutine pushing at the same time. Everything it hands back is
	// pushed past it: routing it back through here would gate it a second time.
	for _, out := range b.gateProcess(f, dir) {
		if err := b.Base.PushFrame(ctx, out.frame, out.dir); err != nil {
			return err
		}
	}
	return nil
}

// processThroughExtensions hands an incoming frame to the extensions from the
// outside in: the one "farthest" from the service processes it first. Each one
// has then seen the frame before it filters anything produced in reaction to
// it.
func (b *Base) processThroughExtensions(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	b.extensionsMu.Lock()
	if md, ok := f.(frames.ServiceMetadata); ok {
		if b.metadataFrames == nil {
			b.metadataFrames = make(map[reflect.Type]frames.ServiceMetadata)
		}
		b.metadataFrames[reflect.TypeOf(md)] = md
	}
	extensions := slices.Clone(b.extensions)
	b.extensionsMu.Unlock()
	for _, extension := range slices.Backward(extensions) {
		if err := extension.ProcessFrame(ctx, f, dir); err != nil {
			return err
		}
	}
	return nil
}

// setupExtensions wires every extension held as the service is set up, then
// advertises their tools and composes the instruction with them, from the
// start: a service that connects at setup sends the model everything known at
// construction.
func (b *Base) setupExtensions(ctx context.Context, s processor.Setup) error {
	b.extensionsMu.Lock()
	b.extensionSetup = &s
	extensions := slices.Clone(b.extensions)
	b.extensionsMu.Unlock()
	for _, extension := range extensions {
		if err := b.wireExtension(ctx, extension); err != nil {
			return err
		}
	}
	b.syncExtensionTools()
	b.composeSystemInstruction()
	return nil
}

// cleanupExtensions unwires every extension held as the service is cleaned up.
func (b *Base) cleanupExtensions(ctx context.Context) {
	for _, extension := range b.Extensions() {
		b.unwireExtension(ctx, extension)
	}
}

// isSetUp reports whether the service has been set up, from when its
// extensions are wired as they attach.
func (b *Base) isSetUp() bool {
	b.extensionsMu.RLock()
	defer b.extensionsMu.RUnlock()
	return b.extensionSetup != nil
}

// updateExtensions attaches the extensions a settings update added, one by one,
// as AddExtension does, wiring each on a service that is set up. held is what
// the settings held before the update.
func (b *Base) updateExtensions(ctx context.Context, held any) error {
	before, _ := held.(settings.Extensions)
	b.settingsMu.RLock()
	v, _ := settings.Get(b.settingsStore(), "extensions")
	b.settingsMu.RUnlock()
	after, _ := v.(settings.Extensions)

	var built []LLMExtension
	for _, entry := range after {
		if slices.ContainsFunc(before, func(h any) bool { return sameEntry(h, entry) }) {
			continue
		}
		extension, ok := buildLLMExtension(entry)
		if !ok {
			return fmt.Errorf("%w: %T is not an LLM extension", settings.ErrType, entry)
		}
		built = append(built, extension)
	}
	// The settings hold what the service held until each is attached.
	b.storeExtensions(b.attachedExtensions())
	for _, extension := range built {
		if err := b.attachExtension(extension); err != nil {
			return err
		}
		if b.isSetUp() {
			if err := b.wireExtension(ctx, extension); err != nil {
				// Not unwired: an extension refused for belonging to another
				// service is still wired to that one.
				b.removeExtension(extension)
				b.composeSystemInstruction()
				return err
			}
		}
	}
	return nil
}

// attachedExtensions are the extensions held, in the order they were attached.
func (b *Base) attachedExtensions() []LLMExtension {
	b.extensionsMu.RLock()
	defer b.extensionsMu.RUnlock()
	return slices.Clone(b.attached)
}

// sameEntry reports whether two extension entries are the same value.
func sameEntry(a, b any) bool {
	av, bv := reflect.ValueOf(a), reflect.ValueOf(b)
	if !av.IsValid() || !bv.IsValid() || av.Type() != bv.Type() || !av.Comparable() {
		return false
	}
	return a == b
}

// registeredTools is the registry as the extensions see it.
func (b *Base) registeredTools() map[string]FunctionCallRegistryItem {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	registered := make(map[string]FunctionCallRegistryItem, len(b.handlers))
	for name, item := range b.handlers {
		registered[name] = registryView(item)
	}
	return registered
}

// syncExtensionTools matches the built-in tools to what the extensions add,
// given the registered tools. It is reconciled from a single reading of the
// registry: a tool an extension no longer returns is withdrawn, one it newly
// returns is registered and advertised, and a name a tool of the application's
// own already has is left to it.
func (b *Base) syncExtensionTools() {
	registered := b.registeredTools()
	wanted := make(map[string]LLMExtensionTool)
	var order []string
	for _, extension := range b.Extensions() {
		for _, tool := range extension.BuiltinTools(registered) {
			if _, seen := wanted[tool.Schema.Name]; !seen {
				order = append(order, tool.Schema.Name)
			}
			wanted[tool.Schema.Name] = tool
		}
	}

	b.handlersMu.Lock()
	var withdrawn []string
	for stale := range b.extensionToolNames {
		if _, keep := wanted[stale]; keep {
			continue
		}
		delete(b.handlers, stale)
		delete(b.extensionToolNames, stale)
		delete(b.announcedExtensionToolNames, stale)
		withdrawn = append(withdrawn, stale)
	}
	var added []frames.Tool
	for _, name := range order {
		tool := wanted[name]
		if b.extensionToolNames[name] {
			continue
		}
		if _, claimed := b.handlers[name]; claimed {
			continue
		}
		handler, ok := extensionToolHandler(tool.Schema.Handler)
		if !ok {
			slog.Error("an extension's tool carries no handler; it is not advertised",
				"service", b.Name(), "function", name)
			continue
		}
		if b.handlers == nil {
			b.handlers = make(map[string]registryItem)
		}
		b.handlers[name] = registryItem{
			name:                 name,
			handler:              handler,
			cancelOnInterruption: tool.cancelOnInterruption(),
		}
		if b.extensionToolNames == nil {
			b.extensionToolNames = make(map[string]bool)
			b.announcedExtensionToolNames = make(map[string]bool)
		}
		b.extensionToolNames[name] = true
		if tool.AnnounceCallStart {
			b.announcedExtensionToolNames[name] = true
		}
		advertised := tool.Schema
		advertised.Handler = nil
		added = append(added, advertised)
	}
	b.handlersMu.Unlock()

	if len(withdrawn) == 0 && len(added) == 0 {
		return
	}
	slices.Sort(withdrawn)
	slog.Debug("extension tools reconciled", "service", b.Name(),
		"added", toolNames(added), "withdrawn", withdrawn)
	b.applyBuiltinTools(added, withdrawn)
}

// extensionToolHandler is the handler an extension's tool carries, as a
// FunctionCallHandler.
func extensionToolHandler(h any) (FunctionCallHandler, bool) {
	switch fn := h.(type) {
	case FunctionCallHandler:
		return fn, fn != nil
	case func(context.Context, FunctionCallParams) error:
		return fn, fn != nil
	default:
		return nil, false
	}
}

// toolNames lists the names of the tools.
func toolNames(tools []frames.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

// applyBuiltinTools brings the tools this service sends into line with the
// extensions' tools. They belong to the service rather than to the
// conversation, so they come and go with the extensions instead of being
// written into a context the application owns.
func (b *Base) applyBuiltinTools(added []frames.Tool, withdrawn []string) {
	holder, ok := b.gen.(AdapterHolder)
	if !ok {
		// A service that converts without an adapter has nowhere to put the
		// tools, and saying so once is better than the model being told to call
		// a tool it is never sent.
		b.warnNoAdapter.Do(func() {
			slog.Warn("an extension's built-in tools need a service that converts through an adapter",
				"service", b.Name())
		})
		return
	}
	adapt := holder.LLMAdapter()
	for _, name := range withdrawn {
		adapt.RemoveBuiltin(name)
	}
	for _, tool := range added {
		adapt.SetBuiltin(adapter.Builtin{Tool: tool})
	}
}

// isExtensionTool reports whether name belongs to an extension's built-in tool,
// whose name is reserved.
func (b *Base) isExtensionTool(name string) bool {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	return b.extensionToolNames[name]
}

// isInternalExtensionTool reports whether name belongs to an extension's
// built-in tool that is the extension's own mechanism, whose calls are not
// announced.
func (b *Base) isInternalExtensionTool(name string) bool {
	b.handlersMu.RLock()
	defer b.handlersMu.RUnlock()
	return b.extensionToolNames[name] && !b.announcedExtensionToolNames[name]
}

// RunningFunctionCalls are the function calls in flight, ordered by
// tool_call_id. It is for a tool handler that acts on calls in progress, such as
// an extension's cancel tool.
func (b *Base) RunningFunctionCalls() []FunctionCallRunnerItem {
	b.callsMu.Lock()
	running := make([]FunctionCallRunnerItem, 0, len(b.calls))
	for _, call := range b.calls {
		running = append(running, FunctionCallRunnerItem{
			RegistryItem: registryView(call.item),
			FunctionName: call.name,
			ToolCallID:   call.toolCallID,
			Arguments:    call.args,
			Context:      call.convo,
			GroupID:      call.groupID,
		})
	}
	b.callsMu.Unlock()
	slices.SortFunc(running, func(x, y FunctionCallRunnerItem) int {
		return strings.Compare(x.ToolCallID, y.ToolCallID)
	})
	return running
}

// CancelFunctionCall cancels the function call in flight with the given
// tool_call_id, whatever it was registered with. Its handler's context is
// canceled, the call is settled downstream as canceled, and
// OnFunctionCallsCanceled is told. reason is what triggered the cancellation,
// for the logs.
func (b *Base) CancelFunctionCall(ctx context.Context, toolCallID, reason string) {
	b.cancelFunctionCallTasks(ctx, func(call *functionCall) bool {
		return call.toolCallID == toolCallID
	}, reason)
}
