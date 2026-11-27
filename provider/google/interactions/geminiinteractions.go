// The LLM adapter in this file converts a universal conversation into the input
// of Gemini's Interactions API: a list of steps rather than of contents.
//
//   - user messages become user_input steps, and tool results function_result
//     steps
//   - assistant messages become model_output and function_call steps
//   - a stored thought signature becomes a thought step ahead of the step it
//     belongs to
//
// Thought signatures are read in the form the Gemini service stores them, so a
// conversation carries them between the two Gemini services.

package interactions

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/google/internal/wire"
	"github.com/gojargo/jargo/service/llm/adapter"
)

// The step types of an Interactions input.
const (
	StepUserInput      = "user_input"
	StepModelOutput    = "model_output"
	StepFunctionCall   = "function_call"
	StepFunctionResult = "function_result"
	StepThought        = "thought"
)

// The keys the steps and their content items are built from.
const (
	keyType      = "type"
	keyContent   = "content"
	keyText      = "text"
	keyID        = "id"
	keyName      = "name"
	keyArguments = "arguments"
	keyCallID    = "call_id"
	keyResult    = "result"
	keySignature = "signature"
	keyBookmark  = "bookmark"
	keyData      = "data"
	keyMIMEType  = "mime_type"
	keyURI       = "uri"
)

// typeThoughtSignature is the type of the conversation message the Gemini
// services keep a thought signature in.
const typeThoughtSignature = "thought_signature"

// noOpUserTurn is the minimal user input appended to steps ending on the
// model's side, which the API rejects.
const noOpUserTurn = "."

// modelStepTypes are the steps the model's side of the conversation is made of.
//
//nolint:gochecknoglobals // a lookup table
var modelStepTypes = []string{StepModelOutput, StepFunctionCall, StepThought}

// stepTypes are every step an Interactions input takes.
//
//nolint:gochecknoglobals // a lookup table
var stepTypes = []string{StepUserInput, StepFunctionResult, StepModelOutput, StepFunctionCall, StepThought}

// placeholderSignature is the signature Google documents for a function call
// that has none, in the base64 form a thought step carries.
//
//nolint:gochecknoglobals // derived once from a constant
var placeholderSignature = base64.StdEncoding.EncodeToString([]byte(gemini.PlaceholderThoughtSignature))

// Params is what one Interactions request takes from the conversation: the
// system instruction carried beside the steps, the steps themselves, and the
// tools to advertise.
type Params struct {
	SystemInstruction string
	Input             []map[string]any
	Tools             []map[string]any
}

// Adapter converts a universal conversation into an Interactions request. The
// zero value is ready to use.
type Adapter struct {
	adapter.Base

	// declarations renders the function tools, and keeps the schema adaptations
	// already reported, so a tool set needing them warns once rather than on
	// every inference.
	declarations gemini.Adapter
	// warnedUnsupportedSpecificMessage records that a Google-specific message
	// the API cannot take was reported.
	warnedUnsupportedSpecificMessage atomic.Bool
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter. It is the Gemini
// adapter's, so the thought signatures either Gemini service stores serve both.
func (*Adapter) IDForLLMSpecificMessages() string { return "google" }

// LLMInvocationParams converts the conversation into the parameters of an
// Interactions request.
//
// The steps always end on the user's side: the API rejects an input that ends
// with a model turn. The system instruction is carried beside the steps, so an
// instruction given for this call replaces the conversation's own prompt.
func (a *Adapter) LLMInvocationParams(convo *frames.LLMContext, opts adapter.Options) (Params, error) {
	messages := convo.MessagesFor(a.IDForLLMSpecificMessages())
	contextSystem := convo.System()
	contextSystem, messages = a.ExtractInitialSystem(contextSystem, opts.SystemInstruction, messages)

	steps, err := a.toSteps(messages)
	if err != nil {
		return Params{}, &adapter.ConversionError{Cause: err}
	}
	if len(steps) == 0 || slices.Contains(modelStepTypes, stepType(steps[len(steps)-1])) {
		steps = append(steps, map[string]any{
			keyType:    StepUserInput,
			keyContent: []map[string]any{{keyType: keyText, keyText: noOpUserTurn}},
		})
	}

	params := Params{
		SystemInstruction: a.ResolveSystemInstruction(contextSystem, opts.SystemInstruction, true),
		Input:             steps,
	}
	if tools := a.WithBuiltins(convo.ToolsSchema()); !tools.Empty() {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// ToProviderToolsFormat implements adapter.LLMAdapter, rendering the tools in
// the Interactions API's format. Each standard tool becomes a function tool,
// with its schema adapted to the subset Gemini accepts, and the custom tools
// written for Gemini are included.
func (a *Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []map[string]any {
	var tools []map[string]any
	if len(schema.Standard) > 0 {
		for _, declaration := range a.declarations.FunctionDeclarations(schema.Standard) {
			tools = append(tools, functionTool(declaration))
		}
	}
	// A custom tool that cannot be read is left out rather than failing the whole
	// conversion: the tools are advertised alongside a conversation that is
	// otherwise sendable.
	custom, err := adapter.CustomToolsFor[map[string]any](schema, frames.AdapterTypeGemini)
	if err != nil {
		slog.Error("leaving out a custom tool the adapter cannot read", "err", err)
		return tools
	}
	return append(tools, ToInteractionsTools(custom)...)
}

// ToInteractionsTools converts tools in Gemini's generateContent format to the
// Interactions API's.
//
// A function_declarations entry becomes one function tool per declaration, and
// a built-in tool keyed by its name, such as {"google_search": {}}, becomes that
// tool's type. A tool already carrying a type passes through as it is.
func ToInteractionsTools(tools []map[string]any) []map[string]any {
	var converted []map[string]any
	for _, tool := range tools {
		if _, typed := tool[keyType]; typed {
			converted = append(converted, tool)
			continue
		}
		declarations, ok := functionDeclarations(tool)
		if ok {
			for _, declaration := range declarations {
				converted = append(converted, functionTool(declaration))
			}
			continue
		}
		for _, name := range sortedKeys(tool) {
			entry := map[string]any{keyType: name}
			if config, ok := tool[name].(map[string]any); ok {
				maps.Copy(entry, config)
			}
			converted = append(converted, entry)
		}
	}
	return converted
}

// functionDeclarations reads a tool's function_declarations entry, reporting
// whether it has one.
func functionDeclarations(tool map[string]any) ([]map[string]any, bool) {
	switch list := tool["function_declarations"].(type) {
	case []map[string]any:
		return list, true
	case []any:
		out := make([]map[string]any, 0, len(list))
		for _, item := range list {
			if declaration, ok := item.(map[string]any); ok {
				out = append(out, declaration)
			}
		}
		return out, true
	default:
		return nil, false
	}
}

// functionTool is a function declaration as an Interactions function tool.
func functionTool(declaration map[string]any) map[string]any {
	tool := map[string]any{keyType: "function"}
	maps.Copy(tool, declaration)
	return tool
}

// sortedKeys returns a map's keys in order, so a tool set converts the same way
// every time.
func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// MessagesForLogging implements adapter.LLMAdapter: the conversation's steps,
// with media data and thought signatures replaced with "...".
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	steps, err := a.toSteps(convo.MessagesFor(a.IDForLLMSpecificMessages()))
	if err != nil {
		return nil
	}
	// The steps share their passed-through ones with the conversation, so they
	// are copied before anything is elided.
	steps = deepCopySteps(steps)
	for _, step := range steps {
		if _, ok := step[keySignature]; ok {
			step[keySignature] = "..."
		}
		for _, item := range contentItems(step) {
			if _, ok := item[keyData]; ok {
				item[keyData] = "..."
			}
		}
	}
	return steps
}

// toSteps converts the conversation's messages to steps, thought signatures
// included.
func (a *Adapter) toSteps(messages []frames.Message) ([]map[string]any, error) {
	var steps []map[string]any
	var signatures []map[string]any
	toolNames := map[string]string{}

	for _, m := range messages {
		if m.IsLLMSpecific() {
			native, err := adapter.NativeMessage[map[string]any](m)
			if err != nil {
				a.warnUnsupportedSpecificMessage()
				continue
			}
			switch kind, _ := native[keyType].(string); {
			case kind == typeThoughtSignature:
				signatures = append(signatures, native)
			case slices.Contains(stepTypes, kind):
				steps = append(steps, native)
			default:
				a.warnUnsupportedSpecificMessage()
			}
			continue
		}
		converted, err := messageSteps(m, toolNames)
		if err != nil {
			return nil, err
		}
		steps = append(steps, converted...)
	}

	steps = insertThoughtSteps(steps, signatures)
	steps = regroupParallelCalls(steps)
	return startModelTurnsWithThoughts(steps), nil
}

// messageSteps converts one standard message to steps. toolNames maps each call
// already seen to its function, so a result can be named after its call.
func messageSteps(m frames.Message, toolNames map[string]string) ([]map[string]any, error) {
	if len(m.ToolResults) > 0 {
		steps := make([]map[string]any, 0, len(m.ToolResults))
		for _, r := range m.ToolResults {
			step := map[string]any{
				keyType:   StepFunctionResult,
				keyCallID: r.ID,
				keyResult: functionResult(r.Content),
			}
			if name, ok := toolNames[r.ID]; ok {
				step[keyName] = name
			}
			steps = append(steps, step)
		}
		return steps, nil
	}

	items := contentItemsOf(m)

	if m.Role == frames.RoleAssistant {
		var steps []map[string]any
		if len(items) > 0 {
			steps = append(steps, map[string]any{keyType: StepModelOutput, keyContent: items})
		}
		for _, c := range m.ToolCalls {
			toolNames[c.ID] = c.Name
			arguments, err := callArguments(c.Args)
			if err != nil {
				return nil, fmt.Errorf("arguments of call %s: %w", c.ID, err)
			}
			steps = append(steps, map[string]any{
				keyType:      StepFunctionCall,
				keyID:        c.ID,
				keyName:      c.Name,
				keyArguments: arguments,
			})
		}
		return steps, nil
	}

	// The Interactions API has no other roles; a non-initial system or
	// developer message speaks for the user.
	if len(items) == 0 {
		return nil, nil
	}
	return []map[string]any{{keyType: StepUserInput, keyContent: items}}, nil
}

// callArguments decodes a call's arguments, an empty one being no arguments.
func callArguments(raw json.RawMessage) (any, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		raw = json.RawMessage("{}")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var arguments any
	if err := dec.Decode(&arguments); err != nil {
		return nil, err
	}
	return arguments, nil
}

// contentItemsOf converts a message's content to Interactions content items.
func contentItemsOf(m frames.Message) []map[string]any {
	if len(m.Content) == 0 {
		if m.Text == "" {
			return nil
		}
		return []map[string]any{{keyType: keyText, keyText: m.Text}}
	}
	var items []map[string]any
	for _, p := range m.Content {
		switch p.Type {
		case frames.ContentPartText:
			if p.Text != "" {
				items = append(items, map[string]any{keyType: keyText, keyText: p.Text})
			}
		case frames.ContentPartImageURL:
			if strings.HasPrefix(p.ImageURL, "data:") {
				items = append(items, dataURLItem(p.ImageURL, ""))
			} else {
				items = append(items, map[string]any{keyType: "image", keyURI: p.ImageURL})
			}
		}
	}
	return items
}

// insertThoughtSteps puts each stored signature, as a thought step, ahead of the
// step it belongs to.
//
// A signature's bookmark names the function call, text or image it was received
// with. Signatures are stored in conversation order, so each one is matched
// among the model steps after the previous match.
func insertThoughtSteps(steps, signatures []map[string]any) []map[string]any {
	thoughts := map[int]string{}
	start := 0
	for _, entry := range signatures {
		signature, ok := encodeSignature(entry[keySignature])
		bookmark, _ := entry[keyBookmark].(map[string]any)
		if !ok || len(bookmark) == 0 {
			continue
		}
		for i := start; i < len(steps); i++ {
			if slices.Contains(modelStepTypes, stepType(steps[i])) && bookmarkMatches(bookmark, steps[i]) {
				thoughts[i] = signature
				start = i + 1
				break
			}
		}
	}

	if len(thoughts) == 0 {
		return steps
	}
	result := make([]map[string]any, 0, len(steps)+len(thoughts))
	for i, step := range steps {
		if signature, ok := thoughts[i]; ok {
			result = append(result, map[string]any{keyType: StepThought, keySignature: signature})
		}
		result = append(result, step)
	}
	return result
}

// regroupParallelCalls puts a batch of parallel calls back together, ahead of
// their results.
//
// The conversation stores each call with its result right after it, while the
// model made the batch's calls in one turn and signed only the first. So a
// signed call starts a batch, the unsigned calls and the results that follow it
// belong to it, and the next thought step ends it.
func regroupParallelCalls(steps []map[string]any) []map[string]any {
	result := make([]map[string]any, 0, len(steps))
	for i := 0; i < len(steps); {
		if stepType(steps[i]) != StepThought || i+1 >= len(steps) || stepType(steps[i+1]) != StepFunctionCall {
			result = append(result, steps[i])
			i++
			continue
		}
		calls := []map[string]any{steps[i]}
		var results []map[string]any
	batch:
		for i++; i < len(steps); i++ {
			switch stepType(steps[i]) {
			case StepFunctionCall:
				calls = append(calls, steps[i])
			case StepFunctionResult:
				results = append(results, steps[i])
			default:
				break batch
			}
		}
		result = append(result, calls...)
		result = append(result, results...)
	}
	return result
}

// startModelTurnsWithThoughts opens each model turn that needs a thought step
// with one.
//
// A model turn is a run of steps between the user's. The API rejects a model
// turn holding a thought step that does not open with one, which a
// conversation produces when it stores a model's text and its function calls as
// separate messages, so that turn's first thought step moves to its start. It
// also rejects function calls with no thought step, which is how calls from
// another provider or the application look, so those turns open with the
// placeholder signature Google documents for them.
func startModelTurnsWithThoughts(steps []map[string]any) []map[string]any {
	result := make([]map[string]any, 0, len(steps))
	var turn []map[string]any

	flush := func() {
		thought := slices.IndexFunc(turn, func(s map[string]any) bool { return stepType(s) == StepThought })
		switch {
		case thought >= 0:
			moved := turn[thought]
			turn = slices.Delete(turn, thought, thought+1)
			turn = slices.Insert(turn, 0, moved)
		case slices.ContainsFunc(turn, func(s map[string]any) bool { return stepType(s) == StepFunctionCall }):
			turn = slices.Insert(turn, 0, map[string]any{keyType: StepThought, keySignature: placeholderSignature})
		}
		result = append(result, turn...)
		turn = nil
	}

	for _, step := range steps {
		if slices.Contains(modelStepTypes, stepType(step)) {
			turn = append(turn, step)
			continue
		}
		flush()
		result = append(result, step)
	}
	flush()
	return result
}

// warnUnsupportedSpecificMessage reports, once, a Google-specific message the
// Interactions API cannot take.
func (a *Adapter) warnUnsupportedSpecificMessage() {
	if a.warnedUnsupportedSpecificMessage.CompareAndSwap(false, true) {
		slog.Warn("Skipping Google-specific context messages the Interactions API can't take. " +
			"Only thought signatures and Interactions steps are sent.")
	}
}

// stepType is a step's type.
func stepType(step map[string]any) string {
	kind, _ := step[keyType].(string)
	return kind
}

// contentItems is a step's content items.
func contentItems(step map[string]any) []map[string]any {
	switch items := step[keyContent].(type) {
	case []map[string]any:
		return items
	case []any:
		out := make([]map[string]any, 0, len(items))
		for _, item := range items {
			if m, ok := item.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	default:
		return nil
	}
}

// mediaType is the Interactions content type for a MIME type.
func mediaType(mimeType string) string {
	for _, media := range []string{"image", "audio", "video"} {
		if strings.HasPrefix(mimeType, media+"/") {
			return media
		}
	}
	return "document"
}

// dataURLItem is a content item for a base64 data URL
// ("data:<mime>;base64,<data>"). mimeType overrides the one the URL names, when
// set.
func dataURLItem(url, mimeType string) map[string]any {
	header, data, _ := strings.Cut(url, ",")
	if mimeType == "" {
		_, rest, _ := strings.Cut(header, ":")
		mimeType, _, _ = strings.Cut(rest, ";")
	}
	return map[string]any{keyType: mediaType(mimeType), keyMIMEType: mimeType, keyData: data}
}

// functionResult is a tool message's content as a function result: a JSON
// object, or text.
func functionResult(content string) any {
	dec := json.NewDecoder(strings.NewReader(content))
	dec.UseNumber()
	var decoded any
	if dec.Decode(&decoded) != nil || dec.More() {
		return content
	}
	if object, ok := decoded.(map[string]any); ok {
		return object
	}
	return content
}

// encodeSignature encodes a thought signature for the wire, reporting whether
// there is one. Signatures are stored as raw bytes; a string is already in the
// base64 form the API takes.
func encodeSignature(signature any) (string, bool) {
	switch s := signature.(type) {
	case []byte:
		return base64.StdEncoding.EncodeToString(s), len(s) > 0
	case string:
		return s, s != ""
	default:
		return "", false
	}
}

// bookmarkMatches reports whether a thought signature's bookmark names this
// step.
func bookmarkMatches(bookmark, step map[string]any) bool {
	if callID, _ := bookmark[gemini.BookmarkFunctionCall].(string); callID != "" {
		id, _ := step[keyID].(string)
		return stepType(step) == StepFunctionCall && id == callID
	}
	if stepType(step) != StepModelOutput {
		return false
	}
	if text, _ := bookmark[gemini.BookmarkText].(string); text != "" {
		var stepText strings.Builder
		for _, item := range contentItems(step) {
			t, _ := item[keyText].(string)
			stepText.WriteString(t)
		}
		return wire.TextBookmarkMatches(text, stepText.String())
	}
	if inlineData, ok := bookmark[gemini.BookmarkInlineData]; ok && inlineData != nil {
		blob, _ := inlineData.(map[string]any)
		want, ok := dataLen(blob[keyData])
		if !ok {
			return false
		}
		// Matching in conversation order, the size is enough to tell images
		// apart.
		for _, item := range contentItems(step) {
			if got, ok := dataLen(item[keyData]); ok && got == want {
				return true
			}
		}
	}
	return false
}

// dataLen measures inline data in bytes, whether it is held raw or in the
// base64 form it travels in.
func dataLen(v any) (int, bool) {
	switch d := v.(type) {
	case []byte:
		return len(d), true
	case string:
		raw, err := base64.StdEncoding.DecodeString(d)
		if err != nil {
			return 0, false
		}
		return len(raw), true
	default:
		return 0, false
	}
}

// deepCopySteps copies steps so that nothing written into the copy reaches the
// originals.
func deepCopySteps(steps []map[string]any) []map[string]any {
	out := make([]map[string]any, len(steps))
	for i, step := range steps {
		out[i], _ = deepCopy(step).(map[string]any)
	}
	return out
}

// deepCopy copies the maps and slices of a step.
func deepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = deepCopy(val)
		}
		return out
	case []map[string]any:
		out := make([]map[string]any, len(t))
		for i, val := range t {
			out[i], _ = deepCopy(val).(map[string]any)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = deepCopy(val)
		}
		return out
	default:
		return v
	}
}
