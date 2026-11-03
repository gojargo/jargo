// Package gemini converts a universal conversation into the request Gemini
// takes, and with it Vertex AI, which serves the same models.
package gemini

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
)

// The keys Gemini's generateContent body is built from.
const (
	keyRole             = "role"
	keyParts            = "parts"
	keyName             = "name"
	keyText             = "text"
	keyID               = "id"
	keyFunctionCall     = "functionCall"
	keyFunctionResponse = "functionResponse"
	keyInlineData       = "inlineData"
	keyData             = "data"
	keyThoughtSignature = "thoughtSignature"
)

// The keys of the conversation message the Gemini service keeps a thought
// signature in. Gemini signs a part of its response, and expects the signature
// back on that same part when the conversation is sent again. The bookmark is
// what finds the part: the id of a function call, the text of the response so
// far, or the inline data the part carried.
const (
	keyType              = "type"
	keySignature         = "signature"
	keyBookmark          = "bookmark"
	typeThoughtSignature = "thought_signature"
	BookmarkFunctionCall = "function_call"
	BookmarkText         = "text"
	BookmarkInlineData   = "inline_data"
)

// NewThoughtSignature builds the conversation message keeping a thought
// signature Gemini put on a part of its response, written for Gemini and sent
// to no other provider. bookmark identifies the part, under one of the
// Bookmark keys.
func NewThoughtSignature(signature any, bookmark map[string]any) frames.Message {
	return adapter.CreateLLMSpecificMessage(&Adapter{}, map[string]any{
		keyType:      typeThoughtSignature,
		keySignature: signature,
		keyBookmark:  bookmark,
	})
}

// The roles Gemini's contents take. It calls the assistant the model, and has
// no system or developer input role: a message in either enters as the user.
const (
	roleUser  = "user"
	roleModel = "model"
)

// unnamedToolResult names a tool result whose call cannot be found. Gemini
// requires a name on every functionResponse, and a result can outlive the call
// it answers: an asynchronous tool's messages carry only the call's id.
const unnamedToolResult = "tool_call_result"

// noOpUserTurn is the minimal user content appended to a conversation ending on
// a model turn. A full stop says nothing in any language, which is the point: it
// is there to satisfy the shape, not to be read.
const noOpUserTurn = "."

// placeholderThoughtSignature is the signature Google documents for a function
// call that has none, which skips the validation the model would otherwise
// refuse the request over. It travels base64-encoded, the field being bytes on
// the wire.
const placeholderThoughtSignature = "skip_thought_signature_validator"

// Params is what one generateContent call takes from the conversation: the
// system instruction Gemini carries beside the conversation, the contents
// themselves, and the tools to advertise.
type Params struct {
	SystemInstruction string
	Contents          []map[string]any
	Tools             []map[string]any
}

// Adapter converts a universal conversation into a Gemini request. The zero
// value is ready to use.
type Adapter struct {
	adapter.Base

	// warnedMu guards warnedSchemaAdaptations, which conversions running at
	// once can both update.
	warnedMu sync.Mutex
	// warnedSchemaAdaptations are the (adaptation, tool) pairs already reported,
	// so a tool set needing them warns once rather than on every inference.
	warnedSchemaAdaptations map[schemaAdaptation]bool
}

// schemaAdaptation is one kind of adaptation made to one tool's schema.
type schemaAdaptation struct {
	note, tool string
}

// Compile-time check that the adapter satisfies the contract.
var _ adapter.LLMAdapter[Params, map[string]any] = (*Adapter)(nil)

// IDForLLMSpecificMessages implements adapter.LLMAdapter.
func (*Adapter) IDForLLMSpecificMessages() string { return "google" }

// LLMInvocationParams converts the conversation into a generateContent request.
//
// Gemini carries the system instruction beside the conversation rather than in
// it, so it has one field to put it in: an instruction given for this call
// replaces the conversation's own prompt rather than supplementing it.
func (a *Adapter) LLMInvocationParams(
	convo *frames.LLMContext, opts adapter.Options,
) (Params, error) {
	params, err := a.ParamsFromMessages(convo.System(), convo.MessagesFor(a.IDForLLMSpecificMessages()), opts)
	if err != nil {
		return Params{}, err
	}
	if tools := a.WithBuiltins(convo.ToolsSchema()); !tools.Empty() {
		params.Tools = a.ToProviderToolsFormat(tools)
	}
	return params, nil
}

// ParamsFromMessages converts a system prompt and the messages written for this
// provider into the system instruction and contents of a request, leaving the
// tools for the caller to settle. It is what LLMInvocationParams runs on the
// conversation's own messages, and what an adapter built on this one runs on
// messages it has reworked first.
func (a *Adapter) ParamsFromMessages(
	contextSystem string, messages []frames.Message, opts adapter.Options,
) (Params, error) {
	fromContext, msgs := a.ExtractInitialSystem(contextSystem, opts.SystemInstruction, messages)
	system := a.ResolveSystemInstruction(fromContext, opts.SystemInstruction, true)

	contents, err := ToContents(msgs)
	if err != nil {
		return Params{}, err
	}
	// A conversation of nothing but tool turns gives the model no prose to answer,
	// and Gemini reads the system instruction as framing rather than as something
	// to act on. Saying it again as a user message is what prompts a reply.
	if system != "" && !hasProse(contents) {
		contents = append(contents, textContent(roleUser, system))
	}
	contents = dropEmpty(contents)
	if opts.EnsureLastMessageIsUser {
		contents = EnsureLastMessageIsUser(contents)
	}

	return Params{SystemInstruction: system, Contents: contents}, nil
}

// ToContents converts the conversation's messages into Gemini contents. The
// assistant role maps to "model", and a system or developer message enters as
// the user, Gemini having neither input role. Tool turns become functionCall
// parts (model) and functionResponse parts (user), paired by the call id
// carried on both.
//
// A message written for Gemini is either a thought signature the service kept
// for a part of an earlier model turn, which is put back on that part, or a
// content already in Gemini's format, which is copied in as it stands.
func ToContents(msgs []frames.Message) ([]map[string]any, error) {
	names := toolCallNames(msgs)
	out := make([]map[string]any, 0, len(msgs))
	var signatures []map[string]any
	for _, m := range msgs {
		switch {
		case m.IsLLMSpecific():
			native, err := adapter.NativeMessage[map[string]any](m)
			if err != nil {
				return nil, err
			}
			if native[keyType] == typeThoughtSignature {
				signatures = append(signatures, native)
				continue
			}
			// The content belongs to the conversation, which other conversions
			// read at the same time, and the request built from it is edited from
			// here on: signatures are put on its parts. It is copied whole so none
			// of that reaches the conversation's own.
			copied, _ := deepCopy(native).(map[string]any)
			out = append(out, copied)
		case len(m.ToolResults) > 0:
			out = append(out, map[string]any{
				keyRole: roleUser, keyParts: toolResultParts(m.ToolResults, names),
			})
		case len(m.ToolCalls) > 0:
			out = append(out, map[string]any{
				keyRole: roleModel, keyParts: toolCallParts(m.Text, m.ToolCalls),
			})
		default:
			role := roleUser
			if m.Role == frames.RoleAssistant {
				role = roleModel
			}
			out = append(out, textContent(role, m.Text))
		}
	}
	applyThoughtSignatures(signatures, out)
	out = mergeParallelToolCallsForThinking(signatures, out)
	addPlaceholderThoughtSignatures(out)
	return out, nil
}

// addPlaceholderThoughtSignatures gives the function calls of an unsigned model
// turn the placeholder signature.
//
// Gemini 3 refuses a model turn whose function calls carry no thought
// signature. That is what a call another provider made looks like after a
// switch mid-conversation, and what a call the application wrote itself looks
// like. A turn already holding a signed part is left as it stands: the model
// signs only the first call of a parallel batch and expects the rest unsigned.
func addPlaceholderThoughtSignatures(contents []map[string]any) {
	signature := base64.StdEncoding.EncodeToString([]byte(placeholderThoughtSignature))
	for _, c := range contents {
		if role, _ := c[keyRole].(string); role != roleModel {
			continue
		}
		parts, ok := c[keyParts].([]map[string]any)
		if !ok || anySigned(parts) {
			continue
		}
		for _, part := range parts {
			if _, isCall := part[keyFunctionCall]; isCall {
				part[keyThoughtSignature] = signature
			}
		}
	}
}

// anySigned reports whether any of the parts already carries a signature.
func anySigned(parts []map[string]any) bool {
	for _, part := range parts {
		if present(part[keyThoughtSignature]) {
			return true
		}
	}
	return false
}

// present reports whether a value is set to something: neither missing nor
// empty.
func present(v any) bool {
	switch v := v.(type) {
	case nil:
		return false
	case string:
		return v != ""
	case []byte:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	default:
		return true
	}
}

// contentParts returns a content's parts, in the form this adapter builds them.
func contentParts(content map[string]any) []map[string]any {
	parts, _ := content[keyParts].([]map[string]any)
	return parts
}

// applyThoughtSignatures puts each kept thought signature back on the part of
// the model turn it was given for.
//
// The signatures are in the order the model produced them, and so are the
// model turns, so each search starts past the turn the previous signature was
// put on. A signature always belongs to the last part of its turn.
func applyThoughtSignatures(signatures, contents []map[string]any) {
	if len(signatures) == 0 {
		return
	}
	var model []map[string]any
	for _, c := range contents {
		if role, _ := c[keyRole].(string); role == roleModel {
			model = append(model, c)
		}
	}

	applied, start := 0, 0
	for _, ts := range signatures {
		signature := ts[keySignature]
		bookmark, _ := ts[keyBookmark].(map[string]any)
		if !present(signature) || len(bookmark) == 0 {
			continue
		}
		for i := start; i < len(model); i++ {
			parts := contentParts(model[i])
			if len(parts) == 0 {
				continue
			}
			last := parts[len(parts)-1]
			if bookmarkMatchesPart(bookmark, last) {
				last[keyThoughtSignature] = signature
				applied++
				start = i + 1
				break
			}
		}
	}
	slog.Debug("applied thought signatures", "kept", len(signatures), "applied", applied)
}

// bookmarkMatchesPart reports whether a signature's bookmark identifies part.
func bookmarkMatchesPart(bookmark, part map[string]any) bool {
	if id, _ := bookmark[BookmarkFunctionCall].(string); id != "" {
		call, _ := part[keyFunctionCall].(map[string]any)
		got, _ := call[keyID].(string)
		return call != nil && got == id
	}
	if text, _ := bookmark[BookmarkText].(string); text != "" {
		return textBookmarkMatches(text, part)
	}
	if blob, _ := bookmark[BookmarkInlineData].(map[string]any); len(blob) > 0 {
		return inlineDataBookmarkMatches(blob, part)
	}
	slog.Warn("unknown thought signature bookmark", "bookmark", bookmark)
	return false
}

// textBookmarkMatches reports whether a text part is the one a text bookmark
// recorded. Whitespace is normalized before comparing, and a prefix either way
// matches: the part is cut short when the response was interrupted, and the
// bookmark holds only the text seen so far when the signature arrived early.
func textBookmarkMatches(bookmark string, part map[string]any) bool {
	text, _ := part[keyText].(string)
	if text == "" {
		return false
	}
	want := strings.Join(strings.Fields(bookmark), " ")
	got := strings.Join(strings.Fields(text), " ")
	return got == want || strings.HasPrefix(want, got) || strings.HasPrefix(got, want)
}

// inlineDataBookmarkMatches reports whether an inline-data part is the one a
// bookmark recorded. Comparing the length of the data is enough, the search
// already running in strict turn order, and comparing the data itself would be
// expensive.
func inlineDataBookmarkMatches(bookmark, part map[string]any) bool {
	blob, _ := part[keyInlineData].(map[string]any)
	if blob == nil {
		return false
	}
	got, okGot := dataLen(blob[keyData])
	want, okWant := dataLen(bookmark[keyData])
	return okGot && okWant && got == want
}

// dataLen measures inline data in bytes, whether it is held raw or in the
// base64 form it travels in.
func dataLen(v any) (int, bool) {
	switch d := v.(type) {
	case []byte:
		return len(d), true
	case string:
		if raw, err := base64.StdEncoding.DecodeString(d); err == nil {
			return len(raw), true
		}
		return len(d), true
	default:
		return 0, false
	}
}

// mergeParallelToolCallsForThinking regroups a batch of parallel tool calls
// into one model turn, and their results into the one user turn after it.
//
// Gemini expects each side of a parallel batch in a single content: all the
// functionCall parts in one model turn, all the matching functionResponse parts
// in the following user turn. Vertex AI rejects a request whose counts differ;
// the Gemini API is more lenient but documents the grouped form. The
// conversation keeps each call, and each result, as a message of its own, so a
// batch arrives split and is put back together here.
//
// It only matters with thinking on, and only then can it be done: the model
// signs the first call of a parallel batch and only that one, so a signed call
// starts a group and the unsigned calls after it belong to it, while each of a
// run of sequential calls carries a signature of its own. Any other message
// found inside a group is kept and put back after it.
func mergeParallelToolCallsForThinking(signatures, contents []map[string]any) []map[string]any {
	if len(contents) == 0 {
		return contents
	}
	// Whether any signature is bookmarked to a function call answers both
	// whether thinking is on and whether the conversation has calls at all.
	callSigned := false
	for _, ts := range signatures {
		if bookmark, _ := ts[keyBookmark].(map[string]any); present(bookmark[BookmarkFunctionCall]) {
			callSigned = true
			break
		}
	}
	if !callSigned {
		return contents
	}

	merged := make([]map[string]any, 0, len(contents))
	for i := 0; i < len(contents); {
		current := contents[i]
		if !isToolCallContent(current) || !anySigned(contentParts(current)) {
			merged = append(merged, current)
			i++
			continue
		}

		calls := append([]map[string]any(nil), contentParts(current)...)
		var responses []map[string]any
		var others []map[string]any
		j := i + 1
	scan:
		for ; j < len(contents); j++ {
			next := contents[j]
			switch {
			case isToolCallContent(next):
				if anySigned(contentParts(next)) {
					// A new parallel group starts here.
					break scan
				}
				calls = append(calls, contentParts(next)...)
			case isToolResponseContent(next):
				responses = append(responses, contentParts(next)...)
			default:
				others = append(others, next)
			}
		}
		merged = append(merged, map[string]any{keyRole: roleModel, keyParts: calls})
		if len(responses) > 0 {
			merged = append(merged, map[string]any{keyRole: roleUser, keyParts: responses})
		}
		merged = append(merged, others...)
		i = j
	}
	return merged
}

// isToolCallContent reports whether a content is a model turn made only of
// function calls.
func isToolCallContent(c map[string]any) bool {
	return onlyPartsWith(c, roleModel, keyFunctionCall)
}

// isToolResponseContent reports whether a content is a user turn made only of
// function responses.
func isToolResponseContent(c map[string]any) bool {
	return onlyPartsWith(c, roleUser, keyFunctionResponse)
}

// onlyPartsWith reports whether a content in role has parts, every one of them
// carrying key.
func onlyPartsWith(c map[string]any, role, key string) bool {
	if r, _ := c[keyRole].(string); r != role {
		return false
	}
	parts := contentParts(c)
	if len(parts) == 0 {
		return false
	}
	for _, p := range parts {
		if !present(p[key]) {
			return false
		}
	}
	return true
}

// deepCopy copies the maps and slices of a content written for Gemini, so the
// copy shares nothing that a conversion writes into. Anything else is a value
// no conversion edits and is carried over as it stands.
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
	case json.RawMessage:
		return append(json.RawMessage(nil), t...)
	case []byte:
		return append([]byte(nil), t...)
	default:
		return v
	}
}

// EnsureLastMessageIsUser appends a minimal user content when the conversation
// ends on a model turn, which a model that cannot continue one rejects.
func EnsureLastMessageIsUser(contents []map[string]any) []map[string]any {
	if n := len(contents); n > 0 {
		if role, _ := contents[n-1][keyRole].(string); role == roleModel {
			return append(contents, textContent(roleUser, noOpUserTurn))
		}
	}
	return contents
}

// textContent builds a content carrying one text part.
func textContent(role, text string) map[string]any {
	return map[string]any{keyRole: role, keyParts: []map[string]any{{keyText: text}}}
}

// hasProse reports whether any content is something the model said or was told
// in words, as opposed to a tool call or its result.
func hasProse(contents []map[string]any) bool {
	for _, c := range contents {
		parts, ok := c[keyParts].([]map[string]any)
		if !ok || len(parts) != 1 {
			continue
		}
		if text, ok := parts[0][keyText].(string); ok && text != "" {
			return true
		}
	}
	return false
}

// dropEmpty removes contents carrying no parts, which Gemini rejects.
func dropEmpty(contents []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(contents))
	for _, c := range contents {
		if parts, ok := c[keyParts].([]map[string]any); ok && len(parts) > 0 {
			out = append(out, c)
		}
	}
	return out
}

// toolCallNames maps each tool call in the conversation to the name it was made
// with, so a result can be named after the call it answers rather than after
// whatever it happens to carry itself.
func toolCallNames(msgs []frames.Message) map[string]string {
	names := make(map[string]string)
	for _, m := range msgs {
		for _, c := range m.ToolCalls {
			names[c.ID] = c.Name
		}
	}
	return names
}

// toolCallParts renders an assistant turn's optional preamble and tool calls as
// Gemini parts.
func toolCallParts(text string, calls []frames.ToolCall) []map[string]any {
	parts := make([]map[string]any, 0, len(calls)+1)
	if text != "" {
		parts = append(parts, map[string]any{keyText: text})
	}
	for _, c := range calls {
		args := c.Args
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		parts = append(parts, map[string]any{
			keyFunctionCall: map[string]any{keyID: c.ID, keyName: c.Name, "args": args},
		})
	}
	return parts
}

// toolResultParts renders tool outputs as Gemini functionResponse parts. Each
// carries the id of the call it answers, which is what tells apart the results
// of a batch of parallel calls to the same tool.
func toolResultParts(results []frames.ToolResult, names map[string]string) []map[string]any {
	parts := make([]map[string]any, 0, len(results))
	for _, r := range results {
		parts = append(parts, map[string]any{
			"functionResponse": map[string]any{
				keyID:      r.ID,
				keyName:    toolResultName(r, names),
				"response": FunctionResponseDict(r.Content),
			},
		})
	}
	return parts
}

// toolResultName names a result after the call it answers, falling back to the
// name recorded on the result itself and then to a placeholder.
func toolResultName(r frames.ToolResult, names map[string]string) string {
	if name, ok := names[r.ID]; ok && name != "" {
		return name
	}
	if r.Name != "" {
		return r.Name
	}
	return unnamedToolResult
}

// FunctionResponseDict shapes a tool result for Gemini's functionResponse,
// which requires an object. A result that decodes to a JSON object passes
// through as it stands. Any other JSON value (null, a scalar, an array) is
// wrapped decoded, as {"value": <value>}, and text that is not JSON at all, such
// as the "COMPLETED" a function returning nothing reports, is wrapped as it
// stands.
func FunctionResponseDict(content string) any {
	dec := json.NewDecoder(strings.NewReader(content))
	dec.UseNumber()
	var decoded any
	if dec.Decode(&decoded) != nil || dec.More() {
		return map[string]any{"value": content}
	}
	if _, ok := decoded.(map[string]any); ok {
		return json.RawMessage(content)
	}
	return map[string]any{"value": decoded}
}

// ToProviderToolsFormat implements adapter.LLMAdapter, rendering the standard
// tools as Gemini functionDeclarations and appending the custom tools written
// for Gemini beside them. Each tool's parameters are adapted to the subset of
// JSON Schema Gemini accepts (see adaptSchema), and the adaptations made are
// reported once per tool.
func (a *Adapter) ToProviderToolsFormat(schema frames.ToolsSchema) []map[string]any {
	var out []map[string]any
	if len(schema.Standard) > 0 {
		decls := make([]map[string]any, 0, len(schema.Standard))
		adapted := map[string][]string{}
		for _, t := range schema.Standard {
			d := map[string]any{keyName: t.Name}
			if t.Description != "" {
				d["description"] = t.Description
			}
			notes := map[string]bool{}
			if params := geminiParameters(t.Parameters, notes); params != nil {
				d["parameters"] = params
			}
			for note := range notes {
				adapted[note] = append(adapted[note], t.Name)
			}
			decls = append(decls, d)
		}
		a.warnSchemaAdaptations(adapted)
		out = append(out, map[string]any{"functionDeclarations": decls})
	}
	// A custom tool that cannot be read is left out rather than failing the whole
	// conversion: the tools are advertised alongside a conversation that is
	// otherwise sendable.
	custom, err := adapter.CustomToolsFor[map[string]any](schema, frames.AdapterTypeGemini)
	if err != nil {
		slog.Error("leaving out a custom tool the adapter cannot read", "err", err)
		return out
	}
	return append(out, custom...)
}

// geminiParameters returns the tool's JSON-Schema parameters adapted to what
// Gemini accepts, recording each kind of adaptation made in notes. On a parse
// error the raw schema passes through unchanged.
func geminiParameters(raw json.RawMessage, notes map[string]bool) any {
	if len(raw) == 0 {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var schema map[string]any
	if dec.Decode(&schema) != nil {
		return raw
	}
	return adaptSchema(schema, notes)
}

// adaptSchema recursively adapts a JSON schema to the subset Gemini accepts.
//
// Tool schemas are JSON Schema, but Gemini takes only a limited subset of it
// (its schema object follows OpenAPI 3.0's, which predates JSON Schema
// alignment) and rejects anything outside that subset. Three constructs need
// adapting:
//
//   - Keys it has no field for, namely additionalProperties and vendor
//     extensions (x- prefixed, such as the x-mcp-header an MCP server may
//     attach to its tool properties), are dropped.
//   - A union type (a list of types) becomes the equivalent anyOf.
//   - enum members must be strings, so an enum holding other values is dropped
//     and its constraint lost.
//
// notes collects one description per kind of adaptation applied.
func adaptSchema(schema any, notes map[string]bool) any {
	obj, ok := schema.(map[string]any)
	if !ok {
		return schema
	}
	result := make(map[string]any, len(obj))
	// Sorted, so a type rewritten as anyOf settles a clash with an explicit
	// anyOf the same way every time.
	for _, key := range slices.Sorted(maps.Keys(obj)) {
		value := obj[key]
		dict, isDict := value.(map[string]any)
		list, isList := value.([]any)
		switch {
		case unsupportedSchemaKey(key):
			notes[fmt.Sprintf("dropped unsupported key '%s'", key)] = true
		case key == "properties" && isDict:
			// Keys here name the tool's parameters rather than schema keywords,
			// so adapt the values but keep every key.
			props := make(map[string]any, len(dict))
			for name, prop := range dict {
				props[name] = adaptSchema(prop, notes)
			}
			result[key] = props
		case key == "type" && isList:
			notes["converted a union 'type' to 'anyOf'"] = true
			result["anyOf"] = unionAsAnyOf(list)
		case key == "enum" && isList && !allStrings(list):
			notes["dropped a non-string 'enum', losing its constraint"] = true
		case isDict:
			result[key] = adaptSchema(dict, notes)
		case isList:
			result[key] = adaptEach(list, notes)
		default:
			result[key] = value
		}
	}
	return result
}

// unsupportedSchemaKey reports whether Gemini has no field for a schema key:
// additionalProperties, and the x- prefixed vendor extensions.
func unsupportedSchemaKey(key string) bool {
	return key == "additionalProperties" || strings.HasPrefix(key, "x-")
}

// unionAsAnyOf renders a union type as the equivalent anyOf.
func unionAsAnyOf(members []any) []any {
	anyOf := make([]any, 0, len(members))
	for _, member := range members {
		anyOf = append(anyOf, map[string]any{"type": member})
	}
	return anyOf
}

// adaptEach adapts every item of a list of schemas.
func adaptEach(items []any, notes map[string]bool) []any {
	out := make([]any, 0, len(items))
	for _, item := range items {
		out = append(out, adaptSchema(item, notes))
	}
	return out
}

// allStrings reports whether every member of an enum is a string.
func allStrings(members []any) bool {
	for _, m := range members {
		if _, ok := m.(string); !ok {
			return false
		}
	}
	return true
}

// warnSchemaAdaptations reports the tool schemas that had to be adapted before
// Gemini would accept them, given each adaptation and the tools it applied to.
//
// Each (adaptation, tool) pair is reported once, so a stable tool set warns on
// the first inference rather than on all of them, while a tool added later
// still gets reported.
func (a *Adapter) warnSchemaAdaptations(adapted map[string][]string) {
	a.warnedMu.Lock()
	defer a.warnedMu.Unlock()
	var lines []string
	for _, note := range slices.Sorted(maps.Keys(adapted)) {
		var fresh []string
		for _, tool := range adapted[note] {
			key := schemaAdaptation{note: note, tool: tool}
			if a.warnedSchemaAdaptations[key] {
				continue
			}
			if a.warnedSchemaAdaptations == nil {
				a.warnedSchemaAdaptations = map[schemaAdaptation]bool{}
			}
			a.warnedSchemaAdaptations[key] = true
			fresh = append(fresh, tool)
		}
		if len(fresh) == 0 {
			continue
		}
		listed := strings.Join(fresh[:min(len(fresh), 3)], ", ")
		if len(fresh) > 3 {
			listed += fmt.Sprintf(" and %d more", len(fresh)-3)
		}
		lines = append(lines, note+": "+listed)
	}
	if len(lines) > 0 {
		slog.Warn("adapted tool schemas Gemini does not accept as-is",
			"adaptations", strings.Join(lines, "; "))
	}
}

// MessagesForLogging implements adapter.LLMAdapter.
func (a *Adapter) MessagesForLogging(convo *frames.LLMContext) []map[string]any {
	params, err := a.LLMInvocationParams(convo, adapter.Options{})
	if err != nil {
		return nil
	}
	// The contents were built for this call and share nothing with the
	// conversation, so the opaque and bulky values are elided in place.
	for _, c := range params.Contents {
		for _, part := range contentParts(c) {
			if blob, ok := part[keyInlineData].(map[string]any); ok {
				blob[keyData] = "..."
			}
			if _, ok := part[keyThoughtSignature]; ok {
				part[keyThoughtSignature] = "..."
			}
		}
	}
	return params.Contents
}
