package frames

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
)

// Role identifies who authored a conversation message.
type Role string

const (
	// RoleSystem is the system prompt that frames the assistant's behavior.
	RoleSystem Role = "system"
	// RoleUser is a message from the user.
	RoleUser Role = "user"
	// RoleAssistant is a message from the assistant.
	RoleAssistant Role = "assistant"
	// RoleDeveloper is an out-of-band instruction to the model. It carries the
	// results an asynchronous tool reports after its turn has moved on (see
	// AsyncToolMessage). Providers without a developer role take it as a user
	// message.
	RoleDeveloper Role = "developer"
)

// Tool is a function the model may call. Parameters is a JSON-Schema object
// (`{"type":"object","properties":{…},"required":[…]}`) describing the
// arguments the tool accepts.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
	// Handler runs the tool. Setting it means the tool carries its own
	// implementation: the LLM service registers it when the toolset is
	// advertised and drops it again when the toolset stops advertising it, so
	// what the model can call and what actually answers stay the same thing.
	//
	// A tool with no Handler is advertise-only, and something must have called
	// RegisterFunction for it. Registering explicitly always wins: a handler
	// registered by hand is never replaced by a tool's own, and never dropped.
	//
	// It is typed as any because the handler signature belongs to the LLM
	// service, which is built on this package. Set it to an llm.FunctionCallHandler;
	// anything else is reported and ignored.
	Handler any
}

// AdapterType names the wire format a custom tool is written in. Several
// providers share one: everything speaking OpenAI's API reads a tool written
// for OpenAI, whether it arrives over chat completions, the Responses API or a
// realtime session.
type AdapterType string

const (
	// AdapterTypeOpenAI is OpenAI's tool format: chat completions, the Responses
	// API and the Realtime API alike.
	AdapterTypeOpenAI AdapterType = "openai"
	// AdapterTypeGemini is Gemini's tool format.
	AdapterTypeGemini AdapterType = "gemini"
)

// ToolsSchema is the toolset a conversation advertises.
//
// Standard holds the tools every provider is offered, described the same way
// for all of them. Custom holds tools written in one provider's own format, for
// the ones no common description fits: a hosted search tool, say, which the
// provider implements itself rather than calling back for. A provider is only
// ever sent the custom tools written for its own format, so a conversation
// carrying them is still usable with every other provider.
//
// Each custom tool must be the tool type its adapter's package defines. Not
// every provider takes them: Anthropic has no custom tools, and anything under
// a key it does not read is left out.
type ToolsSchema struct {
	Standard []Tool
	Custom   map[AdapterType][]any
}

// clone returns a schema sharing nothing with this one, so a later change to
// either is invisible to the other.
func (s ToolsSchema) clone() ToolsSchema {
	out := ToolsSchema{Standard: append([]Tool(nil), s.Standard...)}
	if len(s.Custom) > 0 {
		out.Custom = make(map[AdapterType][]any, len(s.Custom))
		for k, v := range s.Custom {
			out.Custom[k] = append([]any(nil), v...)
		}
	}
	return out
}

// CustomFor returns the custom tools written for the named format, or nil if
// there are none.
func (s ToolsSchema) CustomFor(t AdapterType) []any { return s.Custom[t] }

// ToolCall is a request from the model to invoke a tool. Args is the raw JSON
// arguments the model produced.
type ToolCall struct {
	ID   string
	Name string
	Args json.RawMessage
}

// ToolResult is the outcome of a tool invocation, paired to a ToolCall by ID.
type ToolResult struct {
	ID      string
	Name    string
	Content string
}

// Message is a single conversation turn. A plain turn carries Text; an assistant
// turn that invoked tools also carries ToolCalls; a turn returning tool outputs
// carries ToolResults.
//
// A message may instead be written in one provider's own format, which is what
// LLM and Native carry. See [NewLLMSpecificMessage].
type Message struct {
	Role Role
	Text string
	// ToolCalls is set on an assistant message that requested tool calls.
	ToolCalls []ToolCall
	// ToolResults is set on a message returning the outputs of tool calls.
	ToolResults []ToolResult
	// LLM names the provider this message is written for. When it is set the
	// message is that provider's own and no other's: only its adapter sends it,
	// and every other adapter leaves it out (see LLMContext.MessagesFor).
	LLM string
	// Native is the message as that provider's API takes it, in the type that
	// provider's adapter defines. It is only read when LLM is set, and the
	// adapter it names is the only thing that knows how to read it.
	Native any
}

// NewLLMSpecificMessage builds a message written in one provider's own format,
// for something the universal conversation has no representation for: a
// reasoning block a model wants handed back to it, a content type only one
// provider takes.
//
// llm is the identifier that provider's adapter answers to
// (adapter.LLMAdapter.IDForLLMSpecificMessages), and native is the message in
// the type that adapter defines. A conversation carrying one can still be sent
// to any provider: the ones it was not written for leave it out.
func NewLLMSpecificMessage(llm string, native any) Message {
	return Message{LLM: llm, Native: native}
}

// IsLLMSpecific reports whether the message is written in one provider's own
// format rather than in the universal one.
func (m Message) IsLLMSpecific() bool { return m.LLM != "" }

// clone returns a message that shares nothing with this one. Copying the struct
// alone would copy the slice headers, leaving both messages pointing at one
// array: a tool result updated in place would then be rewritten under whoever
// holds the other copy, mid-read.
//
// Native is copied as it stands. What it points at belongs to the provider's
// adapter, which is the only thing that can read it, so there is nothing here
// that could copy it; treat a native message as written once and not edited
// afterwards.
func (m Message) clone() Message {
	if len(m.ToolCalls) > 0 {
		m.ToolCalls = append([]ToolCall(nil), m.ToolCalls...)
	}
	if len(m.ToolResults) > 0 {
		m.ToolResults = append([]ToolResult(nil), m.ToolResults...)
	}
	return m
}

// cloneMessages returns a deep copy of msgs, one that no later in-place update
// can reach.
func cloneMessages(msgs []Message) []Message {
	out := make([]Message, len(msgs))
	for i, m := range msgs {
		out[i] = m.clone()
	}
	return out
}

// LLMContext holds the conversation so far: a system prompt plus the running
// list of user and assistant messages. The user and assistant aggregators
// append to a shared context as the conversation proceeds, and the LLM service
// reads it to generate each response. It is safe for concurrent use.
type LLMContext struct {
	mu         sync.Mutex
	system     string
	summary    string // rolling summary of compacted older turns; empty until the first Compact
	recall     string // transient retrieved context (e.g. long-term memories) for the next generation
	messages   []Message
	tools      ToolsSchema
	toolChoice ToolChoice
}

// summaryHeader introduces the rolling summary appended to the system prompt
// once older turns have been compacted away by Compact.
const summaryHeader = "Summary of the earlier conversation:"

// NewLLMContext builds a context with the given system prompt.
func NewLLMContext(system string) *LLMContext {
	return &LLMContext{system: system}
}

// System returns the system prompt the LLM should run with. Once older turns
// have been compacted (see Compact), the rolling summary is appended so the
// model retains that history even though the messages themselves are gone.
func (c *LLMContext) System() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.systemLocked()
}

// systemLocked composes the base system prompt with the rolling summary, any
// transient recalled context, and whatever the LLM service adds for its own
// built-in tools. The caller must hold c.mu.
func (c *LLMContext) systemLocked() string {
	parts := make([]string, 0, 3)
	if c.system != "" {
		parts = append(parts, c.system)
	}
	if c.summary != "" {
		parts = append(parts, summaryHeader+"\n"+c.summary)
	}
	if c.recall != "" {
		parts = append(parts, c.recall)
	}
	return strings.Join(parts, "\n\n")
}

// SystemParts returns the system prompt split at its first volatile point:
// stable is the part that survives from one turn to the next, volatile the
// transient recalled context that a memory service replaces every turn.
// Concatenated with a blank line between them they are exactly System().
//
// The split exists for prompt caching. A provider that caches a prefix of the
// prompt can only reuse it while that prefix is byte-identical, so a breakpoint
// placed after the recalled context would be rewritten every turn and never
// read back. Caching stable and leaving volatile outside the breakpoint keeps
// the bulk of the prompt reusable.
func (c *LLMContext) SystemParts() (stable, volatile string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	parts := make([]string, 0, 2)
	if c.system != "" {
		parts = append(parts, c.system)
	}
	if c.summary != "" {
		parts = append(parts, summaryHeader+"\n"+c.summary)
	}
	return strings.Join(parts, "\n\n"), c.recall
}

// Summary returns the rolling summary of compacted older turns, or "" if the
// conversation has not been compacted yet.
func (c *LLMContext) Summary() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.summary
}

// SetRecall sets transient retrieved context — typically long-term memories
// surfaced by a memory service — that is folded into the system prompt for
// subsequent generations, replacing any previous value. The text is used
// verbatim, so include any framing (e.g. a "recalled memories" header) in it. A
// memory processor refreshes it each turn; pass "" to clear it.
func (c *LLMContext) SetRecall(recall string) {
	c.mu.Lock()
	c.recall = recall
	c.mu.Unlock()
}

// Recall returns the transient retrieved context folded into the system prompt,
// or "" if none is set.
func (c *LLMContext) Recall() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.recall
}

// SetSystem replaces the system prompt. Used to switch the assistant's behavior
// mid-session (the next generation picks up the new prompt).
func (c *LLMContext) SetSystem(system string) {
	c.mu.Lock()
	c.system = system
	c.mu.Unlock()
}

// Tools returns a copy of the standard tools the model may call: the ones
// described the same way for every provider. For the whole toolset, custom
// tools included, see ToolsSchema.
//
// It is what the conversation advertises. A tool the LLM service implements
// itself is not here: that belongs to the service, and lives on the adapter it
// converts through, so a conversation shared by two services does not offer
// each of them the other's.
func (c *LLMContext) Tools() []Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Tool(nil), c.tools.Standard...)
}

// ToolsSchema returns a copy of the whole toolset the conversation advertises,
// the tools written in one provider's own format included. It is what an
// adapter reads.
func (c *LLMContext) ToolsSchema() ToolsSchema {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.tools.clone()
}

// SetTools replaces the set of tools the model may call.
//
// This mutates the context and notifies nobody. A text LLM reads the context on
// its next run and so picks the change up, but a realtime (speech-to-speech)
// service is generating continuously and will keep offering the old toolset. To
// change tools on a running pipeline, push an [LLMSetToolsFrame] instead: the
// aggregator applies it here and forwards it downstream so realtime services are
// told. Use this directly only to seed the toolset before the pipeline starts.
func (c *LLMContext) SetTools(tools []Tool) {
	c.mu.Lock()
	c.tools = ToolsSchema{Standard: tools}
	c.mu.Unlock()
}

// SetToolsSchema replaces the whole toolset, the tools written in one
// provider's own format included. The same caveat as SetTools applies: on a
// running pipeline push an [LLMSetToolsFrame] so realtime services learn of the
// change.
func (c *LLMContext) SetToolsSchema(schema ToolsSchema) {
	c.mu.Lock()
	c.tools = schema.clone()
	c.mu.Unlock()
}

// ToolChoice returns whether the model may or must call a tool. It is
// ToolChoiceAuto unless set.
func (c *LLMContext) ToolChoice() ToolChoice {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.toolChoice == "" {
		return ToolChoiceAuto
	}
	return c.toolChoice
}

// SetToolChoice sets whether the model may or must call a tool. The same caveat
// as SetTools applies: on a running pipeline push an [LLMSetToolChoiceFrame] so
// realtime services learn of the change.
func (c *LLMContext) SetToolChoice(choice ToolChoice) {
	c.mu.Lock()
	c.toolChoice = choice
	c.mu.Unlock()
}

// SetMessages replaces the conversation messages, in contrast to the Add methods
// which append. The system prompt, tools and rolling summary are left alone. On
// a running pipeline push an [LLMMessagesUpdateFrame] so the replacement is
// ordered against the conversation rather than racing an in-flight generation.
func (c *LLMContext) SetMessages(messages []Message) {
	c.mu.Lock()
	c.messages = cloneMessages(messages)
	c.mu.Unlock()
}

// AddUserMessage appends a user message.
func (c *LLMContext) AddUserMessage(text string) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: RoleUser, Text: text})
	c.mu.Unlock()
}

// AddAssistantMessage appends an assistant message.
func (c *LLMContext) AddAssistantMessage(text string) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: RoleAssistant, Text: text})
	c.mu.Unlock()
}

// ReplaceLastAssistantText replaces the text of the most recent message when it
// is a plain assistant message (one carrying no tool calls or results),
// reporting whether it did. The assistant aggregator uses it to keep an
// in-progress assistant turn updated with the words spoken so far, so that an
// interruption leaves exactly the spoken text in the context.
func (c *LLMContext) ReplaceLastAssistantText(text string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if n := len(c.messages); n > 0 {
		m := &c.messages[n-1]
		if m.Role == RoleAssistant && len(m.ToolCalls) == 0 && len(m.ToolResults) == 0 {
			m.Text = text
			return true
		}
	}
	return false
}

// AddMessage appends a message as it stands. The assistant aggregator uses it
// for the developer messages an asynchronous tool call reports its progress
// through (see AsyncToolMessage).
func (c *LLMContext) AddMessage(m Message) {
	c.mu.Lock()
	c.messages = append(c.messages, m)
	c.mu.Unlock()
}

// AddAssistantToolCall appends an assistant message requesting a single tool
// call. One message per call, each followed straight away by that call's result
// message, is what keeps every tool-use block adjacent to the tool-result block
// answering it, so the conversation is a valid one at every moment rather than
// only once a turn has finished.
func (c *LLMContext) AddAssistantToolCall(call ToolCall) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: RoleAssistant, ToolCalls: []ToolCall{call}})
	c.mu.Unlock()
}

// AddToolResult appends a message returning the output of a single tool call.
// It is written as soon as the call starts, carrying a placeholder, and updated
// in place by UpdateToolResult once the call reports.
func (c *LLMContext) AddToolResult(r ToolResult) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: RoleUser, ToolResults: []ToolResult{r}})
	c.mu.Unlock()
}

// UpdateToolResult rewrites the content of the result message belonging to
// toolCallID, reporting whether it found one. Updating in place, rather than
// appending, is what stops a late result from landing after messages that were
// added while the call was running, which would separate it from the tool call
// it answers.
func (c *LLMContext) UpdateToolResult(toolCallID, content string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.messages {
		for j := range c.messages[i].ToolResults {
			if c.messages[i].ToolResults[j].ID == toolCallID {
				c.messages[i].ToolResults[j].Content = content
				return true
			}
		}
	}
	return false
}

// Messages returns a copy of the conversation messages, deep enough that a
// later update cannot reach it. A tool result is rewritten in place once its
// call reports, so a caller reading a shallow copy would be reading an array
// being written under it.
func (c *LLMContext) Messages() []Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return cloneMessages(c.messages)
}

// MessagesFor returns the messages to send to the named provider: every
// universal one, plus the provider's own, and none written for a different
// provider. It is what an adapter reads rather than Messages, so a conversation
// carrying one provider's native messages can still be sent to another.
//
// Leaving a message out is reported: a message written for a provider that
// never sees it is almost always a mistake, and it would otherwise go missing
// in silence.
func (c *LLMContext) MessagesFor(llm string) []Message {
	c.mu.Lock()
	msgs := cloneMessages(c.messages)
	c.mu.Unlock()

	out := make([]Message, 0, len(msgs))
	dropped := 0
	for _, m := range msgs {
		if m.IsLLMSpecific() && m.LLM != llm {
			dropped++
			continue
		}
		out = append(out, m)
	}
	if dropped > 0 {
		slog.Error("leaving out conversation messages written for another provider",
			"llm", llm, "messages", dropped)
	}
	return out
}

// EstimatedTokens is a rough estimate of the context's size in tokens, used to
// decide when to compact. It approximates four characters per token across the
// system prompt, the rolling summary, and every message.
//
// A message in a provider's own format is counted by what its rendering
// measures, since nothing here can read the format itself.
func (c *LLMContext) EstimatedTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := len(c.system) + len(c.summary) + len(c.recall)
	for _, m := range c.messages {
		if m.IsLLMSpecific() {
			n += len(fmt.Sprint(m.Native))
			continue
		}
		n += len(m.Text)
		for _, tc := range m.ToolCalls {
			n += len(tc.Name) + len(tc.Args)
		}
		for _, tr := range m.ToolResults {
			n += len(tr.Name) + len(tr.Content)
		}
	}
	return n / 4
}

// Compact shrinks a long conversation: it drops the oldest messages beyond the
// keepRecent most recent — cutting on a clean user-turn boundary so the
// preserved tail stays a valid message list — and folds them into the rolling
// summary, which System then appends to the prompt. summarize turns the prior
// summary and the dropped messages into the new summary; it is invoked WITHOUT
// the context lock held, so it may call out to an LLM. Compact reports whether
// it compacted anything.
//
// Compact only ever removes a prefix, and only the summary (not the messages)
// carries the dropped history forward, so messages appended at the tail while
// summarize runs are preserved. It must not be run concurrently with itself on
// the same context.
func (c *LLMContext) Compact(
	ctx context.Context,
	keepRecent int,
	summarize func(ctx context.Context, prior string, dropped []Message) (string, error),
) (bool, error) {
	c.mu.Lock()
	cut := cleanCut(c.messages, len(c.messages)-keepRecent)
	if cut <= 0 {
		c.mu.Unlock()
		return false, nil
	}
	dropped := cloneMessages(c.messages[:cut])
	prior := c.summary
	c.mu.Unlock()

	next, err := summarize(ctx, prior, dropped)
	if err != nil {
		return false, err
	}
	if next = strings.TrimSpace(next); next == "" {
		return false, nil
	}

	c.mu.Lock()
	// Only appends can have happened since we read the prefix, so the first cut
	// messages are still the ones we summarized; drop exactly those.
	if cut <= len(c.messages) {
		c.messages = append([]Message(nil), c.messages[cut:]...)
	}
	c.summary = next
	c.mu.Unlock()
	return true, nil
}

// cleanCut returns the largest index i in [1, limit] at which msgs[i] begins a
// new user turn (a plain user message, not a tool result), or 0 if there is
// none. Cutting there keeps msgs[i:] a valid standalone list — it starts with a
// user message and never orphans a tool result from its tool call.
func cleanCut(msgs []Message, limit int) int {
	if limit > len(msgs)-1 {
		limit = len(msgs) - 1
	}
	for i := limit; i >= 1; i-- {
		m := msgs[i]
		// A message in a provider's own format is not a turn this can read, so it
		// is never a boundary to cut on.
		if m.IsLLMSpecific() {
			continue
		}
		if m.Role == RoleUser && len(m.ToolResults) == 0 {
			return i
		}
	}
	return 0
}

// LLMContextFrame carries the conversation context to the LLM service to
// trigger a response. It is a data frame.
type LLMContextFrame struct {
	BaseDataFrame
	// Context is the conversation to generate a response from.
	Context *LLMContext
}

// NewLLMContextFrame builds an LLMContextFrame.
func NewLLMContextFrame(ctx *LLMContext) *LLMContextFrame {
	return &LLMContextFrame{
		BaseDataFrame: NewBaseDataFrame("LLMContextFrame"),
		Context:       ctx,
	}
}

// String implements fmt.Stringer.
func (f *LLMContextFrame) String() string {
	n := 0
	if f.Context != nil {
		n = len(f.Context.Messages())
	}
	return fmt.Sprintf("%s(messages: %d)", f.Name(), n)
}

// LLMRunFrame instructs the LLM service to process the current context and
// generate a response. Queue it to make the bot speak first at the start of a
// session, or to re-run after editing the context. It carries no data — the
// user aggregator runs its current shared context. It is a data frame.
type LLMRunFrame struct {
	BaseDataFrame
}

// NewLLMRunFrame builds an LLMRunFrame.
func NewLLMRunFrame() *LLMRunFrame {
	return &LLMRunFrame{BaseDataFrame: NewBaseDataFrame("LLMRunFrame")}
}

// LLMSetToolsFrame changes the set of tools advertised to the model
// mid-conversation. The context aggregator applies it to the shared context and
// forwards it downstream: a text LLM picks the change up on its next run, but a
// realtime (speech-to-speech) service is generating continuously and would never
// see it, so it must be told. Always change tools through this frame rather than
// calling LLMContext.SetTools directly, or realtime services will keep using the
// old toolset. It is a data frame, so the change is ordered against the
// surrounding conversation instead of racing an in-flight generation.
type LLMSetToolsFrame struct {
	BaseDataFrame
	// Tools is the new toolset; nil or empty clears the tools.
	Tools []Tool
}

// NewLLMSetToolsFrame builds an LLMSetToolsFrame advertising tools.
func NewLLMSetToolsFrame(tools []Tool) *LLMSetToolsFrame {
	return &LLMSetToolsFrame{
		BaseDataFrame: NewBaseDataFrame("LLMSetToolsFrame"),
		Tools:         tools,
	}
}

// String implements fmt.Stringer.
func (f *LLMSetToolsFrame) String() string {
	return fmt.Sprintf("%s(tools: %d)", f.Name(), len(f.Tools))
}

// ToolChoice tells the model whether it may or must call a tool.
type ToolChoice string

const (
	// ToolChoiceAuto lets the model decide whether to call a tool.
	ToolChoiceAuto ToolChoice = "auto"
	// ToolChoiceNone forbids tool calls.
	ToolChoiceNone ToolChoice = "none"
	// ToolChoiceRequired requires the model to call a tool.
	ToolChoiceRequired ToolChoice = "required"
)

// LLMSetToolChoiceFrame changes whether the model may or must call a tool. Like
// LLMSetToolsFrame it is applied to the shared context and forwarded downstream
// so realtime services learn of the change. It is a data frame.
type LLMSetToolChoiceFrame struct {
	BaseDataFrame
	// ToolChoice is the new setting.
	ToolChoice ToolChoice
}

// NewLLMSetToolChoiceFrame builds an LLMSetToolChoiceFrame.
func NewLLMSetToolChoiceFrame(choice ToolChoice) *LLMSetToolChoiceFrame {
	return &LLMSetToolChoiceFrame{
		BaseDataFrame: NewBaseDataFrame("LLMSetToolChoiceFrame"),
		ToolChoice:    choice,
	}
}

// String implements fmt.Stringer.
func (f *LLMSetToolChoiceFrame) String() string {
	return fmt.Sprintf("%s(choice: %s)", f.Name(), f.ToolChoice)
}

// LLMMessagesUpdateFrame replaces the conversation messages in the shared
// context, in contrast to LLMMessagesAppendFrame which adds to them. Use it to
// swap the conversation wholesale — restoring a saved session, or resetting the
// conversation without rebuilding the pipeline. It is a data frame, so the
// replacement is ordered against the surrounding conversation.
type LLMMessagesUpdateFrame struct {
	BaseDataFrame
	// Messages replaces the context's current messages.
	Messages []Message
	// RunLLM reports whether the LLM should run on the updated context.
	RunLLM bool
}

// NewLLMMessagesUpdateFrame builds an LLMMessagesUpdateFrame.
func NewLLMMessagesUpdateFrame(messages []Message) *LLMMessagesUpdateFrame {
	return &LLMMessagesUpdateFrame{
		BaseDataFrame: NewBaseDataFrame("LLMMessagesUpdateFrame"),
		Messages:      messages,
	}
}

// String implements fmt.Stringer.
func (f *LLMMessagesUpdateFrame) String() string {
	return fmt.Sprintf("%s(messages: %d)", f.Name(), len(f.Messages))
}

// Compile-time interface checks.
var (
	_ DataFrame = (*LLMContextFrame)(nil)
	_ DataFrame = (*LLMRunFrame)(nil)
	_ DataFrame = (*LLMSetToolsFrame)(nil)
	_ DataFrame = (*LLMSetToolChoiceFrame)(nil)
	_ DataFrame = (*LLMMessagesUpdateFrame)(nil)
)
