package frames

import (
	"context"
	"encoding/json"
	"fmt"
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
)

// Tool is a function the model may call. Parameters is a JSON-Schema object
// (`{"type":"object","properties":{…},"required":[…]}`) describing the
// arguments the tool accepts.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

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
	IsError bool
}

// Message is a single conversation turn. A plain turn carries Text; an assistant
// turn that invoked tools also carries ToolCalls; a turn returning tool outputs
// carries ToolResults.
type Message struct {
	Role Role
	Text string
	// ToolCalls is set on an assistant message that requested tool calls.
	ToolCalls []ToolCall
	// ToolResults is set on a message returning the outputs of tool calls.
	ToolResults []ToolResult
}

// LLMContext holds the conversation so far: a system prompt plus the running
// list of user and assistant messages. The user and assistant aggregators
// append to a shared context as the conversation proceeds, and the LLM service
// reads it to generate each response. It is safe for concurrent use.
type LLMContext struct {
	mu       sync.Mutex
	system   string
	summary  string // rolling summary of compacted older turns; empty until the first Compact
	messages []Message
	tools    []Tool
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

// systemLocked composes the base system prompt with the rolling summary. The
// caller must hold c.mu.
func (c *LLMContext) systemLocked() string {
	if c.summary == "" {
		return c.system
	}
	if c.system == "" {
		return summaryHeader + "\n" + c.summary
	}
	return c.system + "\n\n" + summaryHeader + "\n" + c.summary
}

// Summary returns the rolling summary of compacted older turns, or "" if the
// conversation has not been compacted yet.
func (c *LLMContext) Summary() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.summary
}

// SetSystem replaces the system prompt. Used to switch the assistant's behavior
// mid-session (the next generation picks up the new prompt).
func (c *LLMContext) SetSystem(system string) {
	c.mu.Lock()
	c.system = system
	c.mu.Unlock()
}

// Tools returns a copy of the tools the model may call.
func (c *LLMContext) Tools() []Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Tool, len(c.tools))
	copy(out, c.tools)
	return out
}

// SetTools replaces the set of tools the model may call. Used alongside
// SetSystem to switch the available toolset mid-session.
func (c *LLMContext) SetTools(tools []Tool) {
	c.mu.Lock()
	c.tools = tools
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

// AddAssistantToolCalls appends an assistant message carrying optional preamble
// text and the tool calls the model requested in the same turn.
func (c *LLMContext) AddAssistantToolCalls(text string, calls []ToolCall) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: RoleAssistant, Text: text, ToolCalls: calls})
	c.mu.Unlock()
}

// AddToolResults appends a user message returning the outputs of one or more
// tool calls. The results of all calls in an assistant turn belong in a single
// message.
func (c *LLMContext) AddToolResults(results []ToolResult) {
	c.mu.Lock()
	c.messages = append(c.messages, Message{Role: RoleUser, ToolResults: results})
	c.mu.Unlock()
}

// Messages returns a copy of the conversation messages.
func (c *LLMContext) Messages() []Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Message, len(c.messages))
	copy(out, c.messages)
	return out
}

// EstimatedTokens is a rough estimate of the context's size in tokens, used to
// decide when to compact. It approximates four characters per token across the
// system prompt, the rolling summary, and every message.
func (c *LLMContext) EstimatedTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := len(c.system) + len(c.summary)
	for _, m := range c.messages {
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
	dropped := append([]Message(nil), c.messages[:cut]...)
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
		if msgs[i].Role == RoleUser && len(msgs[i].ToolResults) == 0 {
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

// Compile-time interface checks.
var (
	_ DataFrame = (*LLMContextFrame)(nil)
	_ DataFrame = (*LLMRunFrame)(nil)
)
