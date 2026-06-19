package frames

import (
	"fmt"
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

// Message is a single conversation turn.
type Message struct {
	Role Role
	Text string
}

// LLMContext holds the conversation so far: a system prompt plus the running
// list of user and assistant messages. The user and assistant aggregators
// append to a shared context as the conversation proceeds, and the LLM service
// reads it to generate each response. It is safe for concurrent use.
type LLMContext struct {
	mu       sync.Mutex
	system   string
	messages []Message
}

// NewLLMContext builds a context with the given system prompt.
func NewLLMContext(system string) *LLMContext {
	return &LLMContext{system: system}
}

// System returns the system prompt.
func (c *LLMContext) System() string { return c.system }

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

// Messages returns a copy of the conversation messages.
func (c *LLMContext) Messages() []Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Message, len(c.messages))
	copy(out, c.messages)
	return out
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

// Compile-time interface check.
var _ DataFrame = (*LLMContextFrame)(nil)
