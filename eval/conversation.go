package eval

import (
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/gojargo/jargo/internal/pytext"
)

// The roles and keys of a conversation file's messages.
//
//nolint:misspell // the file format spells its key "cancelled"
const (
	// A conversation between a user and a bot, as a conversation file holds
	// it.
	//
	// A conversation file is a JSON list of messages, in the order they
	// happened:
	//
	//   - {"role": "user", "content": ...}: what the user said in one turn.
	//   - {"role": "assistant", "content": ...}: everything the bot said
	//     between two user turns, however turn detection or a function call
	//     split it. A turn in which the bot said nothing is not a turn, so the
	//     Nth assistant message is bot turn N, the turn a metric's verdict
	//     names.
	//   - {"role": "tool_call", "name": ..., "arguments": {...}, "cancelled": ...}:
	//     a function the bot called, placed before the bot turn it was made in.
	//
	// The file holds what each side said and the calls the bot made, which is
	// what the judge reads. It has no system prompt, no tool results, and no
	// timing. The harness saves one for every simulation run, and
	// ConversationObserver records one from a bot's own pipeline. For example:
	//
	//	[
	//	  {"role": "assistant", "content": "Hi, how can I help?"},
	//	  {"role": "user", "content": "A table for four on Friday, please."},
	//	  {"role": "tool_call", "name": "check_availability",
	//	   "arguments": {"day": "Friday", "party_size": 4}, "cancelled": false},
	//	  {"role": "assistant", "content": "Let me check. Friday at seven works."}
	//	]

	// ConversationRoleUser is a user turn.
	ConversationRoleUser = "user"
	// ConversationRoleAssistant is a bot turn.
	ConversationRoleAssistant = "assistant"
	// ConversationRoleToolCall is a function the bot called.
	ConversationRoleToolCall = "tool_call"

	keyRole      = "role"
	keyContent   = "content"
	keyArguments = "arguments"
)

// Conversation builds a conversation's messages as the conversation happens, in
// the format a conversation file holds (see ConversationRoleUser).
//
// The user's turns and the bot's calls are added as they come. What the bot says
// gathers into its current turn, which the next user turn, or EndBotTurn,
// closes. A Conversation is safe for concurrent use.
type Conversation struct {
	mu       sync.Mutex
	messages []map[string]any
	botSaid  []string
}

// NewConversation is an empty conversation.
func NewConversation() *Conversation { return &Conversation{} }

// Messages are the messages so far, without the bot's turn in progress.
func (c *Conversation) Messages() []map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]map[string]any, 0, len(c.messages))
	for _, m := range c.messages {
		out = append(out, maps.Clone(m))
	}
	return out
}

// UserSpokeLast reports whether a user turn is the last thing said, with nothing
// from the bot since.
func (c *Conversation) UserSpokeLast() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.botSaid) > 0 {
		return false
	}
	for _, m := range slices.Backward(c.messages) {
		role := m[keyRole]
		if role != ConversationRoleToolCall {
			return role == ConversationRoleUser
		}
	}
	return false
}

// AddUserMessage adds a user turn, which ends the bot's turn; empty text only
// ends the bot's turn.
func (c *Conversation) AddUserMessage(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.endBotTurnLocked()
	if text = pytext.Strip(text); text != "" {
		c.messages = append(c.messages, map[string]any{keyRole: ConversationRoleUser, keyContent: text})
	}
}

// AddAssistantMessage adds a segment of what the bot said to its current turn.
func (c *Conversation) AddAssistantMessage(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if text = pytext.Strip(text); text != "" {
		c.botSaid = append(c.botSaid, text)
	}
}

// AddToolCall adds a function the bot called.
func (c *Conversation) AddToolCall(name string, arguments map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	args := maps.Clone(arguments)
	if args == nil {
		args = map[string]any{}
	}
	c.messages = append(c.messages, map[string]any{
		keyRole: ConversationRoleToolCall, keyName: name, keyArguments: args, keyCanceled: false,
	})
}

// CancelToolCall marks the bot's latest call to name that is not canceled yet
// as canceled.
func (c *Conversation) CancelToolCall(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, m := range slices.Backward(c.messages) {
		if m[keyRole] == ConversationRoleToolCall && m[keyName] == name && m[keyCanceled] == false {
			m[keyCanceled] = true
			return
		}
	}
}

// EndBotTurn ends the bot's turn: what it said since the user last spoke
// becomes one message.
func (c *Conversation) EndBotTurn() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.endBotTurnLocked()
}

func (c *Conversation) endBotTurnLocked() {
	text := strings.Join(c.botSaid, " ")
	c.botSaid = nil
	if text != "" {
		c.messages = append(c.messages, map[string]any{keyRole: ConversationRoleAssistant, keyContent: text})
	}
}
