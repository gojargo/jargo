package events

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
)

// ItemContent is one piece of a conversation item's content.
type ItemContent struct {
	// Type is "text", "audio", "input_text", "input_audio", "input_image",
	// "output_text" or "output_audio".
	Type string `json:"type"`
	// Text is the text of a text piece.
	Text string `json:"text,omitempty"`
	// Audio is base64-encoded audio.
	Audio string `json:"audio,omitempty"`
	// Transcript is the transcript of an audio piece.
	Transcript string `json:"transcript,omitempty"`
	// ImageURL is an image as a base64 data URI.
	ImageURL string `json:"image_url,omitempty"`
	// Detail is how much the model reads out of an image: "auto", "low" or
	// "high".
	Detail string `json:"detail,omitempty"`
}

// The conversation item types.
const (
	ItemMessage            = "message"
	ItemFunctionCall       = "function_call"
	ItemFunctionCallOutput = "function_call_output"
)

// ConversationItem is one item of the session's conversation: a message, a
// function call, or a function call's output.
type ConversationItem struct {
	// ID identifies the item; NewConversationItem generates one.
	ID     string `json:"id"`
	Object string `json:"object,omitempty"`
	// Type is ItemMessage, ItemFunctionCall or ItemFunctionCallOutput.
	Type string `json:"type"`
	// Status is "completed", "in_progress" or "incomplete".
	Status string `json:"status,omitempty"`
	// Role is "user", "assistant" or "system", for a message.
	Role    string        `json:"role,omitempty"`
	Content []ItemContent `json:"content,omitempty"`
	// CallID, Name and Arguments describe a function call; CallID and Output
	// its output.
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

// NewConversationItem builds an item of the given type with a fresh id.
func NewConversationItem(itemType string) ConversationItem {
	return ConversationItem{ID: strings.ReplaceAll(uuid.NewString(), "-", ""), Type: itemType}
}

// RealtimeConversation is the conversation a session holds.
type RealtimeConversation struct {
	ID     string `json:"id"`
	Object string `json:"object"`
}

// ResponseProperties configure one response.
type ResponseProperties struct {
	// OutputModalities are what the response is made of.
	OutputModalities []Modality          `json:"output_modalities,omitempty"`
	Instructions     string              `json:"instructions,omitempty"`
	Audio            *AudioConfiguration `json:"audio,omitempty"`
	Tools            []map[string]any    `json:"tools,omitempty"`
	ToolChoice       string              `json:"tool_choice,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	MaxOutputTokens  *TokenLimit         `json:"max_output_tokens,omitempty"`
}

// RealtimeError is an error the server reports.
type RealtimeError struct {
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	EventID string `json:"event_id,omitempty"`
}

// ClientEvent is the part every event the client sends shares: a fresh id and
// the event's type.
type ClientEvent struct {
	EventID string `json:"event_id"`
	Type    string `json:"type"`
}

func newClientEvent(eventType string) ClientEvent {
	return ClientEvent{EventID: uuid.NewString(), Type: eventType}
}

// SessionUpdateEvent updates the session's properties.
type SessionUpdateEvent struct {
	ClientEvent
	Session SessionProperties `json:"session"`
}

// NewSessionUpdateEvent builds a session update.
func NewSessionUpdateEvent(session SessionProperties) SessionUpdateEvent {
	return SessionUpdateEvent{ClientEvent: newClientEvent("session.update"), Session: session}
}

// InputAudioBufferAppendEvent appends base64-encoded audio to the input buffer.
type InputAudioBufferAppendEvent struct {
	ClientEvent
	Audio string `json:"audio"`
}

// NewInputAudioBufferAppendEvent builds an append of base64-encoded audio.
func NewInputAudioBufferAppendEvent(audio string) InputAudioBufferAppendEvent {
	return InputAudioBufferAppendEvent{ClientEvent: newClientEvent("input_audio_buffer.append"), Audio: audio}
}

// NewInputAudioBufferCommitEvent commits the input buffer as a user message.
func NewInputAudioBufferCommitEvent() ClientEvent { return newClientEvent("input_audio_buffer.commit") }

// NewInputAudioBufferClearEvent clears the input buffer.
func NewInputAudioBufferClearEvent() ClientEvent { return newClientEvent("input_audio_buffer.clear") }

// ConversationItemCreateEvent adds an item to the conversation.
type ConversationItemCreateEvent struct {
	ClientEvent
	PreviousItemID string           `json:"previous_item_id,omitempty"`
	Item           ConversationItem `json:"item"`
}

// NewConversationItemCreateEvent builds the addition of item.
func NewConversationItemCreateEvent(item ConversationItem) ConversationItemCreateEvent {
	return ConversationItemCreateEvent{ClientEvent: newClientEvent("conversation.item.create"), Item: item}
}

// ConversationItemTruncateEvent truncates an assistant message's audio at the
// point it was heard to.
type ConversationItemTruncateEvent struct {
	ClientEvent
	ItemID       string `json:"item_id"`
	ContentIndex int    `json:"content_index"`
	AudioEndMs   int    `json:"audio_end_ms"`
}

// NewConversationItemTruncateEvent builds a truncation of an item's audio.
func NewConversationItemTruncateEvent(itemID string, contentIndex, audioEndMs int) ConversationItemTruncateEvent {
	return ConversationItemTruncateEvent{
		ClientEvent: newClientEvent("conversation.item.truncate"),
		ItemID:      itemID, ContentIndex: contentIndex, AudioEndMs: audioEndMs,
	}
}

// ItemEvent is a client event about one item: its deletion or retrieval.
type ItemEvent struct {
	ClientEvent
	ItemID string `json:"item_id"`
}

// NewConversationItemDeleteEvent builds the deletion of an item.
func NewConversationItemDeleteEvent(itemID string) ItemEvent {
	return ItemEvent{ClientEvent: newClientEvent("conversation.item.delete"), ItemID: itemID}
}

// NewConversationItemRetrieveEvent builds the retrieval of an item.
func NewConversationItemRetrieveEvent(itemID string) ItemEvent {
	return ItemEvent{ClientEvent: newClientEvent("conversation.item.retrieve"), ItemID: itemID}
}

// ResponseCreateEvent asks the model to respond.
type ResponseCreateEvent struct {
	ClientEvent
	Response *ResponseProperties `json:"response,omitempty"`
}

// NewResponseCreateEvent builds a request for a response.
func NewResponseCreateEvent(response *ResponseProperties) ResponseCreateEvent {
	return ResponseCreateEvent{ClientEvent: newClientEvent("response.create"), Response: response}
}

// NewResponseCancelEvent cancels the response in progress.
func NewResponseCancelEvent() ClientEvent { return newClientEvent("response.cancel") }

// CachedTokensDetails splits the cached input by modality.
type CachedTokensDetails struct {
	TextTokens  *int64 `json:"text_tokens"`
	AudioTokens *int64 `json:"audio_tokens"`
}

// TokenDetails splits a side's tokens by modality and cache.
type TokenDetails struct {
	CachedTokens        *int64               `json:"cached_tokens"`
	TextTokens          *int64               `json:"text_tokens"`
	AudioTokens         *int64               `json:"audio_tokens"`
	CachedTokensDetails *CachedTokensDetails `json:"cached_tokens_details"`
	ImageTokens         *int64               `json:"image_tokens"`
}

// Usage is a response's token accounting.
type Usage struct {
	TotalTokens        *int64        `json:"total_tokens"`
	InputTokens        *int64        `json:"input_tokens"`
	OutputTokens       *int64        `json:"output_tokens"`
	InputTokenDetails  *TokenDetails `json:"input_token_details"`
	OutputTokenDetails *TokenDetails `json:"output_token_details"`
}

// Response is a response the server describes.
type Response struct {
	ID     string `json:"id,omitempty"`
	Object string `json:"object,omitempty"`
	// Status is "completed", "in_progress", "incomplete", "failed", or the
	// API's spelling of canceled.
	Status string `json:"status,omitempty"`
	// StatusDetails explains a status other than completed.
	StatusDetails     map[string]any      `json:"status_details,omitempty"`
	Output            []ConversationItem  `json:"output,omitempty"`
	OutputModalities  []Modality          `json:"output_modalities,omitempty"`
	MaxOutputTokens   *TokenLimit         `json:"max_output_tokens,omitempty"`
	Audio             *AudioConfiguration `json:"audio,omitempty"`
	Usage             *Usage              `json:"usage,omitempty"`
	Voice             string              `json:"voice,omitempty"`
	Temperature       *float64            `json:"temperature,omitempty"`
	OutputAudioFormat string              `json:"output_audio_format,omitempty"`
}

// ServerEvent is an event the server sends. Every event type the API documents
// is one of its shapes, so the fields an event does not carry are left empty.
type ServerEvent struct {
	EventID string `json:"event_id"`
	Type    string `json:"type"`

	Session      *SessionProperties    `json:"session,omitempty"`
	Conversation *RealtimeConversation `json:"conversation,omitempty"`
	Item         *ConversationItem     `json:"item,omitempty"`
	Response     *Response             `json:"response,omitempty"`
	Part         *ItemContent          `json:"part,omitempty"`
	Error        *RealtimeError        `json:"error,omitempty"`

	PreviousItemID string           `json:"previous_item_id,omitempty"`
	ItemID         string           `json:"item_id,omitempty"`
	ResponseID     string           `json:"response_id,omitempty"`
	CallID         string           `json:"call_id,omitempty"`
	OutputIndex    int              `json:"output_index,omitempty"`
	ContentIndex   int              `json:"content_index,omitempty"`
	AudioStartMs   int              `json:"audio_start_ms,omitempty"`
	AudioEndMs     int              `json:"audio_end_ms,omitempty"`
	Delta          string           `json:"delta,omitempty"`
	Text           string           `json:"text,omitempty"`
	Transcript     string           `json:"transcript,omitempty"`
	Arguments      string           `json:"arguments,omitempty"`
	RateLimits     []map[string]any `json:"rate_limits,omitempty"`
}

// serverEventTypes are the event types the server documents, which the
// service handles or knows to leave alone.
//
//nolint:gochecknoglobals // a lookup table
var serverEventTypes = map[string]bool{
	"error":                                                 true,
	"session.created":                                       true,
	"session.updated":                                       true,
	"conversation.created":                                  true,
	"input_audio_buffer.committed":                          true,
	"input_audio_buffer.cleared":                            true,
	"input_audio_buffer.speech_started":                     true,
	"input_audio_buffer.speech_stopped":                     true,
	"conversation.item.added":                               true,
	"conversation.item.done":                                true,
	"conversation.item.input_audio_transcription.delta":     true,
	"conversation.item.input_audio_transcription.completed": true,
	"conversation.item.input_audio_transcription.failed":    true,
	"conversation.item.truncated":                           true,
	"conversation.item.deleted":                             true,
	"conversation.item.retrieved":                           true,
	"response.created":                                      true,
	"response.done":                                         true,
	"response.output_item.added":                            true,
	"response.output_item.done":                             true,
	"response.content_part.added":                           true,
	"response.content_part.done":                            true,
	"response.output_text.delta":                            true,
	"response.output_text.done":                             true,
	"response.output_audio_transcript.delta":                true,
	"response.output_audio_transcript.done":                 true,
	"response.output_audio.delta":                           true,
	"response.output_audio.done":                            true,
	"response.function_call_arguments.delta":                true,
	"response.function_call_arguments.done":                 true,
	"rate_limits.updated":                                   true,
}

// ParseServerEvent parses an event the server sent. An event of a type the API
// does not document here parses to nil, with no error; one that cannot be read
// is an error quoting it.
func ParseServerEvent(data []byte) (*ServerEvent, error) {
	var ev ServerEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("events: %w\n\n%s", err, data)
	}
	if !serverEventTypes[ev.Type] {
		return nil, nil //nolint:nilnil // an event of an undocumented type is not an error
	}
	return &ev, nil
}
