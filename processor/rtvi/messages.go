// Package rtvi implements the RTVI protocol over a transport's messaging
// channel: a JSON message format and a processor that completes the client
// handshake and reports pipeline events to the client.
//
// RTVI (Real-Time Voice Interface) is the protocol the Pipecat client SDKs
// speak, so a jargo server interoperates with existing RTVI web, iOS and
// Android clients. Messages are JSON objects of the form
// {"label":"rtvi-ai","type":...,"id":...,"data":...} exchanged over the WebRTC
// data channel.
package rtvi

import "encoding/json"

const (
	// MessageLabel tags every RTVI message.
	MessageLabel = "rtvi-ai"
	// ProtocolVersion is the RTVI protocol version this implementation speaks.
	ProtocolVersion = "2.0.0"
)

// Message types exchanged over the data channel.
const (
	TypeClientReady           = "client-ready"
	TypeSendText              = "send-text"
	TypeBotReady              = "bot-ready"
	TypeError                 = "error"
	TypeUserTranscription     = "user-transcription"
	TypeBotTranscription      = "bot-transcription"
	TypeBotTTSText            = "bot-tts-text"
	TypeBotLLMText            = "bot-llm-text"
	TypeUserStartedSpeaking   = "user-started-speaking"
	TypeUserStoppedSpeaking   = "user-stopped-speaking"
	TypeBotStartedSpeaking    = "bot-started-speaking"
	TypeBotStoppedSpeaking    = "bot-stopped-speaking"
	TypeBotLLMStarted         = "bot-llm-started"
	TypeBotLLMStopped         = "bot-llm-stopped"
	TypeBotTTSStarted         = "bot-tts-started"
	TypeBotTTSStopped         = "bot-tts-stopped"
	TypeLLMFunctionCall       = "llm-function-call-in-progress"
	TypeLLMFunctionCallResult = "llm-function-call-result"
	TypeMetrics               = "metrics"
)

// Message is the RTVI message envelope. Outgoing event messages omit id; bot-ready
// and responses echo the request id.
type Message struct {
	Label string `json:"label"`
	Type  string `json:"type"`
	ID    string `json:"id,omitempty"`
	Data  any    `json:"data,omitempty"`
}

// newMessage builds a Message with the RTVI label.
func newMessage(msgType, id string, data any) Message {
	return Message{Label: MessageLabel, Type: msgType, ID: id, Data: data}
}

// Incoming is a received RTVI message with its data left as raw JSON for
// type-specific decoding.
type Incoming struct {
	Label string          `json:"label"`
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Data  json.RawMessage `json:"data"`
}

// ParseIncoming decodes a received RTVI message.
func ParseIncoming(raw []byte) (Incoming, error) {
	var m Incoming
	err := json.Unmarshal(raw, &m)
	return m, err
}

// SendTextOptions controls how the pipeline processes a send-text message.
// Both fields default to true when absent, matching the RTVI client SDKs.
type SendTextOptions struct {
	RunImmediately *bool `json:"run_immediately,omitempty"`
	AudioResponse  *bool `json:"audio_response,omitempty"`
}

// SendTextData is the payload of a send-text message: user text to inject into
// the conversation, with options controlling whether the LLM runs immediately.
type SendTextData struct {
	Content string           `json:"content"`
	Options *SendTextOptions `json:"options,omitempty"`
}

// RunImmediately reports whether the LLM should run as soon as the text is
// appended. Absent options (or an absent flag) default to true.
func (d SendTextData) RunImmediately() bool {
	return d.Options == nil || d.Options.RunImmediately == nil || *d.Options.RunImmediately
}

// ParseSendTextData decodes the data payload of a send-text message.
func ParseSendTextData(raw json.RawMessage) (SendTextData, error) {
	var d SendTextData
	err := json.Unmarshal(raw, &d)
	return d, err
}

// BotReadyData is the payload of a bot-ready message.
type BotReadyData struct {
	Version string `json:"version"`
}

// BotReady builds a bot-ready message in reply to the client-ready with id.
func BotReady(id string) Message {
	return newMessage(TypeBotReady, id, BotReadyData{Version: ProtocolVersion})
}

// ErrorData is the payload of an error message.
type ErrorData struct {
	Error string `json:"error"`
	Fatal bool   `json:"fatal"`
}

// Error builds an error message.
func Error(msg string, fatal bool) Message {
	return newMessage(TypeError, "", ErrorData{Error: msg, Fatal: fatal})
}

// TextData is the payload of text messages (bot-transcription, bot-tts-text,
// bot-llm-text).
type TextData struct {
	Text string `json:"text"`
}

// BotTranscription builds a bot-transcription message.
func BotTranscription(text string) Message {
	return newMessage(TypeBotTranscription, "", TextData{Text: text})
}

// BotTTSText builds a bot-tts-text message.
func BotTTSText(text string) Message {
	return newMessage(TypeBotTTSText, "", TextData{Text: text})
}

// BotLLMText builds a bot-llm-text message.
func BotLLMText(text string) Message {
	return newMessage(TypeBotLLMText, "", TextData{Text: text})
}

// LLMFunctionCallData is the payload of a llm-function-call-in-progress message.
// Arguments are intentionally omitted: the call name and id are enough for a
// client to surface an "in progress" indication without exposing argument
// values.
type LLMFunctionCallData struct {
	FunctionName string `json:"function_name"`
	ToolCallID   string `json:"tool_call_id"`
}

// LLMFunctionCall builds a llm-function-call-in-progress message.
func LLMFunctionCall(name, toolCallID string) Message {
	return newMessage(TypeLLMFunctionCall, "", LLMFunctionCallData{
		FunctionName: name,
		ToolCallID:   toolCallID,
	})
}

// LLMFunctionCallResultData is the payload of a llm-function-call-result message.
type LLMFunctionCallResultData struct {
	FunctionName string `json:"function_name"`
	ToolCallID   string `json:"tool_call_id"`
	Result       string `json:"result"`
}

// LLMFunctionCallResult builds a llm-function-call-result message.
func LLMFunctionCallResult(name, toolCallID, result string) Message {
	return newMessage(TypeLLMFunctionCallResult, "", LLMFunctionCallResultData{
		FunctionName: name,
		ToolCallID:   toolCallID,
		Result:       result,
	})
}

// UserTranscriptionData is the payload of a user-transcription message.
type UserTranscriptionData struct {
	Text      string `json:"text"`
	UserID    string `json:"user_id"`
	Timestamp string `json:"timestamp"`
	Final     bool   `json:"final"`
}

// UserTranscription builds a user-transcription message.
func UserTranscription(text, userID, timestamp string, final bool) Message {
	return newMessage(TypeUserTranscription, "", UserTranscriptionData{
		Text:      text,
		UserID:    userID,
		Timestamp: timestamp,
		Final:     final,
	})
}

// MetricData is one timing or count entry in a metrics message (ttfb,
// processing or characters). Value is in seconds for timings, or a count.
type MetricData struct {
	Processor string  `json:"processor"`
	Value     float64 `json:"value"`
	Model     string  `json:"model,omitempty"`
}

// TokenMetricData is one LLM token-usage entry in a metrics message.
type TokenMetricData struct {
	Processor        string `json:"processor"`
	Model            string `json:"model,omitempty"`
	PromptTokens     int64  `json:"prompt_tokens"`
	CompletionTokens int64  `json:"completion_tokens"`
	TotalTokens      int64  `json:"total_tokens"`
}

// MetricsData is the payload of a metrics message: each kind is a list so a
// single message can report several processors at once.
type MetricsData struct {
	TTFB       []MetricData      `json:"ttfb,omitempty"`
	Processing []MetricData      `json:"processing,omitempty"`
	Characters []MetricData      `json:"characters,omitempty"`
	Tokens     []TokenMetricData `json:"tokens,omitempty"`
}

// Metrics builds a metrics message from data.
func Metrics(data MetricsData) Message {
	return newMessage(TypeMetrics, "", data)
}

// event builds a data-less event message (speaking events).
func event(msgType string) Message {
	return newMessage(msgType, "", nil)
}
