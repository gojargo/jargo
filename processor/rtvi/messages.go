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
	TypeClientReady          = "client-ready"
	TypeSendText             = "send-text"
	TypeBotReady             = "bot-ready"
	TypeError                = "error"
	TypeUserTranscription    = "user-transcription"
	TypeBotTranscription     = "bot-transcription"
	TypeBotTTSText           = "bot-tts-text"
	TypeBotLLMText           = "bot-llm-text"
	TypeUserStartedSpeaking  = "user-started-speaking"
	TypeUserStoppedSpeaking  = "user-stopped-speaking"
	TypeVADUserStarted       = "vad-user-started-speaking"
	TypeVADUserStopped       = "vad-user-stopped-speaking"
	TypeDTMF                 = "dtmf"
	TypeBotStartedSpeaking   = "bot-started-speaking"
	TypeBotStoppedSpeaking   = "bot-stopped-speaking"
	TypeBotInterrupted       = "bot-interrupted"
	TypeBotLLMStarted        = "bot-llm-started"
	TypeBotLLMStopped        = "bot-llm-stopped"
	TypeBotTTSStarted        = "bot-tts-started"
	TypeBotTTSStopped        = "bot-tts-stopped"
	TypeLLMFunctionCallStart = "llm-function-call-started"
	TypeLLMFunctionCall      = "llm-function-call-in-progress"
	TypeLLMFunctionCallStop  = "llm-function-call-stopped"
	TypeMetrics              = "metrics"
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

// AudioResponse reports whether the reply to the injected text should be
// spoken. Absent options (or an absent flag) default to true.
func (d SendTextData) AudioResponse() bool {
	return d.Options == nil || d.Options.AudioResponse == nil || *d.Options.AudioResponse
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
// The tool call id is always present; the name and the arguments are omitted
// unless the observer's report level for the function allows them (see
// FunctionCallReportLevel), because either can carry information a client has no
// business seeing.
type LLMFunctionCallData struct {
	ToolCallID   string          `json:"tool_call_id"`
	FunctionName string          `json:"function_name,omitempty"`
	Arguments    json.RawMessage `json:"arguments,omitempty"`
}

// LLMFunctionCall builds a llm-function-call-in-progress message carrying as
// much of the call as level allows.
func LLMFunctionCall(name, toolCallID string, args json.RawMessage, level FunctionCallReportLevel) Message {
	d := LLMFunctionCallData{ToolCallID: toolCallID}
	if level == ReportName || level == ReportFull {
		d.FunctionName = name
	}
	if level == ReportFull {
		d.Arguments = args
	}
	return newMessage(TypeLLMFunctionCall, "", d)
}

// LLMFunctionCallStartData is the payload of a llm-function-call-started
// message: the model has asked for a call, before it begins executing. The name
// is omitted unless the observer's report level for the function allows it.
type LLMFunctionCallStartData struct {
	FunctionName string `json:"function_name,omitempty"`
}

// LLMFunctionCallStart builds a llm-function-call-started message carrying as
// much of the call as level allows.
func LLMFunctionCallStart(name string, level FunctionCallReportLevel) Message {
	var d LLMFunctionCallStartData
	if level == ReportName || level == ReportFull {
		d.FunctionName = name
	}
	return newMessage(TypeLLMFunctionCallStart, "", d)
}

// LLMFunctionCallStoppedData is the payload of a llm-function-call-stopped
// message, sent when a call completes with a result or is canceled. As with the
// in-progress payload, the name and the result are omitted unless the observer's
// report level for the function allows them.
type LLMFunctionCallStoppedData struct {
	ToolCallID string `json:"tool_call_id"`
	// Canceled reports whether the call was canceled rather than completing. The
	// wire name keeps the protocol's spelling, which the clients already send.
	Canceled     bool   `json:"cancelled"` //nolint:misspell // the protocol spells it this way
	FunctionName string `json:"function_name,omitempty"`
	Result       string `json:"result,omitempty"`
}

// LLMFunctionCallStopped builds a llm-function-call-stopped message carrying as
// much of the outcome as level allows. A canceled call has no result to report.
func LLMFunctionCallStopped(
	name, toolCallID, result string, canceled bool, level FunctionCallReportLevel,
) Message {
	d := LLMFunctionCallStoppedData{ToolCallID: toolCallID, Canceled: canceled}
	if level == ReportName || level == ReportFull {
		d.FunctionName = name
	}
	if level == ReportFull && !canceled {
		d.Result = result
	}
	return newMessage(TypeLLMFunctionCallStop, "", d)
}

// DTMFData is the payload of a dtmf message: the keypad keys the client
// pressed, in the order they were pressed.
type DTMFData struct {
	Buttons []string `json:"buttons"`
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

// TTFAMetricData is one time-to-first-audible-sample entry, reported with the
// breakdown that makes it up: the time to first byte it builds on, and the
// silence padded on before the first audible sample. TTFB here is the same
// measurement reported under "ttfb", not another one.
type TTFAMetricData struct {
	Processor      string  `json:"processor"`
	Model          string  `json:"model,omitempty"`
	TTFA           float64 `json:"ttfa"`
	TTFB           float64 `json:"ttfb"`
	LeadingSilence float64 `json:"leading_silence"`
}

// MetricsData is the payload of a metrics message: each kind is a list so a
// single message can report several processors at once.
type MetricsData struct {
	TTFB            []MetricData      `json:"ttfb,omitempty"`
	TTFA            []TTFAMetricData  `json:"ttfa,omitempty"`
	Processing      []MetricData      `json:"processing,omitempty"`
	Characters      []MetricData      `json:"characters,omitempty"`
	STTUsage        []MetricData      `json:"stt_usage,omitempty"`
	TextAggregation []MetricData      `json:"text_aggregation,omitempty"`
	Tokens          []TokenMetricData `json:"tokens,omitempty"`
	Turn            []TurnMetricData  `json:"turn,omitempty"`
}

// TurnMetricData is one end-of-turn prediction: whether the analyzer judged the
// turn finished, how confident it was, and how long deciding took.
type TurnMetricData struct {
	Processor    string  `json:"processor"`
	Complete     bool    `json:"complete"`
	Probability  float64 `json:"probability"`
	ProcessingMs float64 `json:"processing_ms"`
}

// Metrics builds a metrics message from data.
func Metrics(data MetricsData) Message {
	return newMessage(TypeMetrics, "", data)
}

// event builds a data-less event message (speaking events).
func event(msgType string) Message {
	return newMessage(msgType, "", nil)
}
