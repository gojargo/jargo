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

import (
	"encoding/json"

	"github.com/gojargo/jargo/frames"
)

const (
	// MessageLabel tags every RTVI message.
	MessageLabel = "rtvi-ai"
	// ProtocolVersion is the RTVI protocol version this implementation speaks.
	ProtocolVersion = "2.1.0"
	// LegacySupportedMajor is the older protocol generation still served. A
	// client of that generation is deprecated but answered rather than turned
	// away, and is told its own version back in the bot-ready rather than this
	// one, so it stays on the paths it understands.
	LegacySupportedMajor = 1
	// LibraryName is what this implementation calls itself in the bot-ready.
	LibraryName = "jargo"
)

// Message types exchanged over the data channel.
const (
	TypeClientReady          = "client-ready"
	TypeSendText             = "send-text"
	TypeDisconnectBot        = "disconnect-bot"
	TypeClientMessage        = "client-message"
	TypeFunctionCallResult   = "llm-function-call-result"
	TypeRawAudio             = "raw-audio"
	TypeRawAudioBatch        = "raw-audio-batch"
	TypeBotReady             = "bot-ready"
	TypeServerMessage        = "server-message"
	TypeServerResponse       = "server-response"
	TypeErrorResponse        = "error-response"
	TypeError                = "error"
	TypeUserTranscription    = "user-transcription"
	TypeBotOutput            = "bot-output"
	TypeBotTranscription     = "bot-transcription"
	TypeBotTTSText           = "bot-tts-text"
	TypeBotLLMText           = "bot-llm-text"
	TypeUserLLMText          = "user-llm-text"
	TypeUserMuteStarted      = "user-mute-started"
	TypeUserMuteStopped      = "user-mute-stopped"
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
	TypeUserAudioLevel       = "user-audio-level"
	TypeBotAudioLevel        = "bot-audio-level"
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

// AboutClientData describes the client an RTVI session is with: which client
// library it uses, on what platform, and whatever else it cares to say. It is
// what a client sends with its client-ready, and the same shape a bot-ready
// sends back about itself.
type AboutClientData struct {
	Library         string `json:"library"`
	LibraryVersion  string `json:"library_version,omitempty"`
	Platform        string `json:"platform,omitempty"`
	PlatformVersion string `json:"platform_version,omitempty"`
	PlatformDetails any    `json:"platform_details,omitempty"`
}

// ClientReadyData is the payload of a client-ready message: the protocol version
// the client speaks, and what it is.
type ClientReadyData struct {
	Version string          `json:"version"`
	About   AboutClientData `json:"about,omitzero"`
}

// ParseClientReadyData decodes the data payload of a client-ready message.
func ParseClientReadyData(raw json.RawMessage) (ClientReadyData, error) {
	var d ClientReadyData
	err := json.Unmarshal(raw, &d)
	return d, err
}

// BotReadyData is the payload of a bot-ready message: the protocol version the
// session settled on, and what the bot is.
type BotReadyData struct {
	Version string           `json:"version"`
	About   *AboutClientData `json:"about,omitempty"`
}

// BotReady builds a bot-ready message in reply to the client-ready with id,
// declaring version and describing the bot with about. A nil about describes
// this library.
func BotReady(id, version string, about *AboutClientData) Message {
	if about == nil {
		about = &AboutClientData{Library: LibraryName, LibraryVersion: LibraryVersion()}
	}
	return newMessage(TypeBotReady, id, BotReadyData{Version: version, About: about})
}

// RawClientMessageData is the payload of a client-message: the client's own
// message type and whatever it carries, both opaque to the protocol. It is how
// a client asks the bot something the protocol has no message for.
type RawClientMessageData struct {
	T string          `json:"t"`
	D json.RawMessage `json:"d,omitempty"`
}

// ParseRawClientMessageData decodes the data payload of a client-message.
func ParseRawClientMessageData(raw json.RawMessage) (RawClientMessageData, error) {
	var d RawClientMessageData
	err := json.Unmarshal(raw, &d)
	return d, err
}

// RawServerResponseData is the payload of a server-response: the type of the
// client message being answered, so the client can pair the answer with what it
// asked, and the answer itself.
type RawServerResponseData struct {
	T string `json:"t"`
	D any    `json:"d,omitempty"`
}

// ServerResponse builds the answer to the client message with id, which asked
// something of type msgType.
func ServerResponse(id, msgType string, data any) Message {
	return newMessage(TypeServerResponse, id, RawServerResponseData{T: msgType, D: data})
}

// ServerMessage builds an unprompted message to the client, carrying whatever
// the bot wants to tell it that the protocol has no message for.
func ServerMessage(data any) Message {
	return newMessage(TypeServerMessage, "", data)
}

// ErrorResponseData is the payload of an error-response.
type ErrorResponseData struct {
	Error string `json:"error"`
}

// ErrorResponse builds the refusal of the client message with id. It is what a
// client gets back instead of a server-response when its request could not be
// carried out, so a request never simply goes unanswered.
func ErrorResponse(id, err string) Message {
	return newMessage(TypeErrorResponse, id, ErrorResponseData{Error: err})
}

// FunctionCallResultData is the payload of an llm-function-call-result: a tool
// call the client ran on the bot's behalf, and what it produced.
type FunctionCallResultData struct {
	FunctionName string          `json:"function_name"`
	ToolCallID   string          `json:"tool_call_id"`
	Arguments    json.RawMessage `json:"arguments"`
	Result       json.RawMessage `json:"result"`
}

// ParseFunctionCallResultData decodes the data payload of an
// llm-function-call-result message.
func ParseFunctionCallResultData(raw json.RawMessage) (FunctionCallResultData, error) {
	var d FunctionCallResultData
	err := json.Unmarshal(raw, &d)
	return d, err
}

// ResultText is the result as the conversation records it. A result sent as a
// JSON string is the string itself; anything else is its JSON text, which is
// what a model reading the tool result expects to see.
func (d FunctionCallResultData) ResultText() string {
	var str string
	if err := json.Unmarshal(d.Result, &str); err == nil {
		return str
	}
	return string(d.Result)
}

// RawAudioData is the payload of a raw-audio or raw-audio-batch message: audio
// the client captured itself, base64-encoded 16-bit PCM, one chunk or a batch of
// them. It is how a client that does its own capture feeds the pipeline over the
// message channel rather than a media track.
type RawAudioData struct {
	//nolint:tagliatelle // RTVI wire fields, camelCase in the protocol
	Base64Audio string `json:"base64Audio,omitempty"`
	//nolint:tagliatelle // RTVI wire fields, camelCase in the protocol
	Base64AudioBatch []string `json:"base64AudioBatch,omitempty"`
	//nolint:tagliatelle // RTVI wire fields, camelCase in the protocol
	SampleRate int `json:"sampleRate"`
	//nolint:tagliatelle // RTVI wire fields, camelCase in the protocol
	NumChannels int `json:"numChannels"`
}

// Chunks are the encoded audio chunks the message carries, in order. A batch is
// taken whole when there is one; otherwise the single chunk stands alone.
func (d RawAudioData) Chunks() []string {
	if len(d.Base64AudioBatch) > 0 {
		return d.Base64AudioBatch
	}
	if d.Base64Audio == "" {
		return nil
	}
	return []string{d.Base64Audio}
}

// ParseRawAudioData decodes the data payload of a raw-audio or raw-audio-batch
// message.
func ParseRawAudioData(raw json.RawMessage) (RawAudioData, error) {
	var d RawAudioData
	err := json.Unmarshal(raw, &d)
	return d, err
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

// SpokenStatus is where a segment of the bot's output has got to in playback.
// It is empty for a segment that is not going to be spoken at all, which has no
// playback to be anywhere in.
type SpokenStatus string

// The playback states a spoken segment passes through.
const (
	// SpokenNew is a segment that is about to be spoken and has not started.
	SpokenNew SpokenStatus = "new"
	// SpokenInProgress is a segment partway through being spoken.
	SpokenInProgress SpokenStatus = "in-progress"
	// SpokenCompleted is a segment whose last word has been spoken.
	SpokenCompleted SpokenStatus = "completed"
)

// SpokenProgressData is how far through a segment the bot has spoken, split at
// the word being said now. A client renders the whole segment and highlights the
// accumulated part.
type SpokenProgressData struct {
	// AccumulatedText is what has been spoken so far, the current word included.
	AccumulatedText string `json:"accumulated_text"`
	// RemainingText is what has not been spoken yet.
	RemainingText string `json:"remaining_text"`
}

// BotOutputData is the payload of a bot-output message: one segment of what the
// bot is saying, with what is known about how it is being said.
//
// The optional fields belong to different protocol generations, and each is
// omitted rather than sent empty so a client sees only the ones its generation
// understands. Spoken is the older generation's account of whether the text has
// been spoken; the rest describe the segment's playback to a current client.
type BotOutputData struct {
	// Text is the segment as the client should render it.
	Text string `json:"text"`
	// AggregatedBy is the unit the text stands for: a sentence, a word, or a
	// name a pattern aggregator gave it.
	AggregatedBy frames.AggregationType `json:"aggregated_by"`
	// SegmentID identifies the segment, so progress reports can be matched to
	// the text they are about.
	SegmentID *uint64 `json:"segment_id,omitempty"`
	// Spoken reports whether the text has been spoken. Older clients only.
	Spoken *bool `json:"spoken,omitempty"`
	// WillBeSpoken reports whether the synthesizer is going to speak the text.
	WillBeSpoken *bool `json:"will_be_spoken,omitempty"`
	// SpokenStatus is where the segment has got to in playback.
	SpokenStatus SpokenStatus `json:"spoken_status,omitempty"`
	// SpokenProgress splits the segment at the word being spoken now.
	SpokenProgress *SpokenProgressData `json:"spoken_progress,omitempty"`
}

// BotOutput builds a bot-output message.
func BotOutput(d BotOutputData) Message {
	return newMessage(TypeBotOutput, "", d)
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

// UserLLMText builds a user-llm-text message: what the user said as the model
// is about to read it.
func UserLLMText(text string) Message {
	return newMessage(TypeUserLLMText, "", TextData{Text: text})
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

// AudioLevelData is how loud one side of the conversation currently is.
type AudioLevelData struct {
	// Value is the volume on the 0..1 scale audio/loudness measures.
	Value float64 `json:"value"`
}

// UserAudioLevel builds a message reporting how loud the user is.
func UserAudioLevel(level float64) Message {
	return newMessage(TypeUserAudioLevel, "", AudioLevelData{Value: level})
}

// BotAudioLevel builds a message reporting how loud the bot is.
func BotAudioLevel(level float64) Message {
	return newMessage(TypeBotAudioLevel, "", AudioLevelData{Value: level})
}

// event builds a data-less event message (speaking events).
func event(msgType string) Message {
	return newMessage(msgType, "", nil)
}
