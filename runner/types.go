package runner

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/transport/rtc"
)

// CallData is what a telephony provider's opening WebSocket messages say about
// the call, parsed by ParseTelephonyWebSocket and exposed on the bot's
// arguments by CreateTransport. Fields a provider does not send stay empty.
type CallData struct {
	// StreamID is the provider's media-stream identifier.
	StreamID string `json:"stream_id,omitempty"`
	// CallID is the provider's call identifier, normalized across providers:
	// Twilio's callSid, Plivo's callId, Exotel's call_sid, Telnyx's
	// call_control_id.
	CallID string `json:"call_id,omitempty"`
	// FromNumber is the caller's number.
	FromNumber string `json:"from,omitempty"`
	// ToNumber is the dialed number.
	ToNumber string `json:"to,omitempty"`
	// Body holds the custom parameters the provider sent, such as Twilio's TwiML
	// stream parameters.
	Body map[string]any `json:"body,omitempty"`

	// OutboundEncoding is Telnyx's outbound media encoding.
	OutboundEncoding string `json:"outbound_encoding,omitempty"`
	// AccountSID is Exotel's account sid.
	AccountSID string `json:"account_sid,omitempty"`
	// CustomParameters are Exotel's custom parameters, a string or an object.
	CustomParameters any `json:"custom_parameters,omitempty"`
}

// Arguments is what the runner hands a bot for one session. Each transport has
// its own type, which a bot can switch on: *WebRTCRunnerArguments,
// *WebSocketRunnerArguments, *LiveKitRunnerArguments, or the plain
// *RunnerArguments.
type Arguments interface {
	// Base returns the fields every session has.
	Base() *RunnerArguments
}

// RunnerArguments holds what every session has, whatever its transport.
type RunnerArguments struct {
	// HandleSigint and HandleSigterm say whether the bot should install its own
	// signal handlers. The runner handles signals itself, so both are false.
	HandleSigint  bool
	HandleSigterm bool
	// PipelineIdleTimeout is how long the pipeline may stay idle before shutting
	// down; zero disables it. The runner sets 300 seconds.
	PipelineIdleTimeout time.Duration
	// Body is the request data passed from the runner's entry point, as JSON. It
	// is nil when none was given, which a bot can tell from an empty object.
	Body json.RawMessage
	// CallData is the parsed telephony handshake, set by CreateTransport for a
	// telephony connection and nil otherwise.
	CallData *CallData
	// SessionID identifies this session.
	SessionID string
	// CLIArgs are the runner's parsed command-line arguments.
	CLIArgs *Args
}

// Base returns the arguments themselves.
func (a *RunnerArguments) Base() *RunnerArguments { return a }

// newRunnerArguments returns the fields every session starts with.
func newRunnerArguments(body json.RawMessage, sessionID string) RunnerArguments {
	return RunnerArguments{
		PipelineIdleTimeout: 300 * time.Second,
		Body:                body,
		SessionID:           sessionID,
	}
}

// WebRTCRunnerArguments is a session over a WebRTC peer connection the runner
// has already answered.
type WebRTCRunnerArguments struct {
	RunnerArguments
	// WebRTCConnection is the session's peer connection.
	WebRTCConnection *rtc.Connection
}

// WebSocketRunnerArguments is a session over a WebSocket the runner has
// accepted: a telephony provider's media stream, or a plain WebSocket client.
type WebSocketRunnerArguments struct {
	RunnerArguments
	// WebSocket is the accepted connection.
	WebSocket *websocket.Conn
	// TransportType is "websocket" for a plain WebSocket client. Empty means a
	// telephony provider, detected from its first messages; CreateTransport
	// then sets it to the provider it found (for example "twilio").
	TransportType string

	// The handshake is parsed once, since the messages it reads are gone from
	// the socket afterwards. consumed keeps them for the transport.
	parseOnce     sync.Once
	parsedType    string
	parsedData    *CallData
	parseErr      error
	consumed      [][]byte
	consumedTaken bool
}

// LiveKitRunnerArguments is a session in a LiveKit room the bot joins.
type LiveKitRunnerArguments struct {
	RunnerArguments
	// RoomName is the room to join.
	RoomName string
	// URL is the LiveKit server URL.
	URL string
	// Token is the bot's access token for the room.
	Token string
}
