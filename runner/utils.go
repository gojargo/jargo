package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/livekit"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/exotel"
	"github.com/gojargo/jargo/transport/wsserver/plivo"
	"github.com/gojargo/jargo/transport/wsserver/telnyx"
	"github.com/gojargo/jargo/transport/wsserver/twilio"
)

// The telephony providers the runner recognizes.
const (
	transportTwilio  = "twilio"
	transportTelnyx  = "telnyx"
	transportPlivo   = "plivo"
	transportExotel  = "exotel"
	transportUnknown = "unknown"
)

// ErrHandshakeClosed is returned when a telephony WebSocket closes before it
// sends any message.
var ErrHandshakeClosed = errors.New("runner: WebSocket closed before receiving telephony handshake messages")

// ErrMissingTransportParams is returned when a session arrives on a transport
// TransportParams does not configure.
var ErrMissingTransportParams = errors.New("runner: missing transport params")

// ErrUnsupportedProvider is returned for a telephony provider the runner does
// not recognize.
var ErrUnsupportedProvider = errors.New(
	"runner: unsupported telephony provider; supported providers: twilio, telnyx, plivo, exotel")

// ErrSerializerRequired is returned when a plain WebSocket session has no
// serializer configured.
var ErrSerializerRequired = errors.New("runner: WebSocketParams.Serializer is required for a plain WebSocket")

// ErrUnsupportedArguments is returned for runner arguments CreateTransport
// cannot build a transport for.
var ErrUnsupportedArguments = errors.New("runner: unsupported runner arguments type")

// WebSocketParams configures a WebSocket transport: its transport parameters
// and the serializer for its wire format. For a telephony provider the runner
// sets the serializer itself.
type WebSocketParams struct {
	wsserver.Params
	Serializer wsserver.Serializer
}

// TransportParams says how to configure each transport a session may arrive
// on. Each field builds the parameters when its transport is selected; a
// transport whose field is nil cannot be used.
type TransportParams struct {
	WebRTC    func() transport.Params
	LiveKit   func() transport.Params
	WebSocket func() WebSocketParams
	Twilio    func() WebSocketParams
	Telnyx    func() WebSocketParams
	Plivo     func() WebSocketParams
	Exotel    func() WebSocketParams
}

// missingParams reports a transport with no parameters configured.
func missingParams(key, field string) error {
	return fmt.Errorf("%w for %q: set TransportParams.%s", ErrMissingTransportParams, key, field)
}

// webSocketParams returns the parameters for a WebSocket transport, by key.
func (p TransportParams) webSocketParams(key string) (WebSocketParams, error) {
	var (
		build func() WebSocketParams
		field string
	)
	switch key {
	case "websocket":
		build, field = p.WebSocket, "WebSocket"
	case transportTwilio:
		build, field = p.Twilio, "Twilio"
	case transportTelnyx:
		build, field = p.Telnyx, "Telnyx"
	case transportPlivo:
		build, field = p.Plivo, "Plivo"
	case transportExotel:
		build, field = p.Exotel, "Exotel"
	default:
		return WebSocketParams{}, fmt.Errorf("%w: %q", ErrUnsupportedProvider, key)
	}
	if build == nil {
		return WebSocketParams{}, missingParams(key, field)
	}
	slog.Debug("runner: using transport params", "transport", key)
	return build(), nil
}

// field reads one string field of a JSON object, or "".
func field(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

// object reads one object field of a JSON object, or an empty one.
func object(m map[string]any, key string) map[string]any {
	o, _ := m[key].(map[string]any)
	if o == nil {
		return map[string]any{}
	}
	return o
}

func has(m map[string]any, key string) bool {
	_, ok := m[key]
	return ok
}

// detectTransportType tells which telephony provider sent message from its
// shape, or reports "unknown".
func detectTransportType(message map[string]any) string {
	slog.Debug("runner: auto-detection analysis")
	start := object(message, "start")
	isStart := field(message, "event") == "start" && has(message, "start")

	switch {
	case isStart && has(start, "streamSid") && has(start, "callSid"):
		return transportTwilio
	case has(message, "stream_id") && has(message, "start") && has(start, "call_control_id"):
		return transportTelnyx
	case has(message, "start") && has(start, "streamId") && has(start, "callId"):
		return transportPlivo
	case isStart && has(start, "stream_sid") && has(start, "call_sid") && has(start, "account_sid"):
		return transportExotel
	}
	slog.Debug("runner: auto-detection failed, unknown format")
	return transportUnknown
}

// readMessage reads the next message as a JSON object. A message that is not
// one reads as an empty object.
func readMessage(ctx context.Context, args *WebSocketRunnerArguments) (map[string]any, error) {
	_, data, err := args.WebSocket.Read(ctx)
	if err != nil {
		return nil, err
	}
	args.consumed = append(args.consumed, data)
	slog.Debug("runner: telephony message", "message", string(data))
	m := map[string]any{}
	if len(data) > 0 {
		if err := json.Unmarshal(data, &m); err != nil || m == nil {
			m = map[string]any{}
		}
	}
	return m, nil
}

// ParseTelephonyWebSocket reads a telephony provider's opening messages and
// reports which provider it is and what it says about the call.
//
// It reads the first message, which is required, and a second one, which some
// providers never send. It is idempotent: the messages it reads are gone from
// the socket, so the first parse is kept on args and returned again, and the
// transport CreateTransport builds is handed the messages to deserialize first.
func ParseTelephonyWebSocket(ctx context.Context, args *WebSocketRunnerArguments) (string, *CallData, error) {
	args.parseOnce.Do(func() {
		args.parsedType, args.parsedData, args.parseErr = parseTelephonyWebSocket(ctx, args)
	})
	return args.parsedType, args.parsedData, args.parseErr
}

func parseTelephonyWebSocket(ctx context.Context, args *WebSocketRunnerArguments) (string, *CallData, error) {
	first, err := readMessage(ctx, args)
	if err != nil {
		return "", nil, fmt.Errorf("%w: %w", ErrHandshakeClosed, err)
	}

	second, err := readMessage(ctx, args)
	if err != nil {
		slog.Warn("runner: only received one WebSocket message, expected two")
		second = map[string]any{}
	}

	transportType := detectTransportType(first)
	raw := first
	switch {
	case transportType != transportUnknown:
		slog.Debug("runner: detected transport from the first message", "transport", transportType)
	case detectTransportType(second) != transportUnknown:
		transportType = detectTransportType(second)
		raw = second
		slog.Debug("runner: detected transport from the second message", "transport", transportType)
	default:
		raw = second
		slog.Warn("runner: could not auto-detect transport type")
	}

	start := object(raw, "start")
	data := &CallData{}
	switch transportType {
	case transportTwilio:
		body := object(start, "customParameters")
		data.StreamID = field(start, "streamSid")
		data.CallID = field(start, "callSid")
		data.Body = body
		// Twilio carries from and to as TwiML stream parameters; promoting them
		// makes the call data read the same across providers.
		data.FromNumber = field(body, "from_number")
		data.ToNumber = field(body, "to_number")
	case transportTelnyx:
		data.StreamID = field(raw, "stream_id")
		// Telnyx's call identifier is its call_control_id.
		data.CallID = field(start, "call_control_id")
		data.OutboundEncoding = field(object(start, "media_format"), "encoding")
		data.FromNumber = field(start, "from")
		data.ToNumber = field(start, "to")
	case transportPlivo:
		data.StreamID = field(start, "streamId")
		data.CallID = field(start, "callId")
	case transportExotel:
		data.StreamID = field(start, "stream_sid")
		data.CallID = field(start, "call_sid")
		data.AccountSID = field(start, "account_sid")
		data.FromNumber = field(start, "from")
		data.ToNumber = field(start, "to")
		data.CustomParameters = start["custom_parameters"]
		if data.CustomParameters == nil {
			data.CustomParameters = ""
		}
	}
	slog.Debug("runner: parsed telephony handshake", "transport", transportType, "call_id", data.CallID)
	return transportType, data, nil
}

// takeConsumed hands over the messages the parse read, once.
func (a *WebSocketRunnerArguments) takeConsumed() [][]byte {
	if a.consumedTaken {
		return nil
	}
	a.consumedTaken = true
	return a.consumed
}

// createTelephonyTransport builds the WebSocket transport for a detected
// telephony provider, with the provider's serializer. Credentials come from the
// environment.
func createTelephonyTransport(args *WebSocketRunnerArguments, params WebSocketParams, transportType string) (
	transport.Transport, error,
) {
	// Telephony audio never carries a WAV header.
	params.AddWAVHeader = false
	slog.Info("runner: using pre-detected telephony provider", "provider", transportType)

	switch transportType {
	case transportTwilio:
		params.Serializer = twilio.New(twilio.Config{
			AccountSID:   os.Getenv("TWILIO_ACCOUNT_SID"),
			AuthToken:    os.Getenv("TWILIO_AUTH_TOKEN"),
			APIKeySID:    os.Getenv("TWILIO_API_KEY"),
			APIKeySecret: os.Getenv("TWILIO_API_SECRET"),
			Region:       os.Getenv("TWILIO_REGION"),
			Edge:         os.Getenv("TWILIO_EDGE"),
		})
	case transportTelnyx:
		params.Serializer = telnyx.New(telnyx.Config{APIKey: os.Getenv("TELNYX_API_KEY")})
	case transportPlivo:
		params.Serializer = plivo.New(plivo.Config{
			AuthID:    os.Getenv("PLIVO_AUTH_ID"),
			AuthToken: os.Getenv("PLIVO_AUTH_TOKEN"),
		})
	case transportExotel:
		params.Serializer = exotel.New(exotel.Config{})
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedProvider, transportType)
	}
	return wsserver.NewTransport(args.WebSocket, params.Serializer, params.Params, args.takeConsumed()...), nil
}

// CreateTransport builds the transport for a session from the arguments the
// runner handed the bot, configured by params.
//
// A WebSocket session that is not a plain WebSocket client is a telephony call:
// its provider is detected from its first messages, and args.TransportType and
// args.CallData are set from what they say, so the bot can personalize the
// call without parsing them again.
func CreateTransport(ctx context.Context, args Arguments, params TransportParams) (transport.Transport, error) {
	switch a := args.(type) {
	case *WebRTCRunnerArguments:
		if params.WebRTC == nil {
			return nil, missingParams("webrtc", "WebRTC")
		}
		return rtc.NewTransport(a.WebRTCConnection, params.WebRTC()), nil

	case *WebSocketRunnerArguments:
		if a.TransportType == "websocket" {
			p, err := params.webSocketParams("websocket")
			if err != nil {
				return nil, err
			}
			if p.Serializer == nil {
				return nil, ErrSerializerRequired
			}
			return wsserver.NewTransport(a.WebSocket, p.Serializer, p.Params), nil
		}
		transportType, data, err := ParseTelephonyWebSocket(ctx, a)
		if err != nil {
			return nil, err
		}
		a.TransportType = transportType
		a.CallData = data
		p, err := params.webSocketParams(transportType)
		if err != nil {
			return nil, err
		}
		return createTelephonyTransport(a, p, transportType)

	case *LiveKitRunnerArguments:
		if params.LiveKit == nil {
			return nil, missingParams("livekit", "LiveKit")
		}
		lkParams := params.LiveKit()
		conn, err := livekit.Connect(livekit.Config{
			URL: a.URL, Token: a.Token, AudioOutChannels: lkParams.AudioOutChannels,
		})
		if err != nil {
			return nil, err
		}
		return livekit.NewTransport(conn, lkParams), nil
	}
	return nil, fmt.Errorf("%w: %T", ErrUnsupportedArguments, args)
}
