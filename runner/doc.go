// Package runner is the jargo development runner. It runs a bot and provides
// what the bot needs around it: answering WebRTC connections, minting LiveKit
// tokens, and accepting telephony and WebSocket connections, behind one HTTP
// server that serves all of them at once.
//
// A bot is a Bot function. The runner calls it once for each session, with the
// arguments for the transport the session arrived on, and CreateTransport turns
// those arguments into a transport:
//
//	func bot(ctx context.Context, args runner.Arguments) error {
//		t, err := runner.CreateTransport(ctx, args, runner.TransportParams{
//			WebRTC: func() transport.Params { return transport.DefaultParams() },
//		})
//		if err != nil {
//			return err
//		}
//		// Build the pipeline around t.Input() and t.Output() and run it.
//	}
//
//	func main() { runner.Main(bot) }
//
// A bot that needs to know its transport switches on the arguments' type:
// *WebRTCRunnerArguments, *WebSocketRunnerArguments or *LiveKitRunnerArguments.
//
// Clients declare the transport they want in the transport field of POST
// /start, which defaults to "webrtc":
//
//	{
//		"transport": "webrtc",  // webrtc, livekit, websocket, twilio, telnyx, plivo or exotel
//		"enableDefaultIceServers": false,  // WebRTC only
//		"body": {...}
//	}
//
// The supported transports are:
//
//   - WebRTC: the browser client at /client, signaling at /api/offer.
//   - LiveKit: mints room tokens and runs the bot as a participant.
//   - Telephony: the webhook and the media WebSocket (/ws) for Twilio, Telnyx,
//     Plivo and Exotel, the provider detected from its first messages.
//   - WebSocket: plain WebSocket clients at /ws-client.
//   - WhatsApp: the calling webhook at /whatsapp, with --whatsapp.
//
// To run a bot locally:
//
//   - All transports (default): go run .
//   - WebRTC only: go run . -t webrtc
//   - ESP32: go run . -t webrtc --esp32 --host 192.168.1.100
//   - WebRTC with custom STUN/TURN: go run . -t webrtc --ice-servers stun:stun.l.google.com:19302
//     --ice-servers '{"urls": "turn:turn.example.com:3478", "username": "user", "credential": "pass"}'
//     (or set JARGO_ICE_SERVERS)
//   - Telephony: go run . -t twilio -x your_username.ngrok.io
//   - Exotel: go run . -t exotel (no proxy needed, but a tunnel to port 7860 is)
//   - LiveKit only: go run . -t livekit
//   - WhatsApp: go run . --whatsapp
//
// Unlike the rest of jargo, the runner reads its configuration from the
// command line and the environment, a .env file included: it is the entry point
// of an application, not a library a program configures.
package runner
