---
title: Development runner
weight: 8
---

# Development runner

The `runner` package runs a bot and everything around it: the HTTP server, WebRTC
signaling, LiveKit tokens, and the WebSocket endpoints telephony providers and
WebSocket clients connect to. A bot is one function, called once per session:

```go
func bot(ctx context.Context, args runner.Arguments) error {
	t, err := runner.CreateTransport(ctx, args, runner.TransportParams{
		WebRTC: transport.DefaultParams,
		Twilio: func() runner.WebSocketParams {
			return runner.WebSocketParams{Params: wsserver.DefaultParams()}
		},
	})
	if err != nil {
		return err
	}
	// Build the pipeline around t.Input() and t.Output(), then run it with ctx.
	return nil
}

func main() { runner.Main(bot) }
```

`go run .` serves every transport on <http://localhost:7860>. A client says which
transport it wants in the `transport` field of `POST /start`; the session then
arrives on that transport and the runner calls `bot`.

## Transports

`CreateTransport` builds the transport from the session's arguments, with the
parameters from the matching `TransportParams` field. A transport whose field is
nil cannot be used.

| Transport | How a session arrives | Arguments |
|---|---|---|
| WebRTC | `/start`, then the offer at `/sessions/{id}/api/offer` (or `/api/offer`) | `*WebRTCRunnerArguments` |
| LiveKit | `/start` mints tokens for the bot and the caller | `*LiveKitRunnerArguments` |
| Twilio, Telnyx, Plivo, Exotel | the provider streams to `/ws` | `*WebSocketRunnerArguments` |
| WebSocket | a client connects to `/ws-client` | `*WebSocketRunnerArguments` |
| WhatsApp | the calling webhook at `/whatsapp`, with `--whatsapp` | `*WebRTCRunnerArguments` |

On `/ws` the provider is detected from its first messages. `CreateTransport`
then sets `TransportType` and `CallData` on the arguments (call id, numbers,
custom parameters) and builds the provider's serializer, with its credentials
from the environment (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` or the API key
pair `TWILIO_API_KEY` and `TWILIO_API_SECRET`, `TELNYX_API_KEY`,
`PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`). A plain WebSocket client
needs a serializer of your choosing in `TransportParams.WebSocket`.

## Command line

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `localhost`, `7860` | Where the server listens. |
| `-t`, `--transport` | all | Serve one transport only: `webrtc`, `websocket`, `livekit`, `twilio`, `telnyx`, `plivo` or `exotel`. |
| `-x`, `--proxy` | | Public host name telephony providers reach the server on; with `-t twilio`, `telnyx` or `plivo`, `POST /` answers the provider's webhook. |
| `--ice-servers` | `JARGO_ICE_SERVERS` | A STUN or TURN server, as a URL or a JSON object with credentials. Repeat for several. |
| `--allowed-origins` | `JARGO_ALLOWED_ORIGINS` | An origin allowed for HTTP and WebSocket requests. Repeat for several. |
| `--ws-auth` | `JARGO_WEBSOCKET_AUTH` or `none` | `token` makes `/ws` and `/ws-client` require a single-use token from `/start`. |
| `-f`, `--downloads-folder` | | Files served at `/files/{name}`. |
| `--runner-body` | | A YAML or JSON file passed as the body of each WebSocket session. |
| `--esp32` | | Rewrite WebRTC answers for ESP32 clients (needs `--host` set to an IP address). |
| `--whatsapp` | | Serve the WhatsApp webhook (needs the `WHATSAPP_*` variables). |
| `-v`, `--verbose` | | More logging. |

The runner loads a `.env` file from the working directory if there is one. Your
own flags go on `flag.CommandLine` before `runner.Main`, and the parsed runner
flags reach the bot as `args.Base().CLIArgs`. Add routes of your own to
`runner.App` before `runner.Main`.

Unlike the rest of jargo, which takes explicit `Config` structs, the runner reads
the command line and the environment: it is an application's entry point, not a
library a program configures.

## The browser client

The runner serves a ready-made browser client at `/client`, and `/` redirects
there: the published build of
[pipecat-ai-prebuilt](https://github.com/pipecat-ai/pipecat-prebuilt), embedded
by the `github.com/gojargo/jargo-web/prebuilt` module. It connects over WebRTC
and shows the conversation, events and metrics, which makes it a debugging UI.
Set `runner.Frontend` before `runner.Main` to serve your own handler instead, or
to nil to serve none.

## Not available

Daily, Vonage and MoQ have no jargo transport, so the runner does not serve
them. File uploads and the eval harness's connection on `/ws` are not served
yet.
