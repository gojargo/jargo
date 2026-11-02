# Examples

Every example is a `main` package you run with `go run ./examples/<name>`. Each
file's doc comment lists the API keys it reads and how to drive it.

Turn-taking (Silero VAD and Smart Turn) needs the ONNX Runtime. Without it the
voice examples still run, on STT endpointing, and lose barge-in. See
[Installation](../docs/getting-started/installation.md).

## No API keys

| Example | What it shows |
| --- | --- |
| [`echo`](echo) | Speak in the browser and hear yourself back over WebRTC. Open <http://localhost:8080>. |
| [`localaudio`](localaudio) | The same echo through the local microphone and speaker, no browser. |
| [`eval`](eval) | A tiny bot served over a WebSocket for the eval harness, driven by `jargo eval run` or a Go test. |

## Voice agents

| Example | What it shows |
| --- | --- |
| [`voicebot`](voicebot) | The full agent: Deepgram, Anthropic and ElevenLabs over WebRTC, with turn-taking, barge-in, RTVI, optional mem0 memory and OpenTelemetry tracing. Serves its own page at <http://localhost:8080>. |
| [`flows`](flows) | A structured coffee order driven by the `flows` package: tool calls move the conversation from node to node. |
| [`voice/<provider>`](voice) | One backend per provider, each wiring its STT, LLM and TTS explicitly. |

The `voice/<provider>` backends and `flows` expose only the WebRTC `POST /offer`
endpoint. Drive them from a browser client such as the `nextjs-voicebot` example
in [jargo-client-react](https://github.com/gojargo/jargo-client-react), with
`NEXT_PUBLIC_JARGO_URL=http://localhost:8080`.

## Telephony

| Example | What it shows |
| --- | --- |
| [`twiliobot`](twiliobot) | An inbound phone agent over Twilio Media Streams, with the user-idle watchdog. |
| [`twilio/outbound`](twilio/outbound) | An outbound call: dial a number, collect a few details through a tool call, hang up. |

Both need a public URL Twilio can reach; see [Telephony](../docs/guides/telephony.md).
