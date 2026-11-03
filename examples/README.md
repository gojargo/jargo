# Examples

Every example is a `main` package you run with `go run ./examples/<name>`. Each
file's doc comment lists the API keys it reads and how to drive it.

Turn-taking (Silero VAD and Smart Turn) needs the ONNX Runtime; see
[Installation](../docs/getting-started/installation.md). The getting-started
voice agents (06, 06a, 07) do not start without it. The other voice examples
still run, on STT endpointing, and lose barge-in.

## Getting started

Each step adds one idea to the one before. The bots run on the
[development runner](../docs/guides/runner.md): `go run` one, open
<http://localhost:7860>, and connect from the browser client it serves.

| Example | What it shows | Keys |
| --- | --- | --- |
| [`01-say-one-thing`](getting-started/01-say-one-thing) | Speak one sentence to whoever connects, then hang up. | Cartesia |
| [`01a-local-audio`](getting-started/01a-local-audio) | The same through the local speaker, no browser. | Cartesia |
| [`02-llm-say-one-thing`](getting-started/02-llm-say-one-thing) | An LLM writes the sentence. | OpenAI, Cartesia |
| [`06-voice-agent`](getting-started/06-voice-agent) | A full voice agent: speech-to-text, LLM, text-to-speech, with VAD. | Deepgram, OpenAI, Cartesia |
| [`06a-voice-agent-local`](getting-started/06a-voice-agent-local) | The voice agent on the local microphone and speaker. | Deepgram, OpenAI, Cartesia |
| [`07-function-calling`](getting-started/07-function-calling) | The voice agent with tools the LLM calls. | Deepgram, OpenAI, Cartesia |

The numbering follows the getting-started series these are ported from. Steps
03 to 05 show images and video, which jargo's transports do not carry, so they
are not here.

## Speech-to-speech

One model hears the caller and speaks its answers. These need the ONNX Runtime
too, for the user aggregator's default turn strategies.

| Example | What it shows | Keys |
| --- | --- | --- |
| [`realtime-azure`](realtime/realtime-azure) | Azure OpenAI Realtime, with tools and transcript logging. | Azure OpenAI |
| [`realtime-grok`](realtime/realtime-grok) | xAI's Grok Voice Agent, with tools and transcript logging. | xAI |

## Transports

| Example | What it shows | Keys |
| --- | --- | --- |
| [`transports-livekit`](transports/transports-livekit) | A voice agent that joins a LiveKit room on its own and takes text chat as speech. | LiveKit, Deepgram, OpenAI, Cartesia |
| [`transports-small-webrtc`](transports/transports-small-webrtc) | A voice agent that builds its WebRTC transport itself, from the runner's connection. | Deepgram, OpenAI, Cartesia |

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
