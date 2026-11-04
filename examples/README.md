# Examples

Every example is a `main` package you run with `go run ./examples/<name>`. Each
file's doc comment lists the API keys it reads and how to drive it.

Turn-taking (Silero VAD and Smart Turn) needs the ONNX Runtime; see
[Installation](../docs/getting-started/installation.md). The getting-started
voice agents (06, 06a, 07) and the `voice`, `function-calling` and `transports`
examples do not start without it. `voicebot`, `flows` and `twiliobot` still run,
on STT endpointing, and lose barge-in.

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

## Voice

The voice agent on each provider's services, wired explicitly: speech-to-text, an
LLM and text-to-speech, with the user aggregator detecting speech through Silero
VAD. They need the ONNX Runtime, like the getting-started voice agents.

| Example | Services |
| --- | --- |
| [`voice-assemblyai`](voice/voice-assemblyai) | AssemblyAI speech-to-text and text-to-speech, with an OpenAI LLM. |
| [`voice-asyncai`](voice/voice-asyncai) | Deepgram speech-to-text, an OpenAI LLM and Async text-to-speech. |
| [`voice-aws`](voice/voice-aws) | AWS Transcribe speech-to-text, Claude on AWS Bedrock and AWS Polly text-to-speech. |
| [`voice-azure-http`](voice/voice-azure-http) | Azure speech-to-text, Azure OpenAI and Azure text-to-speech over HTTP. |
| [`voice-camb`](voice/voice-camb) | Deepgram speech-to-text, an OpenAI LLM and Camb AI text-to-speech. |
| [`voice-cartesia`](voice/voice-cartesia) | Cartesia speech-to-text and text- to-speech, with an OpenAI LLM. |
| [`voice-cartesia-http`](voice/voice-cartesia-http) | Cartesia speech-to-text, an OpenAI LLM and Cartesia text-to-speech over HTTP. |
| [`voice-cartesia-turns`](voice/voice-cartesia-turns) | Cartesia speech-to-text that detects the turns itself, an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-deepgram-flux`](voice/voice-deepgram-flux) | Deepgram Flux speech-to-text and text-to-speech, with an OpenAI LLM. |
| [`voice-deepgram-http`](voice/voice-deepgram-http) | Deepgram speech-to-text and text-to-speech (over HTTP), with an OpenAI LLM. |
| [`voice-elevenlabs`](voice/voice-elevenlabs) | ElevenLabs realtime speech-to- text and streaming text-to-speech, with an OpenAI LLM. |
| [`voice-elevenlabs-http`](voice/voice-elevenlabs-http) | ElevenLabs speech-to-text and text-to-speech over HTTP, with an OpenAI LLM. |
| [`voice-fal`](voice/voice-fal) | Fal speech-to-text, an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-fish`](voice/voice-fish) | Deepgram speech-to-text, an OpenAI LLM and Fish Audio text-to-speech. |
| [`voice-gladia`](voice/voice-gladia) | Gladia speech-to-text, an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-gladia-vad`](voice/voice-gladia-vad) | Gladia (with its own voice activity detection) speech-to-text, an OpenAI LLM and Cartesia text-to- speech. |
| [`voice-gradium`](voice/voice-gradium) | Gradium speech-to-text and text-to- speech, with an OpenAI LLM. |
| [`voice-groq`](voice/voice-groq) | Groq speech-to-text, LLM and text-to- speech. |
| [`voice-hume`](voice/voice-hume) | Deepgram speech-to-text, an OpenAI LLM and Hume text-to-speech. |
| [`voice-inworld-http`](voice/voice-inworld-http) | Deepgram speech-to-text, an OpenAI LLM and Inworld (over HTTP) text-to-speech. |
| [`voice-kokoro`](voice/voice-kokoro) | Deepgram speech-to-text, an OpenAI LLM and Kokoro text-to-speech. |
| [`voice-minimax`](voice/voice-minimax) | Deepgram speech-to-text, an OpenAI LLM and MiniMax text-to-speech. |
| [`voice-mistral`](voice/voice-mistral) | Mistral speech-to-text and text-to- speech, with an OpenAI LLM. |
| [`voice-neuphonic`](voice/voice-neuphonic) | Deepgram speech-to-text, an OpenAI LLM and Neuphonic text-to-speech. |
| [`voice-nvidia`](voice/voice-nvidia) | NVIDIA speech-to-text, LLM and text- to-speech. |
| [`voice-nvidia-segmented`](voice/voice-nvidia-segmented) | NVIDIA speech-to-text (transcribing each utterance once it ends), LLM and text-to-speech. |
| [`voice-openai`](voice/voice-openai) | OpenAI's realtime transcription, LLM and text-to-speech. |
| [`voice-openai-http`](voice/voice-openai-http) | OpenAI transcription, LLM and text-to-speech, transcribing over HTTP. |
| [`voice-openai-responses`](voice/voice-openai-responses) | Deepgram speech-to-text, OpenAI's Responses API and Cartesia text-to-speech. |
| [`voice-openai-responses-http`](voice/voice-openai-responses-http) | Deepgram speech-to- text, OpenAI's Responses API over HTTP and Cartesia text-to-speech. |
| [`voice-openai-turn-detection`](voice/voice-openai-turn-detection) | OpenAI's realtime transcription with the server's VAD driving turns, an OpenAI LLM and OpenAI text-to-speech. |
| [`voice-piper`](voice/voice-piper) | Deepgram speech-to-text, an OpenAI LLM and Piper text-to-speech. |
| [`voice-pockettts`](voice/voice-pockettts) | Deepgram speech-to-text, an OpenAI LLM and Pocket TTS text-to-speech. |
| [`voice-resemble`](voice/voice-resemble) | Deepgram speech-to-text, an OpenAI LLM and Resemble AI text-to-speech. |
| [`voice-rime-http`](voice/voice-rime-http) | Deepgram speech-to-text, an OpenAI LLM and Rime (over HTTP) text-to-speech. |
| [`voice-sarvam`](voice/voice-sarvam) | Sarvam speech-to-text, LLM and text- to-speech. |
| [`voice-soniox`](voice/voice-soniox) | Soniox speech-to-text and text-to- speech, with an OpenAI LLM. |
| [`voice-together`](voice/voice-together) | Together AI speech-to-text, LLM and text-to-speech. |
| [`voice-xai`](voice/voice-xai) | XAI speech-to-text, Grok and xAI text- to-speech. |
| [`voice-xai-http`](voice/voice-xai-http) | Deepgram speech-to-text, Grok and xAI text-to-speech over HTTP. |
| [`voice-xtts`](voice/voice-xtts) | Deepgram speech-to-text, an OpenAI LLM and XTTS text-to-speech. |

## Function calling

The voice agent with tools the LLM calls, each tool carrying its own handler.

| Example | LLM | Speech-to-text, text-to-speech |
| --- | --- | --- |
| [`function-calling-anthropic`](function-calling/function-calling-anthropic) | Anthropic | Deepgram, Cartesia |
| [`function-calling-anthropic-async`](function-calling/function-calling-anthropic-async) | Anthropic, with tools that run while the conversation goes on, a deadline, and one the LLM can cancel | Deepgram, Cartesia |
| [`function-calling-anthropic-async-stream`](function-calling/function-calling-anthropic-async-stream) | Anthropic, with a tool that reports intermediate results while it runs | Deepgram, Cartesia |
| [`function-calling-aws`](function-calling/function-calling-aws) | Claude on AWS Bedrock | AWS Transcribe, AWS Polly |
| [`function-calling-azure`](function-calling/function-calling-azure) | Azure OpenAI | Deepgram, Cartesia |
| [`function-calling-cerebras`](function-calling/function-calling-cerebras) | Cerebras | Deepgram, Cartesia |
| [`function-calling-deepseek`](function-calling/function-calling-deepseek) | DeepSeek | Deepgram, Cartesia |
| [`function-calling-fireworks`](function-calling/function-calling-fireworks) | Fireworks AI | Deepgram, Cartesia |
| [`function-calling-google-async-stream`](function-calling/function-calling-google-async-stream) | Google Gemini, with a tool that reports intermediate results while it runs | Deepgram, Cartesia |
| [`function-calling-google-vertex`](function-calling/function-calling-google-vertex) | Gemini on Google Vertex AI | Deepgram, ElevenLabs |
| [`function-calling-grok`](function-calling/function-calling-grok) | xAI Grok | Deepgram, xAI |
| [`function-calling-groq`](function-calling/function-calling-groq) | Groq | Groq, Cartesia |
| [`function-calling-inception`](function-calling/function-calling-inception) | Inception | Deepgram, Cartesia |
| [`function-calling-mistral`](function-calling/function-calling-mistral) | Mistral | Deepgram, Cartesia |
| [`function-calling-nebius`](function-calling/function-calling-nebius) | Nebius | Deepgram, Cartesia |
| [`function-calling-novita`](function-calling/function-calling-novita) | Novita AI | Deepgram, Cartesia |
| [`function-calling-nvidia`](function-calling/function-calling-nvidia) | NVIDIA | Deepgram, Cartesia |
| [`function-calling-ollama`](function-calling/function-calling-ollama) | A local Ollama model | Deepgram, Cartesia |
| [`function-calling-openai`](function-calling/function-calling-openai) | OpenAI | OpenAI realtime transcription, OpenAI |
| [`function-calling-openai-async`](function-calling/function-calling-openai-async) | OpenAI, with tools that run while the conversation goes on, a deadline, and one the LLM can cancel | OpenAI realtime transcription, OpenAI |
| [`function-calling-openai-async-stream`](function-calling/function-calling-openai-async-stream) | OpenAI, with a tool that reports intermediate results while it runs | Deepgram, Cartesia |
| [`function-calling-openai-responses`](function-calling/function-calling-openai-responses) | OpenAI Responses API | Deepgram, Cartesia |
| [`function-calling-openai-responses-async`](function-calling/function-calling-openai-responses-async) | OpenAI Responses API, with tools that run while the conversation goes on, a deadline, and one the LLM can cancel | Deepgram, Cartesia |
| [`function-calling-openai-responses-async-stream`](function-calling/function-calling-openai-responses-async-stream) | OpenAI Responses API, with a tool that reports intermediate results while it runs | Deepgram, Cartesia |
| [`function-calling-openai-responses-http`](function-calling/function-calling-openai-responses-http) | OpenAI Responses API over HTTP | Deepgram, Cartesia |
| [`function-calling-openrouter`](function-calling/function-calling-openrouter) | OpenRouter | Deepgram, Azure |
| [`function-calling-perplexity`](function-calling/function-calling-perplexity) | Perplexity Agent API | Deepgram, Cartesia |
| [`function-calling-qwen`](function-calling/function-calling-qwen) | Qwen | Deepgram, Cartesia |
| [`function-calling-sambanova`](function-calling/function-calling-sambanova) | SambaNova | Deepgram, Cartesia |
| [`function-calling-sarvam`](function-calling/function-calling-sarvam) | Sarvam AI | Sarvam, Sarvam |
| [`function-calling-together`](function-calling/function-calling-together) | Together AI | Deepgram, Cartesia |
| [`function-calling-missing-handler`](function-calling/function-calling-missing-handler) | OpenAI, with a tool deliberately left without a handler, to show how the LLM service recovers | Deepgram, Cartesia |

## Speech-to-speech

One model hears the caller and speaks its answers. These need the ONNX Runtime
too, for the user aggregator's default turn strategies.

| Example | What it shows | Keys |
| --- | --- | --- |
| [`realtime-openai`](realtime/realtime-openai) | OpenAI Realtime, with semantic turn detection, noise reduction, a tool added at runtime and transcript logging. | OpenAI |
| [`realtime-azure`](realtime/realtime-azure) | Azure OpenAI Realtime, with tools and transcript logging. | Azure OpenAI |
| [`realtime-grok`](realtime/realtime-grok) | xAI's Grok Voice Agent, with tools and transcript logging. | xAI |
| [`realtime-gemini-live`](realtime/realtime-gemini-live) | Gemini Live, with tools configured on the service, Google Search and transcript logging. | Google AI |
| [`realtime-gemini-live-vertex`](realtime/realtime-gemini-live-vertex) | Gemini Live through Vertex AI, with tools configured on the service. | Google Cloud |
| [`realtime-aws-nova-sonic`](realtime/realtime-aws-nova-sonic) | Amazon Nova Sonic, with a tool, session continuation and transcript logging. | AWS |

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

`flows` exposes only the WebRTC `POST /offer` endpoint. Drive it from a browser
client such as the `nextjs-voicebot` example in
[jargo-client-react](https://github.com/gojargo/jargo-client-react), with
`NEXT_PUBLIC_JARGO_URL=http://localhost:8080`.

## Telephony

| Example | What it shows |
| --- | --- |
| [`twiliobot`](twiliobot) | An inbound phone agent over Twilio Media Streams, with the user-idle watchdog. |
| [`twilio/outbound`](twilio/outbound) | An outbound call: dial a number, collect a few details through a tool call, hang up. |

Both need a public URL Twilio can reach; see [Telephony](../docs/guides/telephony.md).
