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
| [`05-speaking-state`](getting-started/05-speaking-state) | The voice agent shows one image as video while it speaks and another while it waits. | Deepgram, OpenAI, Cartesia |
| [`06-voice-agent`](getting-started/06-voice-agent) | A full voice agent: speech-to-text, LLM, text-to-speech, with VAD. | Deepgram, OpenAI, Cartesia |
| [`06a-voice-agent-local`](getting-started/06a-voice-agent-local) | The voice agent on the local microphone and speaker. | Deepgram, OpenAI, Cartesia |
| [`07-function-calling`](getting-started/07-function-calling) | The voice agent with tools the LLM calls. | Deepgram, OpenAI, Cartesia |

The numbering follows the getting-started series these are ported from. Steps
03, 03a and 04 need image generation services jargo does not have yet, so they
are not here.

## Voice

The voice agent on each provider's services, wired explicitly: speech-to-text, an
LLM and text-to-speech, with the user aggregator detecting speech through Silero
VAD. They need the ONNX Runtime, like the getting-started voice agents.

| Example | Services |
| --- | --- |
| [`voice-assemblyai`](voice/voice-assemblyai) | AssemblyAI speech-to-text and text-to-speech, with an OpenAI LLM. |
| [`voice-asyncai`](voice/voice-asyncai) | Deepgram speech-to-text, an OpenAI LLM and Async text-to-speech. |
| [`voice-asyncai-http`](voice/voice-asyncai-http) | Deepgram speech-to-text, an OpenAI LLM and Async text-to-speech (over HTTP). |
| [`voice-aws`](voice/voice-aws) | AWS Transcribe speech-to-text, Claude on AWS Bedrock and AWS Polly text-to-speech. |
| [`voice-azure`](voice/voice-azure) | Azure speech-to-text, Azure OpenAI and Azure text-to-speech (streaming). |
| [`voice-azure-http`](voice/voice-azure-http) | Azure speech-to-text, Azure OpenAI and Azure text-to-speech over HTTP. |
| [`voice-bland`](voice/voice-bland) | Deepgram speech-to-text, OpenAI's Responses API and Bland text-to-speech. |
| [`voice-bland-http`](voice/voice-bland-http) | Deepgram speech-to-text, OpenAI's Responses API and Bland text-to-speech (over HTTP). |
| [`voice-camb`](voice/voice-camb) | Deepgram speech-to-text, an OpenAI LLM and Camb AI text-to-speech. |
| [`voice-cartesia`](voice/voice-cartesia) | Cartesia speech-to-text and text- to-speech, with an OpenAI LLM. |
| [`voice-cartesia-http`](voice/voice-cartesia-http) | Cartesia speech-to-text, an OpenAI LLM and Cartesia text-to-speech over HTTP. |
| [`voice-cartesia-turns`](voice/voice-cartesia-turns) | Cartesia speech-to-text that detects the turns itself, an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-deepgram`](voice/voice-deepgram) | Deepgram speech-to-text and text-to-speech (over WebSocket), with an OpenAI LLM. |
| [`voice-deepgram-flux`](voice/voice-deepgram-flux) | Deepgram Flux speech-to-text and text-to-speech, with an OpenAI LLM. |
| [`voice-deepgram-flux-eager`](voice/voice-deepgram-flux-eager) | Deepgram Flux speech-to-text and text-to-speech, with an OpenAI LLM, answering Flux's eager end of turn while the LLM's turn-completion markers hold incomplete turns open. |
| [`voice-deepgram-http`](voice/voice-deepgram-http) | Deepgram speech-to-text and text-to-speech (over HTTP), with an OpenAI LLM. |
| [`voice-elevenlabs`](voice/voice-elevenlabs) | ElevenLabs realtime speech-to- text and streaming text-to-speech, with an OpenAI LLM. |
| [`voice-elevenlabs-http`](voice/voice-elevenlabs-http) | ElevenLabs speech-to-text and text-to-speech over HTTP, with an OpenAI LLM. |
| [`voice-fal`](voice/voice-fal) | Fal speech-to-text, an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-fish`](voice/voice-fish) | Deepgram speech-to-text, an OpenAI LLM and Fish Audio text-to-speech. |
| [`voice-gladia`](voice/voice-gladia) | Gladia speech-to-text, an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-gladia-vad`](voice/voice-gladia-vad) | Gladia (with its own voice activity detection) speech-to-text, an OpenAI LLM and Cartesia text-to- speech. |
| [`voice-google`](voice/voice-google) | Google Cloud speech-to-text and streaming text-to-speech, with a Gemini LLM. |
| [`voice-google-http`](voice/voice-google-http) | Google Cloud speech-to-text, a Gemini LLM and Google Cloud text-to-speech over HTTP. |
| [`voice-google-interactions`](voice/voice-google-interactions) | Deepgram speech-to-text and Cartesia text-to-speech, with a Gemini LLM on the Interactions API. |
| [`voice-gradium`](voice/voice-gradium) | Gradium speech-to-text and text-to- speech, with an OpenAI LLM. |
| [`voice-gradium-turns`](voice/voice-gradium-turns) | Gradium speech-to-text that detects the turns itself, an OpenAI LLM and Gradium text-to-speech. |
| [`voice-groq`](voice/voice-groq) | Groq speech-to-text, LLM and text-to- speech. |
| [`voice-hume`](voice/voice-hume) | Deepgram speech-to-text, an OpenAI LLM and Hume text-to-speech. |
| [`voice-inworld`](voice/voice-inworld) | Deepgram speech-to-text, an OpenAI LLM and Inworld text-to-speech (over WebSocket). |
| [`voice-inworld-http`](voice/voice-inworld-http) | Deepgram speech-to-text, an OpenAI LLM and Inworld (over HTTP) text-to-speech. |
| [`voice-kokoro`](voice/voice-kokoro) | Deepgram speech-to-text, an OpenAI LLM and local Kokoro text-to-speech (espeak-ng and the ONNX Runtime, no server). |
| [`voice-minimax`](voice/voice-minimax) | Deepgram speech-to-text, an OpenAI LLM and MiniMax text-to-speech. |
| [`voice-moonshine`](voice/voice-moonshine) | Local Moonshine speech-to-text (libmoonshine, no server), an OpenAI LLM and Cartesia text-to-speech. |
| [`voice-mistral`](voice/voice-mistral) | Mistral speech-to-text and text-to- speech, with an OpenAI LLM. |
| [`voice-neuphonic`](voice/voice-neuphonic) | Deepgram speech-to-text, an OpenAI LLM and Neuphonic text-to-speech. |
| [`voice-neuphonic-http`](voice/voice-neuphonic-http) | Deepgram speech-to-text, an OpenAI LLM and Neuphonic text-to-speech (over HTTP). |
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
| [`voice-rime`](voice/voice-rime) | Rime speech-to-text, an OpenAI LLM and Rime text-to-speech (over WebSocket). |
| [`voice-rime-http`](voice/voice-rime-http) | Rime speech-to-text, an OpenAI LLM and Rime text-to-speech (over HTTP). |
| [`voice-rime-v1`](voice/voice-rime-v1) | Rime speech-to-text, an OpenAI LLM and Rime text-to-speech (over WebSocket v1). |
| [`voice-sarvam`](voice/voice-sarvam) | Sarvam speech-to-text, LLM and text- to-speech. |
| [`voice-sarvam-vad-http`](voice/voice-sarvam-vad-http) | Sarvam speech-to-text (with its VAD signals), LLM and text-to-speech (over HTTP). |
| [`voice-smallest`](voice/voice-smallest) | Smallest AI speech-to-text, LLM (Electron) and text-to-speech. |
| [`voice-soniox`](voice/voice-soniox) | Soniox speech-to-text and text-to- speech, with an OpenAI LLM. |
| [`voice-speechify-http`](voice/voice-speechify-http) | Deepgram speech-to-text, an OpenAI LLM and Speechify text-to-speech (over HTTP). |
| [`voice-speechmatics`](voice/voice-speechmatics) | Speechmatics speech-to-text (with diarization) and text-to-speech, with an OpenAI LLM. |
| [`voice-speechmatics-vad`](voice/voice-speechmatics-vad) | Speechmatics speech-to-text (with diarization and its own voice activity detection) and text-to-speech, with an OpenAI LLM. |
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
| [`function-calling-anthropic-video`](function-calling/function-calling-anthropic-video) | Anthropic, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-advanced-functionschema`](function-calling/function-calling-advanced-functionschema) | OpenAI, with a tool schema written out by hand (an enum parameter) | OpenAI realtime transcription, OpenAI |
| [`function-calling-aws`](function-calling/function-calling-aws) | Claude on AWS Bedrock | AWS Transcribe, AWS Polly |
| [`function-calling-aws-video`](function-calling/function-calling-aws-video) | Claude on AWS Bedrock, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-azure`](function-calling/function-calling-azure) | Azure OpenAI | Deepgram, Cartesia |
| [`function-calling-baseten`](function-calling/function-calling-baseten) | Baseten | Deepgram, Cartesia |
| [`function-calling-cerebras`](function-calling/function-calling-cerebras) | Cerebras | Deepgram, Cartesia |
| [`function-calling-crusoe`](function-calling/function-calling-crusoe) | Crusoe Cloud | Deepgram, Cartesia |
| [`function-calling-deepseek`](function-calling/function-calling-deepseek) | DeepSeek | Deepgram, Cartesia |
| [`function-calling-direct`](function-calling/function-calling-direct) | OpenAI, with tools that carry their own handlers | Deepgram, Cartesia |
| [`function-calling-fireworks`](function-calling/function-calling-fireworks) | Fireworks AI | Deepgram, Cartesia |
| [`function-calling-google`](function-calling/function-calling-google) | Google Gemini, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-google-async`](function-calling/function-calling-google-async) | Google Gemini, with tools that run while the conversation goes on, a deadline, one the LLM can cancel, and a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-google-async-stream`](function-calling/function-calling-google-async-stream) | Google Gemini, with a tool that reports intermediate results while it runs | Deepgram, Cartesia |
| [`function-calling-google-interactions`](function-calling/function-calling-google-interactions) | Google Gemini on the Interactions API | Deepgram, Cartesia |
| [`function-calling-google-interactions-async`](function-calling/function-calling-google-interactions-async) | Google Gemini on the Interactions API, with tools that run while the conversation goes on, a deadline, one the LLM can cancel, and a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-google-interactions-vertex`](function-calling/function-calling-google-interactions-vertex) | Gemini on Google Vertex AI, through the Interactions API | Deepgram, ElevenLabs |
| [`function-calling-google-interactions-video`](function-calling/function-calling-google-interactions-video) | Google Gemini on the Interactions API, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-google-video`](function-calling/function-calling-google-video) | Google Gemini, with a tool that looks at the user's camera | Deepgram, Cartesia |
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
| [`function-calling-openai-video`](function-calling/function-calling-openai-video) | OpenAI, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-openai-responses`](function-calling/function-calling-openai-responses) | OpenAI Responses API | Deepgram, Cartesia |
| [`function-calling-openai-responses-async`](function-calling/function-calling-openai-responses-async) | OpenAI Responses API, with tools that run while the conversation goes on, a deadline, and one the LLM can cancel | Deepgram, Cartesia |
| [`function-calling-openai-responses-async-stream`](function-calling/function-calling-openai-responses-async-stream) | OpenAI Responses API, with a tool that reports intermediate results while it runs | Deepgram, Cartesia |
| [`function-calling-openai-responses-http`](function-calling/function-calling-openai-responses-http) | OpenAI Responses API over HTTP | Deepgram, Cartesia |
| [`function-calling-openai-responses-video`](function-calling/function-calling-openai-responses-video) | OpenAI's Responses API, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-openai-responses-video-http`](function-calling/function-calling-openai-responses-video-http) | OpenAI's Responses API over HTTP, with a tool that looks at the user's camera | Deepgram, Cartesia |
| [`function-calling-openrouter`](function-calling/function-calling-openrouter) | OpenRouter | Deepgram, Azure |
| [`function-calling-perplexity`](function-calling/function-calling-perplexity) | Perplexity Agent API | Deepgram, Cartesia |
| [`function-calling-qwen`](function-calling/function-calling-qwen) | Qwen | Deepgram, Cartesia |
| [`function-calling-sambanova`](function-calling/function-calling-sambanova) | SambaNova | Deepgram, Cartesia |
| [`function-calling-sarvam`](function-calling/function-calling-sarvam) | Sarvam AI | Sarvam, Sarvam |
| [`function-calling-smallest`](function-calling/function-calling-smallest) | Smallest AI (Electron) | Smallest AI, Smallest AI |
| [`function-calling-together`](function-calling/function-calling-together) | Together AI | Deepgram, Cartesia |
| [`function-calling-missing-handler`](function-calling/function-calling-missing-handler) | OpenAI, with a tool deliberately left without a handler, to show how the LLM service recovers | Deepgram, Cartesia |

## Speech-to-speech

One model hears the caller and speaks its answers. These need the ONNX Runtime
too, for the user aggregator's default turn strategies.

| Example | What it shows | Keys |
| --- | --- | --- |
| [`realtime-openai`](realtime/realtime-openai) | OpenAI Realtime, with semantic turn detection, noise reduction, a tool added at runtime and transcript logging. | OpenAI |
| [`realtime-openai-async-tool`](realtime/realtime-openai-async-tool) | OpenAI Realtime, with a slow tool that runs while the conversation goes on. | OpenAI |
| [`realtime-openai-locally-driven-turns`](realtime/realtime-openai-locally-driven-turns) | OpenAI Realtime with server turn detection off and turns driven by a local VAD, a tool added at runtime and transcript logging. | OpenAI |
| [`realtime-openai-text`](realtime/realtime-openai-text) | OpenAI Realtime answering in text, spoken by Cartesia, with tools. | OpenAI, Cartesia |
| [`realtime-azure`](realtime/realtime-azure) | Azure OpenAI Realtime, with tools and transcript logging. | Azure OpenAI |
| [`realtime-azure-async-tool`](realtime/realtime-azure-async-tool) | Azure OpenAI Realtime, with a slow tool that runs while the conversation goes on. | Azure OpenAI |
| [`realtime-grok`](realtime/realtime-grok) | xAI's Grok Voice Agent, with tools and transcript logging. | xAI |
| [`realtime-grok-async-tool`](realtime/realtime-grok-async-tool) | xAI's Grok Voice Agent, with a slow tool that runs while the conversation goes on. | xAI |
| [`realtime-grok-locally-driven-turns`](realtime/realtime-grok-locally-driven-turns) | xAI's Grok Voice Agent with server turn detection off and turns driven by a local VAD, with tools and transcript logging. | xAI |
| [`realtime-gemini-live`](realtime/realtime-gemini-live) | Gemini Live, with tools configured on the service, Google Search and transcript logging. | Google AI |
| [`realtime-gemini-live-async-tool`](realtime/realtime-gemini-live-async-tool) | Gemini Live, with a slow tool that runs while the conversation goes on. | Google AI |
| [`realtime-gemini-live-google-search`](realtime/realtime-gemini-live-google-search) | Gemini Live answering with the latest news through Google Search. | Google AI |
| [`realtime-gemini-live-graceful-end`](realtime/realtime-gemini-live-graceful-end) | Gemini Live ending the conversation itself through a tool, then saying goodbye. | Google AI |
| [`realtime-gemini-live-grounding-metadata`](realtime/realtime-gemini-live-grounding-metadata) | Gemini Live with Google Search and code execution, logging the grounding metadata of each searched answer. | Google AI |
| [`realtime-gemini-live-locally-driven-turns`](realtime/realtime-gemini-live-locally-driven-turns) | Gemini Live with server VAD off and turns driven by a local VAD, with transcript logging. | Google AI |
| [`realtime-gemini-live-vertex`](realtime/realtime-gemini-live-vertex) | Gemini Live through Vertex AI, with tools configured on the service. | Google Cloud |
| [`realtime-aws-nova-sonic`](realtime/realtime-aws-nova-sonic) | Amazon Nova Sonic, with a tool, session continuation and transcript logging. | AWS |
| [`realtime-aws-nova-sonic-async-tool`](realtime/realtime-aws-nova-sonic-async-tool) | Amazon Nova Sonic, with a slow tool that runs while the conversation goes on. | AWS |

## Features

| Example | What it shows | Keys |
| --- | --- | --- |
| [`features-dtmf-menu`](features/features-dtmf-menu) | A phone menu driven by keypad presses only: a DTMF aggregator turns each key sequence into a transcription for the LLM, with no speech-to-text. Runs on a telephony transport. | OpenAI, Cartesia |
| [`features-voicemail-detection`](features/features-voicemail-detection) | Tells a person from a voicemail greeting with a Jev classifier, holds the bot's speech until it knows, and leaves a message on voicemail. | Deepgram, OpenAI, Cartesia, TypeSafe |
| [`features-add-tool-change-messages`](features/features-add-tool-change-messages) | Tool change messages announcing toolset changes made with LLMSetToolsFrame mid-conversation (Ollama). | Deepgram, Cartesia |
| [`features-app-resources`](features/features-app-resources) | Sharing application resources with tool handlers and custom processors through WorkerConfig.AppResources. | Deepgram, Cartesia, OpenAI |
| [`features-before-and-after-events`](features/features-before-and-after-events) | Processor before-process and after-push frame events used for synchronization. | Deepgram, Cartesia, OpenAI |
| [`features-concurrent-llm-evaluation`](features/features-concurrent-llm-evaluation) | Two LLMs answering the same turns in a ParallelPipeline, with one shared UserTurnProcessor. | Deepgram, Cartesia, OpenAI, Groq |
| [`features-concurrent-llm-rtvi-ignored-sources`](features/features-concurrent-llm-rtvi-ignored-sources) | Keeping a silent evaluator branch's RTVI messages from the client with IgnoredSources. | Deepgram, Cartesia, OpenAI |
| [`features-custom-frame-processor`](features/features-custom-frame-processor) | A custom processor that pretty-prints metrics frames. | Deepgram, Cartesia, OpenAI |
| [`features-live-translation`](features/features-live-translation) | English to Spanish live translation, with a turn opened on every transcript. | Deepgram, Cartesia, OpenAI |
| [`features-pattern-pair-voice-switching`](features/features-pattern-pair-voice-switching) | Switching TTS voices on `<voice>` tags with a PatternPairAggregator. | Deepgram, Cartesia, OpenAI |
| [`features-processor-usable`](features/features-processor-usable) | Processor health events, error categories and TTS failover. | Deepgram, Cartesia, ElevenLabs, OpenAI |
| [`features-pronunciation-dictionary`](features/features-pronunciation-dictionary) | Saying medication names right with an IPA pronunciation transform. | Cartesia, OpenAI |
| [`features-screen-share`](features/features-screen-share) | Tools that fetch a camera or screen-share frame on request for the LLM to describe. | Deepgram, Cartesia, OpenAI |
| [`features-service-switcher`](features/features-service-switcher) | Manually switching the STT, the LLM and the TTS between providers while the bot runs. | Cartesia, Deepgram, OpenAI, Google AI |
| [`features-switch-languages`](features/features-switch-languages) | A tool that switches between English and Spanish voices behind function filters. | Deepgram, Cartesia, OpenAI |
| [`features-switch-voices`](features/features-switch-voices) | A tool that switches between three voices behind function filters. | Deepgram, Cartesia, OpenAI |
| [`features-text-transforms`](features/features-text-transforms) | Composing text transforms, and stripping emoji in an LLM text processor. | Cartesia, ElevenLabs, OpenAI |
| [`features-voice-formatter`](features/features-voice-formatter) | The voice formatter reading amounts, dates and acronyms naturally. | Cartesia, ElevenLabs, OpenAI |
| [`features-wake-phrase`](features/features-wake-phrase) | Answering only after a wake phrase. | Deepgram, Cartesia, OpenAI |

## MCP

| Example | What it shows | Keys |
| --- | --- | --- |
| [`mcp-stdio`](mcp/mcp-stdio) | Tools from the memory MCP server, started over stdio with npx. Needs Node.js. | Deepgram, Anthropic, Cartesia |
| [`mcp-streamable-http`](mcp/mcp-streamable-http) | Tools from GitHub's MCP server over streamable HTTP, authenticated with a header. | GitHub, Deepgram, Google AI, Cartesia |
| [`mcp-multiple-mcp`](mcp/mcp-multiple-mcp) | Tools from two MCP servers at once: memory over stdio and GitHub over streamable HTTP. | GitHub, Deepgram, Anthropic, Cartesia |
| [`mcp-streamable-http-gemini-live`](mcp/mcp-streamable-http-gemini-live) | GitHub MCP tools over streamable HTTP with Gemini Live. | GitHub, Google AI |

## Vision

An image and a question about it open the conversation, read from the session
body (`image_path`, `question`) and defaulting to [`assets/cat.jpg`](vision/assets/cat.jpg);
the bot describes the image, then talks by voice.

| Example | LLM | Keys |
| --- | --- | --- |
| [`vision-anthropic`](vision/vision-anthropic) | Anthropic | Deepgram, Anthropic, Cartesia |
| [`vision-aws`](vision/vision-aws) | Claude on AWS Bedrock | Deepgram, AWS, Cartesia |
| [`vision-gemini-flash`](vision/vision-gemini-flash) | Google Gemini | Deepgram, Google AI, Cartesia |
| [`vision-google-interactions`](vision/vision-google-interactions) | Google Gemini on the Interactions API | Deepgram, Google AI, Cartesia |
| [`vision-openai`](vision/vision-openai) | OpenAI | Deepgram, OpenAI, Cartesia |
| [`vision-openai-responses`](vision/vision-openai-responses) | OpenAI Responses API | Deepgram, OpenAI, Cartesia |
| [`vision-openai-responses-http`](vision/vision-openai-responses-http) | OpenAI Responses API over HTTP | Deepgram, OpenAI, Cartesia |

## Video processing

| Example | What it shows | Keys |
| --- | --- | --- |
| [`video-processing-mirror`](video-processing/video-processing-mirror) | Sends the user's audio and camera straight back to them. | None |

## Turn management

| Example | What it shows | Keys |
| --- | --- | --- |
| [`turn-management-classifier-turn-completion`](turn-management/turn-management-classifier-turn-completion) | A Jev classifier deciding that the user's turn is complete. | Deepgram, Cartesia, OpenAI, TypeSafe |
| [`turn-management-detect-user-idle`](turn-management/turn-management-detect-user-idle) | Escalating idle prompts, then ending, with idle detection toggled at runtime. | Deepgram, Cartesia, OpenAI |
| [`turn-management-eager-end-of-turn`](turn-management/turn-management-eager-end-of-turn) | Answering Deepgram Flux's predicted end of turn ahead of the committed one. | Deepgram, Cartesia, OpenAI |
| [`turn-management-empty-user-turn`](turn-management/turn-management-empty-user-turn) | Recovering from a user turn that ends with no transcript. | Deepgram, Cartesia, OpenAI |
| [`turn-management-filter-incomplete-turns`](turn-management/turn-management-filter-incomplete-turns) | LLM turn-completion markers suppressing answers to incomplete turns. | Deepgram, Cartesia, OpenAI |
| [`turn-management-filter-incomplete-turns-function-calling`](turn-management/turn-management-filter-incomplete-turns-function-calling) | Filtering incomplete turns with a tool the LLM can call. | Deepgram, Cartesia, OpenAI |
| [`turn-management-filter-incomplete-turns-user-idle`](turn-management/turn-management-filter-incomplete-turns-user-idle) | Filtering incomplete turns combined with user idle re-prompts. | Deepgram, Cartesia, OpenAI |
| [`turn-management-interruption-config`](turn-management/turn-management-interruption-config) | Requiring three words before the user interrupts, with a transcription log. | Deepgram, Cartesia, OpenAI |
| [`turn-management-smart-turn-local`](turn-management/turn-management-smart-turn-local) | Local Smart Turn end-of-turn detection, logging the turn metrics. | Deepgram, Cartesia, OpenAI |
| [`turn-management-user-assistant-turns`](turn-management/turn-management-user-assistant-turns) | A transcript kept from the user and assistant turn events. | Deepgram, Cartesia, OpenAI |
| [`turn-management-user-mute-strategy`](turn-management/turn-management-user-mute-strategy) | Muting the user until the first bot reply and during tool calls. | Deepgram, OpenAI |

## Thinking

The thoughts the model reports are logged as they arrive.

| Example | What it shows | Keys |
| --- | --- | --- |
| [`thinking-anthropic`](thinking/thinking-anthropic) | Anthropic extended thinking. | Deepgram, Cartesia, Anthropic |
| [`thinking-functions-anthropic`](thinking/thinking-functions-anthropic) | Anthropic thinking with function calling. | Deepgram, Cartesia, Anthropic |
| [`thinking-google`](thinking/thinking-google) | Gemini thinking, with thoughts included. | Deepgram, Cartesia, Google AI |
| [`thinking-functions-google`](thinking/thinking-functions-google) | Gemini thinking with function calling. | Deepgram, Cartesia, Google AI |
| [`thinking-google-vertex`](thinking/thinking-google-vertex) | Gemini thinking on Vertex AI. | Deepgram, Cartesia, Google Cloud |
| [`thinking-functions-google-vertex`](thinking/thinking-functions-google-vertex) | Vertex Gemini thinking with function calling. | Deepgram, Cartesia, Google Cloud |
| [`thinking-functions-google-interactions`](thinking/thinking-functions-google-interactions) | Gemini on the Interactions API thinks before it checks a flight and books a taxi, with each thought summary logged. | Deepgram, Cartesia, Google AI |
| [`thinking-openai-responses`](thinking/thinking-openai-responses) | OpenAI Responses reasoning summaries over WebSocket. | Deepgram, Cartesia, OpenAI |
| [`thinking-openai-responses-http`](thinking/thinking-openai-responses-http) | OpenAI Responses reasoning summaries over HTTP. | Deepgram, Cartesia, OpenAI |
| [`thinking-functions-openai-responses`](thinking/thinking-functions-openai-responses) | OpenAI Responses reasoning with function calling. | Deepgram, Cartesia, OpenAI |

## Transcription

Each transcribes the user with one STT service, logging the interim and final transcripts; nothing answers.

| Example | Speech-to-text | Keys |
| --- | --- | --- |
| [`transcription-assemblyai`](transcription/transcription-assemblyai) | AssemblyAI. | AssemblyAI |
| [`transcription-azure`](transcription/transcription-azure) | Azure Speech. | Azure Speech |
| [`transcription-cartesia`](transcription/transcription-cartesia) | Cartesia. | Cartesia |
| [`transcription-cartesia-turns`](transcription/transcription-cartesia-turns) | Cartesia, with its own turn detection. | Cartesia |
| [`transcription-deepgram`](transcription/transcription-deepgram) | Deepgram. | Deepgram |
| [`transcription-deepgram-flux`](transcription/transcription-deepgram-flux) | Deepgram Flux. | Deepgram |
| [`transcription-elevenlabs`](transcription/transcription-elevenlabs) | ElevenLabs realtime. | ElevenLabs |
| [`transcription-gladia`](transcription/transcription-gladia) | Gladia. | Gladia |
| [`transcription-google`](transcription/transcription-google) | Google Cloud Speech-to-Text. | Google Cloud |
| [`transcription-google-gemini`](transcription/transcription-google-gemini) | Gemini. | Google AI |
| [`transcription-gradium`](transcription/transcription-gradium) | Gradium. | Gradium |
| [`transcription-mistral`](transcription/transcription-mistral) | Mistral. | Mistral |
| [`transcription-nvidia`](transcription/transcription-nvidia) | NVIDIA Riva. | NVIDIA |
| [`transcription-openai`](transcription/transcription-openai) | OpenAI realtime transcription. | OpenAI |
| [`transcription-rime`](transcription/transcription-rime) | Rime. | Rime |
| [`transcription-soniox`](transcription/transcription-soniox) | Soniox. | Soniox |
| [`transcription-speechmatics`](transcription/transcription-speechmatics) | Speechmatics, with speakers tagged by diarization. | Speechmatics |
| [`transcription-together`](transcription/transcription-together) | Together AI. | Together AI |
| [`transcription-xai`](transcription/transcription-xai) | xAI. | xAI |

## Audio

| Example | What it shows | Keys |
| --- | --- | --- |
| [`audio-recording`](audio/audio-recording) | Recording the conversation to merged, user and bot WAV files. | Deepgram, Cartesia, OpenAI |
| [`audio-sound-effects`](audio/audio-sound-effects) | Playing sound effects at each change of turn. | Deepgram, Cartesia, OpenAI |
| [`audio-volume`](audio/audio-volume) | Changing speech volume with VolumeGainFrame and VolumeFrame. | Deepgram |

## Expressive conversation

The LLM writes expression tags that the TTS performs; see
[Expressive conversation](../docs/guides/expressive-conversation.md).

| Example | What it shows | Keys |
| --- | --- | --- |
| [`expressive-conversation-azure`](expressive-conversation/expressive-conversation-azure) | The LLM writes expression tags and an Azure HD voice performs style markers and sounds. | Deepgram, OpenAI, Azure |
| [`expressive-conversation-cartesia`](expressive-conversation/expressive-conversation-cartesia) | Cartesia emotions, with a policy offering 16 deliveries. | Cartesia, OpenAI |
| [`expressive-conversation-fish`](expressive-conversation/expressive-conversation-fish) | Fish Audio S2 tags. | Deepgram, OpenAI, Fish Audio |
| [`expressive-conversation-inworld`](expressive-conversation/expressive-conversation-inworld) | Inworld TTS-2. | Deepgram, OpenAI, Inworld |
| [`expressive-conversation-minimax`](expressive-conversation/expressive-conversation-minimax) | MiniMax speech-2.8, offering its whole vocabulary. | Deepgram, OpenAI, MiniMax |
| [`expressive-conversation-mistral`](expressive-conversation/expressive-conversation-mistral) | Mistral Voxtral 3 emotion and pace tags. | Deepgram, OpenAI, Mistral |
| [`expressive-conversation-soniox`](expressive-conversation/expressive-conversation-soniox) | Soniox tts-rt-v2 audio tags. | Deepgram, OpenAI, Soniox |

## Context summarization

| Example | What it shows | Keys |
| --- | --- | --- |
| [`context-summarization-openai`](context-summarization/context-summarization-openai) | Automatic context summarization. | Deepgram, Cartesia, OpenAI |
| [`context-summarization-google`](context-summarization/context-summarization-google) | Automatic context summarization with Gemini. | Deepgram, Cartesia, Google AI |
| [`context-summarization-dedicated-llm`](context-summarization/context-summarization-dedicated-llm) | Summarization on a dedicated Gemini model, with a custom prompt, template and timeout. | Deepgram, Cartesia, OpenAI, Google AI |
| [`context-summarization-manual-openai`](context-summarization/context-summarization-manual-openai) | Summarization on demand through a function call. | Deepgram, Cartesia, OpenAI |

## Persistent context

Tools save, list and load the conversation as files.

| Example | What it shows | Keys |
| --- | --- | --- |
| [`persistent-context-openai`](persistent-context/persistent-context-openai) | Saving, listing and loading conversations with tools, on OpenAI. | Deepgram, Cartesia, OpenAI |
| [`persistent-context-openai-responses`](persistent-context/persistent-context-openai-responses) | The same on the OpenAI Responses API. | Deepgram, Cartesia, OpenAI |
| [`persistent-context-openai-responses-http`](persistent-context/persistent-context-openai-responses-http) | The same on the OpenAI Responses API over HTTP. | Deepgram, Cartesia, OpenAI |
| [`persistent-context-anthropic`](persistent-context/persistent-context-anthropic) | The same on Anthropic. | Deepgram, Cartesia, Anthropic |
| [`persistent-context-gemini`](persistent-context/persistent-context-gemini) | The same on Gemini, plus a camera tool. | Deepgram, Cartesia, Google AI |
| [`persistent-context-aws-nova-sonic`](persistent-context/persistent-context-aws-nova-sonic) | The same on Amazon Nova Sonic; loading resets the session. | AWS |

## RAG and observability

| Example | What it shows | Keys |
| --- | --- | --- |
| [`rag-gemini-grounding-metadata`](rag/rag-gemini-grounding-metadata) | Google Search grounding with Gemini, logging the search responses. | Deepgram, Cartesia, Google AI |
| [`observability-heartbeats`](observability/observability-heartbeats) | Heartbeat monitoring catching a stalled pipeline. | None |
| [`observability-observer`](observability/observability-observer) | A custom observer alongside the LLM log and filtered debug log observers. | Deepgram, Cartesia, OpenAI |

## Update settings

Each changes its service's settings while the conversation runs, starting ten
seconds after the client connects, with the frame that service takes.

| Example | Service |
| --- | --- |
| [`llm-anthropic`](update-settings/llm/llm-anthropic) | Anthropic LLM |
| [`llm-aws-bedrock`](update-settings/llm/llm-aws-bedrock) | AWS Bedrock LLM |
| [`llm-aws-nova-sonic`](update-settings/llm/llm-aws-nova-sonic) | Amazon Nova Sonic LLM |
| [`llm-azure`](update-settings/llm/llm-azure) | Azure OpenAI LLM |
| [`llm-azure-realtime`](update-settings/llm/llm-azure-realtime) | Azure OpenAI Realtime LLM |
| [`llm-cerebras`](update-settings/llm/llm-cerebras) | Cerebras LLM |
| [`llm-deepseek`](update-settings/llm/llm-deepseek) | DeepSeek LLM |
| [`llm-fireworks`](update-settings/llm/llm-fireworks) | Fireworks LLM |
| [`llm-gemini-live`](update-settings/llm/llm-gemini-live) | Gemini Live LLM |
| [`llm-gemini-live-vertex`](update-settings/llm/llm-gemini-live-vertex) | Gemini Live on Vertex AI LLM |
| [`llm-google`](update-settings/llm/llm-google) | Google Gemini LLM |
| [`llm-google-vertex`](update-settings/llm/llm-google-vertex) | Gemini on Vertex AI LLM |
| [`llm-grok`](update-settings/llm/llm-grok) | xAI Grok LLM |
| [`llm-groq`](update-settings/llm/llm-groq) | Groq LLM |
| [`llm-mistral`](update-settings/llm/llm-mistral) | Mistral LLM |
| [`llm-nvidia`](update-settings/llm/llm-nvidia) | NVIDIA LLM |
| [`llm-ollama`](update-settings/llm/llm-ollama) | Ollama LLM |
| [`llm-openai`](update-settings/llm/llm-openai) | OpenAI LLM |
| [`llm-openai-realtime`](update-settings/llm/llm-openai-realtime) | OpenAI Realtime LLM |
| [`llm-openai-responses`](update-settings/llm/llm-openai-responses) | OpenAI Responses API LLM |
| [`llm-openai-responses-http`](update-settings/llm/llm-openai-responses-http) | OpenAI Responses API over HTTP LLM |
| [`llm-openrouter`](update-settings/llm/llm-openrouter) | OpenRouter LLM |
| [`llm-perplexity`](update-settings/llm/llm-perplexity) | Perplexity LLM |
| [`llm-qwen`](update-settings/llm/llm-qwen) | Qwen LLM |
| [`llm-sambanova`](update-settings/llm/llm-sambanova) | SambaNova LLM |
| [`llm-sarvam`](update-settings/llm/llm-sarvam) | Sarvam LLM |
| [`llm-smallest`](update-settings/llm/llm-smallest) | Smallest AI LLM |
| [`llm-together`](update-settings/llm/llm-together) | Together AI LLM |
| [`stt-cartesia`](update-settings/stt/stt-cartesia) | Cartesia speech-to-text |
| [`stt-deepgram`](update-settings/stt/stt-deepgram) | Deepgram speech-to-text |
| [`stt-deepgram-flux`](update-settings/stt/stt-deepgram-flux) | Deepgram Flux speech-to-text |
| [`stt-google`](update-settings/stt/stt-google) | Google Cloud speech-to-text |
| [`stt-gradium`](update-settings/stt/stt-gradium) | Gradium speech-to-text |
| [`stt-soniox`](update-settings/stt/stt-soniox) | Soniox speech-to-text |
| [`stt-speechmatics`](update-settings/stt/stt-speechmatics) | Speechmatics speech-to-text |
| [`tts-assemblyai`](update-settings/tts/tts-assemblyai) | AssemblyAI text-to-speech |
| [`tts-asyncai`](update-settings/tts/tts-asyncai) | Async text-to-speech |
| [`tts-azure`](update-settings/tts/tts-azure) | Azure OpenAI text-to-speech |
| [`tts-cartesia`](update-settings/tts/tts-cartesia) | Cartesia text-to-speech |
| [`tts-cartesia-http`](update-settings/tts/tts-cartesia-http) | Cartesia over HTTP text-to-speech |
| [`tts-deepgram`](update-settings/tts/tts-deepgram) | Deepgram text-to-speech |
| [`tts-deepgram-flux`](update-settings/tts/tts-deepgram-flux) | Deepgram Flux text-to-speech |
| [`tts-deepgram-http`](update-settings/tts/tts-deepgram-http) | Deepgram over HTTP text-to-speech |
| [`tts-elevenlabs`](update-settings/tts/tts-elevenlabs) | ElevenLabs text-to-speech |
| [`tts-google-stream`](update-settings/tts/tts-google-stream) | Google Cloud streaming text-to-speech |
| [`tts-gradium`](update-settings/tts/tts-gradium) | Gradium text-to-speech |
| [`tts-inworld`](update-settings/tts/tts-inworld) | Inworld text-to-speech |
| [`tts-kokoro`](update-settings/tts/tts-kokoro) | local Kokoro text-to-speech |
| [`tts-neuphonic`](update-settings/tts/tts-neuphonic) | Neuphonic text-to-speech |
| [`tts-neuphonic-http`](update-settings/tts/tts-neuphonic-http) | Neuphonic over HTTP text-to-speech |
| [`tts-pockettts`](update-settings/tts/tts-pockettts) | local Pocket TTS text-to-speech |
| [`tts-resembleai`](update-settings/tts/tts-resembleai) | Resemble AI text-to-speech |
| [`tts-rime`](update-settings/tts/tts-rime) | Rime text-to-speech |
| [`tts-sarvam`](update-settings/tts/tts-sarvam) | Sarvam text-to-speech |

## Multi-worker

| Example | What it shows | Keys |
| --- | --- | --- |
| [`local-handoff-two-agents`](multi-worker/local-handoff/local-handoff-two-agents) | A main worker bridges the transport to the bus, and two LLM workers (greeter and support) hand the conversation to each other with tools. | Deepgram, OpenAI, Cartesia |
| [`local-handoff-two-agents-tts`](multi-worker/local-handoff/local-handoff-two-agents-tts) | The same handoff, with each LLM worker running its own voice and the main worker playing the audio it receives over the bus. | Deepgram, OpenAI, Cartesia |
| [`parallel-debate`](multi-worker/parallel-debate/parallel-debate) | A moderator tool sends a topic to three LLM workers in parallel through a job group and summarizes their answers. | Deepgram, OpenAI, Cartesia |
| [`sensor-controller`](multi-worker/sensor-controller/sensor-controller) | A voice agent forwards requests as jobs to a second worker running a simulated temperature sensor and an LLM with tools over it. | Deepgram, OpenAI, Cartesia |
| [`openai-responses-frontend`](multi-worker/llm-with-backend/openai-responses-frontend) | A frontend on the OpenAI Responses API delegates tool work to a Claude backend worker. | Deepgram, OpenAI, Anthropic, Cartesia |
| [`openai-responses-frontend-predictive-delegation`](multi-worker/llm-with-backend/openai-responses-frontend-predictive-delegation) | The same, with a Jev classifier deciding after each request whether to start the delegation early. | Deepgram, OpenAI, Anthropic, Cartesia, TypeSafe |
| [`anthropic-frontend`](multi-worker/llm-with-backend/anthropic-frontend) | A frontend on Claude delegates tool work to a Claude backend worker. | Deepgram, Anthropic, Cartesia |
| [`google-frontend`](multi-worker/llm-with-backend/google-frontend) | A frontend on Gemini delegates tool work to a Claude backend worker. | Deepgram, Google AI, Anthropic, Cartesia |
| [`openai-realtime-frontend`](multi-worker/llm-with-backend/openai-realtime-frontend) | An OpenAI Realtime frontend delegates tool work to a Claude backend worker. | OpenAI, Anthropic |
| [`gemini-live-frontend`](multi-worker/llm-with-backend/gemini-live-frontend) | A Gemini Live frontend delegates tool work to a Claude backend worker. | Google AI, Anthropic |
| [`redis-handoff`](multi-worker/distributed-handoff/redis-handoff) | The local handoff split into a transport process and one process per LLM worker, connected by a Redis bus. A module of its own. | Deepgram, OpenAI, Cartesia |
| [`remote-proxy-assistant`](multi-worker/remote-proxy-assistant) | A transport process reaches an LLM worker on a remote server through WebSocket proxy workers. | Deepgram, OpenAI, Cartesia |

## Flows

| Example | What it shows | Keys |
| --- | --- | --- |
| [`hello_world`](flows/hello_world) | A two-node flow that asks for a favorite color, records it and ends the call. | Cartesia, Google AI |
| [`food_ordering`](flows/food_ordering) | A pizza or sushi ordering flow with a pre-action, a global function, confirmation, revision and an end-of-call post-action, on the LLM picked by `LLM_PROVIDER`. | Deepgram, Cartesia, OpenAI |
| [`food_ordering_advanced_functionschema`](flows/food_ordering_advanced_functionschema) | The same with each function schema written out by hand, with enum and range constraints. | Deepgram, Cartesia, OpenAI |
| [`multi_worker_handoff`](flows/multi_worker_handoff) | A transport worker, a free-form router worker and a reservation worker driven by a flow, handing off over the bus. | Deepgram, Cartesia, OpenAI |

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
