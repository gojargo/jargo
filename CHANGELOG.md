# Changelog

All notable changes to jargo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Development status.** While jargo is in early development the version stays
> in the `0.0.x` range: the public API is unstable and may change in any
> release, with no backwards-compatibility guarantees. `0.1.0` will mark the
> first release intended for wider use.

## [Unreleased]

## [0.0.1] - 2026-06-29

First tagged development release: a WebRTC-native, audio-first conversational-AI
framework for Go, ported from [Pipecat](https://github.com/pipecat-ai/pipecat).

### Added

- **Pipeline & engine** — frame-based streaming engine with independent,
  concurrent processors; interruptions modelled as frames. `ParallelPipeline`
  and a `ServiceSwitcher` (manual and failover strategies).
- **WebRTC transport** — pure-Go [Pion](https://github.com/pion); audio in and
  out of the browser, with RTVI riding the data channel for interoperability
  with existing RTVI clients.
- **Voice pipeline** — streaming STT → LLM → TTS with prompt caching, plus
  LLM function/tool calling.
- **Turn-taking & barge-in** — Silero VAD + Smart Turn v3 end-of-turn detection
  via local ONNX, with a user-idle watchdog.
- **Speech-to-speech** — single-model voice agents: OpenAI Realtime, Gemini
  Live, AWS Nova Sonic.
- **Telephony** — inbound/outbound phone calls over Twilio Media Streams.
- **Long-term memory** — mem0 integration.
- **Context management** — automatic LLM context summarization.
- **Observability** — OpenTelemetry tracing and in-band pipeline metrics.
- **Providers** — pluggable services behind small interfaces:
  - STT: Deepgram, AssemblyAI, Gladia, Speechmatics, Soniox, Whisper
    (OpenAI/Groq/local whisper.cpp), Azure.
  - LLM: Anthropic (direct + Bedrock), OpenAI, Google Gemini, Groq, Together,
    Fireworks, DeepSeek, Cerebras, Perplexity, OpenRouter, xAI, Ollama, NVIDIA,
    Mistral, Nebius, SambaNova, Qwen, Azure OpenAI.
  - TTS: ElevenLabs, Cartesia, Rime, LMNT, Kokoro, Piper, Deepgram, OpenAI,
    Azure, Hume, Fish, MiniMax.
- **Audio** — libsoxr resampling and an optional libopus encoder
  (`-tags libopus`); G.711 for telephony.
- **Examples** — runnable `echo`, `voicebot` (full stack: turn-taking, mem0,
  tracing) and `twiliobot` bots, plus `examples/voice/<provider>` — one small
  bot per provider, each wiring its STT/LLM/TTS explicitly in Go.

[Unreleased]: https://github.com/gojargo/jargo/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/gojargo/jargo/releases/tag/v0.0.1
