<div align="center">

<img src="assets/logo.png" alt="jargo" width="200" />

**A WebRTC-native, audio-first conversational-AI framework for Go.**

[![CI](https://github.com/gojargo/jargo/actions/workflows/ci.yml/badge.svg)](https://github.com/gojargo/jargo/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/gojargo/jargo.svg)](https://pkg.go.dev/github.com/gojargo/jargo)
[![Go Report Card](https://goreportcard.com/badge/github.com/gojargo/jargo)](https://goreportcard.com/report/github.com/gojargo/jargo)
![Go version](https://img.shields.io/github/go-mod/go-version/gojargo/jargo)
[![Release](https://img.shields.io/github/v/release/gojargo/jargo?sort=semver)](https://github.com/gojargo/jargo/releases)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

</div>

---

**jargo** builds real-time voice agents in Go: audio in over WebRTC, a streaming
transcription → reasoning → speech pipeline with turn-taking and barge-in, and
audio back out — over [RTVI](https://docs.pipecat.ai/client/introduction) so
existing clients interoperate.

> **Status:** early work in progress. APIs are unstable and will change.

## Why?

[Pipecat](https://github.com/pipecat-ai/pipecat) is great, and jargo is a port of
it — the architecture and many design decisions are Pipecat's.

### Python might not be the way

This port exists for one reason: I'd rather not run a voice agent on Python.

Python is the right tool when you need the AI/data-science ecosystem. A
real-time voice *server* doesn't: the models run as services or as ONNX, and
what's left is plumbing — audio framing, WebRTC, concurrency, and shipping a
binary. For that, Go is a better fit: one static binary to deploy, low and
predictable memory, fast startup, and real concurrency for many simultaneous
sessions without a GIL. The heavy numerics stay where they belong (the ONNX
Runtime, the remote services), so giving up Python costs little here. See the
[benchmarks](https://github.com/gojargo/jargo-benchmarks) for the honest performance picture.

### No Daily, no lock-in

jargo stays on plain, standard WebRTC via [Pion](https://github.com/pion) — no
Daily, no hosted transport, no proprietary SDK or cloud to sign up for. You ship
one binary, the browser connects with vanilla WebRTC, and RTVI rides the data
channel. Keeping the transport open and self-hosted is a deliberate goal, not an
afterthought.

## Features

- **WebRTC**, pure Go ([Pion](https://github.com/pion)) — audio in and out of the browser.
- **Opus**, not pure Go yet, waiting for *pion/opus* to be ready.
- **Streaming voice pipeline**: STT → LLM → TTS, with prompt caching.
- **Speech-to-speech**: single-model voice agents (OpenAI Realtime, Gemini Live, AWS Nova Sonic).
- **Turn-taking & barge-in**: Silero VAD + Smart Turn v3, local ONNX.
- **Telephony** (optional): inbound/outbound phone calls over Twilio Media Streams.
- **User-idle watchdog**: re-engage or hang up when the caller goes silent.
- **RTVI** data channel — works with existing RTVI clients.
- **Pluggable services**: swap any STT/LLM/TTS behind a small interface.
- **Concurrent by design**: independent processors; interruptions are frames.

## Providers

Pick any per category; each is a small `Config` + constructor.

- **STT**: Deepgram, AssemblyAI, Gladia, Speechmatics, Soniox, Whisper (OpenAI/Groq/local), Azure.
- **LLM**: Anthropic (direct + Bedrock), OpenAI, Google Gemini, Groq, Together, Fireworks, DeepSeek,
  Cerebras, Perplexity, OpenRouter, xAI, Ollama, NVIDIA, Mistral, Nebius, SambaNova, Qwen, Azure OpenAI.
- **TTS**: ElevenLabs, Cartesia, Rime, LMNT, Kokoro, Piper, Deepgram, OpenAI, Azure, Hume, Fish, MiniMax.
- **Speech-to-speech**: OpenAI Realtime, Gemini Live, AWS Nova Sonic.
- **Memory**: mem0.

## Dependencies

jargo uses cgo (`CGO_ENABLED=0` is not supported) and a few native libraries:

- **libsoxr** — audio resampling, linked at build time (`libsoxr-dev`).
- **libopus** — optional C Opus encoder, selected with `-tags libopus`
  (`libopus-dev`); the default build ships a pure-Go encoder, but libopus
  sounds noticeably better on speech.
- **ONNX Runtime** — loaded at run time for VAD + end-of-turn detection.

The container image bundles all of them.

## Usage

```sh
go get github.com/gojargo/jargo
```

**Locally** — install the native deps, then build with cgo:

```sh
# Debian/Ubuntu: apt-get install -y libsoxr-dev libopus-dev
CGO_ENABLED=1 go run ./examples/echo                    # open http://localhost:8080
CGO_ENABLED=1 go run -tags libopus ./examples/voicebot  # libopus speech encoder
```

**With Docker** — the image bundles every native dependency, so there's no host
setup:

```sh
docker build -t jargo-voicebot .
docker run --rm -p 8080:8080 \
  -e DEEPGRAM_API_KEY=… -e ANTHROPIC_API_KEY=… -e ELEVENLABS_API_KEY=… \
  jargo-voicebot
```

See the **[Quickstart](docs/quickstart.md)** for the full setup.

## Examples

Runnable bots live in [`examples/`](examples):

- **echo** — hear yourself back, no API keys.
- **voicebot** — the full voice agent (STT → LLM → TTS over WebRTC) with
  turn-taking, long-term memory, and tracing.
- **voice/** — one bot per provider, each wiring its STT/LLM/TTS explicitly;
  run with `go run ./examples/voice/<provider>` (e.g. `deepgram`, `cartesia`,
  `openai`).
- **twiliobot** — a phone agent over Twilio Media Streams, with the idle watchdog.

The fastest way to try them — locally or with Docker — is the
**[Quickstart](docs/quickstart.md)**.

```sh
go run ./examples/echo                 # then open http://localhost:8080
```

## Documentation

See **[docs/index.md](docs/index.md)** for the full documentation.

## License & attribution

jargo is a Go port of [Pipecat](https://github.com/pipecat-ai/pipecat),
distributed under the same **BSD 2-Clause License**. The upstream copyright —
*Copyright (c) 2024–2026, Daily* — is preserved verbatim in [`LICENSE`](LICENSE);
see [`NOTICE`](NOTICE) for details. jargo is an independent project, not
affiliated with or endorsed by Daily.
