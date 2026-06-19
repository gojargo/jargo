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

jargo builds real-time voice agents: audio comes in over WebRTC, flows through a
pipeline of processors (transcription → reasoning → speech), and audio goes back
out — with natural turn-taking, barge-in, and [RTVI](https://docs.pipecat.ai/client/introduction)
on the data channel so existing RTVI clients interoperate.

> **Status:** early work in progress. APIs are unstable and will change.

## Why?

[Pipecat](https://github.com/pipecat-ai/pipecat) is great, and jargo is a port of
it — the architecture and many design decisions are Pipecat's. This port exists
for one reason: I'd rather not run a voice agent on Python.

Python is the right tool when you need the AI/data-science ecosystem. A
real-time voice *server* doesn't: the models run as services or as ONNX, and
what's left is plumbing — audio framing, WebRTC, concurrency, and shipping a
binary. For that, Go is a better fit: one static binary to deploy, low and
predictable memory, fast startup, and real concurrency for many simultaneous
sessions without a GIL. The heavy numerics stay where they belong (the ONNX
Runtime, the remote services), so giving up Python costs little here. See the
[benchmarks](docs/benchmarks.md) for the honest performance picture.

## Features

- **WebRTC + Opus**, pure Go ([Pion](https://github.com/pion)) — audio in and out of the browser.
- **Streaming voice pipeline**: STT → LLM → TTS, with prompt caching.
- **Turn-taking & barge-in**: Silero VAD + Smart Turn v3, local ONNX.
- **RTVI** data channel — works with existing RTVI clients.
- **Pluggable services**: swap any STT/LLM/TTS behind a small interface.
- **Concurrent by design**: independent processors; interruptions are frames.

## Services

Each category shares a base (`service/llm`, `service/stt`, `service/tts`), so a
provider implements only what differs and gets the frame contract, streaming,
and sentence aggregation for free.

| Category | Providers |
| -------- | --------- |
| **LLM** | Anthropic, OpenAI, Google Gemini, plus the OpenAI-compatible family — Groq, Together, Fireworks, DeepSeek, Cerebras, xAI (Grok), OpenRouter, Perplexity, NVIDIA NIM, Ollama |
| **STT** | Deepgram, AssemblyAI, Gladia (streaming); OpenAI, Groq (segmented Whisper) |
| **TTS** | ElevenLabs, Cartesia, OpenAI, Deepgram Aura, Rime, LMNT |

Any OpenAI-compatible endpoint works directly via
`openai.NewCompatLLM(name, baseURL, envVar, model, cfg)`. Each service reads its
API key from a provider-specific env var (e.g. `OPENAI_API_KEY`,
`CARTESIA_API_KEY`) when the config field is empty.

The voicebot example selects providers by env var:

```sh
STT=assemblyai LLM=openai TTS=cartesia go run ./examples/voicebot
```

## Install

```sh
go get github.com/gojargo/jargo
```

jargo uses cgo and two native libraries — **libsoxr** (linked) and the **ONNX
Runtime** (loaded at run time). Install `libsoxr-dev` or just run the container
image, which bundles both; the [Quickstart](docs/quickstart.md) covers setup.
`CGO_ENABLED=0` builds are not supported.

## Examples

Two runnable bots live in [`examples/`](examples): an **echo** bot (no API keys)
and a full **voice** bot (STT → LLM → TTS). The fastest way to try either —
locally or with Docker — is the **[Quickstart](docs/quickstart.md)**.

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
