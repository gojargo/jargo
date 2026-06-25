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

### Python is not the way

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

- **WebRTC + Opus**, pure Go ([Pion](https://github.com/pion)) — audio in and out of the browser.
- **Streaming voice pipeline**: STT → LLM → TTS, with prompt caching.
- **Turn-taking & barge-in**: Silero VAD + Smart Turn v3, local ONNX.
- **RTVI** data channel — works with existing RTVI clients.
- **Pluggable services**: swap any STT/LLM/TTS behind a small interface.
- **Concurrent by design**: independent processors; interruptions are frames.

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
