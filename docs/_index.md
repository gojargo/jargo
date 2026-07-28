---
title: Documentation
weight: 1
---

# jargo documentation

**jargo** is a WebRTC-native, audio-first conversational-AI framework for Go:
audio in over WebRTC, a streaming transcription → reasoning → speech pipeline with
turn-taking and barge-in, and audio back out.

> **Early work in progress.** The public API is unstable and changes in any
> release. Pin an exact version and read the
> [changelog](https://github.com/gojargo/jargo/blob/main/CHANGELOG.md) before
> upgrading.

## Three ways in

**I want it running.** → [Installation](getting-started/installation.md) →
[Quickstart](getting-started/quickstart.md) →
[Your first bot](getting-started/your-first-bot.md)

**I want to understand it.** → [Architecture](concepts/architecture.md) →
[Frames](concepts/frames.md) → [Processors](concepts/processors.md) →
[Pipeline & Task](concepts/pipeline.md)

**I want to extend it.** → [Writing a processor](extending/custom-processor.md) →
[Writing a service](extending/custom-service.md)

## Concepts

The engine is small: `frames`, `processor` and `pipeline` are about 2,000 lines
between them. Understanding it makes everything else obvious.

| Page | What it covers |
|---|---|
| [Architecture](concepts/architecture.md) | The whole system on one page, with a full turn traced end to end. |
| [Frames](concepts/frames.md) | The three categories, why they exist, and the complete catalog. |
| [Processors](concepts/processors.md) | The two goroutines inside every processor, and why. |
| [Pipeline & Task](concepts/pipeline.md) | Building the chain and driving it. |
| [Interruptions](concepts/interruptions.md) | Barge-in: the mechanism, not the metaphor. |
| [LLM context](concepts/llm-context.md) | How the conversation accumulates, and when the LLM runs. |

## Guides

| Page | What it covers |
|---|---|
| [Turn-taking](guides/turn-taking.md) | Silero VAD + Smart Turn v3, and the strategies that tune them. |
| [Services](guides/services.md) | Swapping STT/LLM/TTS providers; tool calling; speech-to-speech. |
| [Audio](guides/audio.md) | Codecs, sample rates, denoising, mixing, recording. |
| [Telephony](guides/telephony.md) | Phone calls over Twilio, Telnyx, Plivo, Exotel; DTMF; the idle watchdog. |
| [RTVI](guides/rtvi.md) | The client event protocol over the data channel. |
| [Observability](guides/observability.md) | Observers, metrics, tracing, and which numbers matter. |

## Extending

| Page | What it covers |
|---|---|
| [Writing a processor](extending/custom-processor.md) | Add your own logic to the chain. |
| [Writing a service](extending/custom-service.md) | Add an STT, LLM or TTS provider. |

## Deploying

| Page | What it covers |
|---|---|
| [Deploy with Docker](deploy/docker.md) | The build and distroless runtime base images, and a Dockerfile. |

## Elsewhere

- **[Go reference](https://pkg.go.dev/github.com/gojargo/jargo)**: the API. These
  pages deliberately do not duplicate it.
- **[Examples](https://github.com/gojargo/jargo/tree/main/examples)**: runnable
  bots, one per provider.
- **[Benchmarks](https://github.com/gojargo/jargo-benchmarks)**: the honest
  performance picture.
- **[Changelog](https://github.com/gojargo/jargo/blob/main/CHANGELOG.md)**: what
  changed, and what broke.
