# Changelog

All notable changes to jargo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Development status.** While jargo is in early development the version stays
> in the `0.0.x` range: the public API is unstable and may change in any
> release, with no backwards-compatibility guarantees. `0.1.0` will mark the
> first release intended for wider use.

## [Unreleased]

### Added

- **Behavioral eval harness** (`eval`) and a **`jargo` CLI** (`cmd/jargo`). The
  harness drives a real bot over RTVI, plays scripted conversation turns from a
  YAML scenario, and asserts on the semantic events the bot emits. Scenarios run
  two ways off one core: in-process from a Go test with `eval.Run(t, path,
  buildBot)` (the bot hosted on a loopback WebSocket via `eval.Handler`), or from
  the command line against a running bot with `jargo eval run <scenario.yaml>
  --bot-url ws://…`. This first iteration covers text-mode scenarios — each user
  turn is delivered as RTVI `send-text`, and expectations match `llm_started`,
  `llm_response` (`text_contains` or an optional LLM `judge`), and `function_call`
  events in order, each within a `within_ms` latency budget.
- **RTVI text input and richer event surface** (`processor/rtvi`): the processor
  now handles an inbound `send-text` message (appending a user turn and running
  the LLM), and emits `bot-llm-started`/`bot-llm-stopped`,
  `bot-tts-started`/`bot-tts-stopped`, `llm-function-call-in-progress`, and
  `llm-function-call-result` server messages — so RTVI clients see the full
  response lifecycle and tool calls.
- **RTVI-over-WebSocket serializer** (`transport/wsserver/rtviws`): a
  `wsserver.Serializer` that carries the RTVI control, event, and text channel
  over a plain WebSocket, so a client can drive a bot without WebRTC.
- **NVIDIA Riva streaming STT** (`provider/nvidia`): `NewSTT` adds a gRPC
  streaming speech-to-text service that talks to NVIDIA's hosted ASR endpoint
  or a locally deployed Riva/NIM model (such as parakeet), selected through
  `Server`/`UseSSL` and the `APIKey`/`FunctionID` auth fields. Emits interim and
  finalized `TranscriptionFrame`s, with Riva's server-side endpointing driving
  the end-of-turn signal, and exposes optional endpointing tuning. The generated
  Riva gRPC client lives in `provider/nvidia/internal/rivapb`.
- **Telephony serializers for Plivo, Telnyx, and Exotel.** Three new
  `wsserver.Serializer` implementations alongside the existing Twilio one:
  Plivo (`transport/wsserver/plivo`, μ-law with REST auto-hang-up), Telnyx
  (`transport/wsserver/telnyx`, μ-law **and** A-law, receive encoding learned
  from the stream, REST auto-hang-up), and Exotel (`transport/wsserver/exotel`,
  raw 16-bit PCM). Each learns its stream/call identifiers from the inbound
  `start` message and emits a barge-in clear on interruption.
- **G.711 A-law codec** (`audio/g711`): `EncodeALaw`/`DecodeALaw` join the
  existing μ-law pair, for PSTN audio outside North America (Telnyx PCMA).
- **TTS text normalization** (`utils/text/`): a pluggable `text.Filter` layer that
  rewrites written text into a more speakable form before synthesis — strips
  Markdown, spells out numbers, currency, percentages, dates, and units, spaces
  out phone-number digits and acronyms, and reads email addresses aloud. A
  `text.VoiceFormatter` bundles a configurable, ordered pipeline of these
  transforms; attach it with the new `(*tts.Base).SetTextFilters`, promoted to
  every TTS service. English number spelling follows the `num2words` "en"
  conventions and needs no external dependency.
- **Local-audio transport** (`transport/localaudio`): capture from the local
  microphone and play back through the local speaker, for running a bot on the
  same machine with no browser or telephony leg. It speaks the PulseAudio native
  protocol through the pure-Go [`github.com/jfreymuth/pulse`](https://github.com/jfreymuth/pulse)
  client, so the default build stays cgo-free; it needs a running PulseAudio or
  PipeWire (pulse-compatible) server. See `examples/localaudio` for a no-keys
  microphone-to-speaker echo.

## [0.0.4] - 2026-07-12

### Added

- **Service self-description and time-to-first-audio.** STT, TTS, and LLM
  services broadcast a metadata frame at pipeline start — an STT service can
  recommend external turn strategies — and the TTS base reports time-to-first-audio
  (the first *audible* sample, via a pure-Go speech-onset detector) alongside the
  existing time-to-first-byte.
- **Pipeline observers** (`observers/`): turn tracking, user-to-bot latency,
  startup timing, and a frame logger, wired into the task's frame-observation hook.
- **Audio recording** (`processor/audiobuffer`): a track-synced recorder with an
  auto-start option plus start/stop-recording control frames and events.
- **Audio mixer, DTMF, and composable noise filters.** An output mixer
  (`audio/mixer`) driven by a `MixerControlFrame`; DTMF tone synthesis and an
  aggregator (`processor/dtmf`), with the Twilio serializer now emitting DTMF; a
  filter-chaining `audio.Chain` and a pure-Go noise gate (`audio/noise/gate`).
- **Framework and extension processors.** A LangChain-style chain bridge
  (`processor/langchain`), an IVR navigator (`processor/ivr`), and a voicemail
  detector (`processor/voicemail`).
- **New STT/LLM/TTS provider modalities** — Camb, Gradium, Inworld, Neuphonic,
  Sarvam, AWS Polly, and AWS Transcribe, plus modality fills for Google,
  Cartesia, ElevenLabs, Groq, Together, Mistral, and Speechmatics.
- **Supply-chain security.** Snyk and TruffleHog jobs in the security workflow,
  and keyless cosign signing of release checksums.

### Changed

- **Package layout.** The `AudioFilter`/`AudioMixer` interfaces moved to a
  neutral `audio` package (`audio.Filter`/`audio.Mixer`); the RNNoise denoiser
  and the noise gate now live under `audio/noise/{rnnoise,gate}`; every concrete
  processor moved under `processor/*` (aggregators, rtvi, turns, vadproc, dtmf,
  audiobuffer, ivr, voicemail, langchain); and the metrics and tracing exporters
  were grouped under `telemetry/*`. Import paths change; the exported APIs are
  otherwise unchanged.

## [0.0.3] - 2026-07-12

### Added

- **Pure-Go ONNX Runtime backend.** `internal/onnxrt` now has two interchangeable
  backends behind one API: the default cgo binding (`yalue/onnxruntime_go`) and a
  cgo-free binding built on [`ebitengine/purego`](https://github.com/ebitengine/purego)
  that calls the ONNX Runtime C API directly, selected by a `CGO_ENABLED=0` build.
  Both load the runtime shared library at run time; VAD and end-of-turn produce
  bit-for-bit the same results either way. `onnxrt.NewWithOptions` adds an
  `IntraOpThreads` cap (useful when many per-stream sessions would otherwise each
  spawn a core-sized thread pool), and `onnxrt.Backend()` reports the active binding.

- **Outbound telephony example** — `examples/twilio/outbound` places an outbound
  Twilio call (via the REST API, no Twilio SDK dependency), runs an STT → LLM →
  TTS pipeline over the media stream, collects a few details through a
  `record_info` tool call, then says goodbye and hangs up. Configured by a YAML
  file (`-config`, see `config.example.yaml`) with environment-variable overrides.

### Changed

- **Pure-Go SILK Opus encoder by default.** The default Opus encoder is now a
  pure-Go SILK encoder (natural speech), replacing the CELT-only default. The C
  libopus encoder stays available behind `-tags libopus`.
- **Resampling is now pure Go by default.** `audio/resample` uses the no-cgo
  [`github.com/gojargo/go-resample`](https://github.com/gojargo/go-resample)
  converter, so the default build links no native resampler and needs no
  `libsoxr-dev`. Build with `-tags libsoxr` to link libsoxr (the SoX Resampler)
  for its highest-quality polyphase conversion instead. The `New`/`Process`/
  `Close` API is unchanged, so callers need no updates.
- **Noise reduction (RNNoise) no longer needs cgo or a build tag.** `audio/rnnoise`
  binds librnnoise through [`ebitengine/purego`](https://github.com/ebitengine/purego)
  and loads it at run time, so it builds in every configuration and is selected at
  runtime by setting it as the transport's `AudioInFilter` — like any STT/LLM
  service. `New` returns `ErrNotAvailable` when librnnoise is absent (point at a
  non-standard install with `JARGO_RNNOISE_LIB`); the `-tags rnnoise` build tag and
  the passthrough stub are gone. Together with the ONNX backend, the default build
  is now fully cgo-free — `CGO_ENABLED=0 go build ./...` works.
- Bumped `github.com/pion/opus` to upstream `main`, pulling in the latest CELT
  encoder quality work (pitch pre-filter, post-filter, dynalloc), which the
  pure-Go encoder builds on.
- The per-provider `examples/voice/<provider>` bots are now self-contained,
  headless **backends**: the shared `run` helper was removed and each example
  inlines the full pipeline and serves only the WebRTC `/offer` endpoint (with
  CORS), so a single file can be copied as a starting point. jargo is a backend
  framework; the browser client is the `nextjs-voicebot` example in
  [jargo-client-react](https://github.com/gojargo/jargo-client-react).

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

[Unreleased]: https://github.com/gojargo/jargo/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/gojargo/jargo/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/gojargo/jargo/compare/v0.0.2...v0.0.3
[0.0.1]: https://github.com/gojargo/jargo/releases/tag/v0.0.1
