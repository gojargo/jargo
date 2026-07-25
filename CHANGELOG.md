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

- **Word-aligned TTS text and interruption-accurate context.** A new
  `frames.TTSTextFrame` carries each spoken word aligned to audio playback, with
  the original written form of transformed spans (e.g. `"$42.50"` for a token
  spoken as "forty two dollars and fifty cents") in its `RawText` field. A TTS
  provider opts in by implementing the optional `tts.WordTimestamps` interface
  (`SynthesizeTimed`); the shared TTS base then diffs the text sent to the
  synthesizer against the original, tracks word completion, and emits a
  `TTSTextFrame` per spoken word as its audio is produced. The output transport
  releases these frames in step with playback, so an interruption drops the ones
  whose audio never played. The assistant context aggregator records the spoken
  words in their original written form and truncates exactly at the interruption
  point, instead of recording the full generated response. Cartesia gains a
  `WordTimestamps` config flag that enables this path; every provider without
  word timings — and Cartesia with the flag off — behaves exactly as before. The
  text-alignment machinery lives in the new `utils/context` package
  (`TextSegmentMap`, `WordCompletionTracker`, `MergePunctTokens`).

- **Audio token usage for realtime models.** `frames.LLMTokenUsage` gains an
  additive per-modality breakdown — `InputAudioTokens`, `OutputAudioTokens`,
  `InputTextTokens`, `OutputTextTokens` — each a subset of the prompt/completion
  totals, for speech-to-speech models that bill audio and text at different
  rates. The Gemini Live service now parses `usageMetadata` (folding the
  prompt/response modality detail into the breakdown) and the OpenAI Realtime
  service parses the `response.done` usage (input/output token details incl.
  audio); both report through a shared `processor.Base.PushTokenUsage`, so
  realtime token usage reaches the in-band `MetricsFrame`, the aggregate metrics,
  and telemetry spans. Token attributes are written under the OpenTelemetry
  GenAI `gen_ai.usage.*` keys (with the audio/text/cache breakdowns added when
  nonzero) alongside the legacy `llm.tokens.*` keys, unified for the cascaded
  and realtime paths. Nova Sonic's bidirectional stream carries no usage event,
  so it reports none.

- **Priceable STT and TTS spans.** Speech spans now carry what a cost-tracking
  backend needs to price them, so a trace shows the cost of the whole turn
  rather than of its LLM call alone. Each synthesis records the provider model
  and its character count; each transcription records the model and the duration
  of audio sent. A provider reports its model through the new optional
  `tts.Describer` (returning a `tts.Metadata`) or through `Model` on the existing
  `stt.Metadata` — ElevenLabs, Deepgram (including Flux), Cartesia, NVIDIA and
  Gradium do; a provider that reports none is unchanged apart from the
  operation name. Streaming STT reports usage once per session, covering all the
  audio the connection carried (silence included, which is what streaming
  providers bill for); batch STT reports per transcribed segment. Usage is
  written as OpenTelemetry GenAI `gen_ai.*` attributes plus
  `langfuse.observation.usage_details`, since the GenAI conventions model usage
  only as token counts and speech is billed per character or per second.
  Characters are counted in runes, so accented text is no longer counted — or
  billed — twice. The model now labels the TTS metrics too, and a new
  `jargo.stt.audio` counter records transcribed audio.

- **Behavioral eval harness** (`eval`) and a **`jargo` CLI** (`cmd/jargo`). The
  harness drives a real bot over RTVI, plays scripted conversation turns from a
  YAML scenario, and asserts on the semantic events the bot emits. Scenarios run
  two ways off one core: in-process from a Go test with `eval.Run(t, path,
  buildBot)` (the bot hosted on a loopback WebSocket via `eval.Handler`), or from
  the command line against a running bot with `jargo eval run <scenario.yaml>
  --bot-url ws://…`. This first iteration covers text-mode scenarios — each user
  turn is delivered as RTVI `send-text`, and expectations match `llm_started`,
  `llm_response` (`text_contains` or an LLM `judge`), and `function_call`
  events in order, each within a `within_ms` latency budget. A `judge:` criterion
  is graded by an LLM judge (`eval.NewLLMJudge`, backed by any jargo LLM service;
  `jargo eval run --judge-model …` for the CLI), with verdicts cached per
  (criterion, reply). **Audio mode** (`Options.UserTTS`) synthesizes each user
  turn and streams it to the bot as microphone audio at real-time cadence, so the
  bot's real VAD, turn detection and STT run — unlocking `user_started_speaking`,
  `user_stopped_speaking` and `user_transcription` assertions. **`jargo eval
  suite <manifest.yaml>`** runs many scenarios across bots concurrently and prints
  an aggregate summary.
- **RTVI text input and richer event surface** (`processor/rtvi`): the processor
  now handles an inbound `send-text` message (appending a user turn and running
  the LLM), and emits `bot-llm-started`/`bot-llm-stopped`,
  `bot-tts-started`/`bot-tts-stopped`, `llm-function-call-in-progress`, and
  `llm-function-call-result` server messages — so RTVI clients see the full
  response lifecycle and tool calls.
- **RTVI-over-WebSocket serializer** (`transport/wsserver/rtviws`): a
  `wsserver.Serializer` that carries the RTVI control, event, and text channel —
  plus inbound microphone audio (`raw-audio` → `InputAudioRawFrame`) — over a
  plain WebSocket, so a client can drive a bot without WebRTC.
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

### Changed

- `metrics.RecordTTSCharacters` takes the provider model, so TTS measurements
  carry the same `model` label the other instruments do.

### Fixed

- **Trailing audio dropped at turn ends** (`transport`): the base output
  transport buffers audio into fixed-size chunks and used to leave the final
  sub-chunk of a turn sitting in the buffer, where a barge-in would clear it —
  clipping the last few milliseconds of the bot's utterance. The remainder is now
  padded to a full chunk with silence and flushed when the turn ends, so it plays
  out. Padding lets the tail pass through downstream whole-frame encoders (Opus)
  that would otherwise strand a short frame. A barge-in still discards the pending
  tail, as it must.

- **Anthropic/Bedrock assistant-prefill constraint** (`provider/anthropic`): the
  Claude 4.6-generation models (Opus 4.8, Sonnet 4.6, …) reject a request whose
  message list ends with an assistant message. When the configured model does not
  support prefill, the service now appends a minimal `.` user message to such
  requests at send time; the stored `LLMContext` is left untouched. Prefill
  support is detected from a frozen allow-list of the models known to still
  accept it, so every current and future no-prefill model is covered by default.
  Bedrock is fixed by the same change, since it runs on the Anthropic service.

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
