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

- **Four streaming speech-to-text variants**, each sitting alongside a service
  the provider already had:

  - `provider/openai/realtime.NewSTT` runs the Realtime API in
    transcription-only mode, selected with an `intent=transcription` handshake.
    It exchanges 24 kHz audio, so run the transport's input at that rate.
  - `provider/elevenlabs.NewRealtimeSTT` holds a connection open and lets
    ElevenLabs commit each utterance on its own silence detection, unlike
    `NewSTT` which transcribes one delimited segment per request.
  - `provider/cartesia.NewTurnsSTT` uses Cartesia's turn-detection endpoint,
    where the server reports turn boundaries. An eagerly predicted end of turn
    can be retracted by the user carrying on, so only a real turn end finalizes
    the transcript.
  - `provider/nvidia.NewSegmentedSTT` transcribes a whole utterance in one Riva
    batch call, for the offline models. It needs a turn detector upstream and
    produces no interim transcripts.

  The three streaming services detect utterance boundaries server-side, so they
  report `frames.UserTurnExternal` and the pipeline needs no end-of-turn
  detection of its own. The batch Riva call needed a `Recognize` RPC added to
  the generated client.

- **Azure OpenAI Realtime speech-to-speech** (`provider/azure/realtime`), the
  Realtime service as served by an Azure OpenAI resource. It wraps
  `provider/openai/realtime` and changes only the addressing and authorization:
  Azure selects the model with a `deployment` query parameter rather than a model
  name, and authorizes with an `api-key` header rather than a bearer token. The
  session URL is derived from the resource endpoint, deployment and API version,
  and an https endpoint pasted from the portal is rewritten to `wss`. A whole URL
  can be given instead for a deployment whose address is not derivable.

- **`realtime.NewWithConnector`**, the hook Azure hangs off. A
  `realtime.Connector` chooses how the Realtime session is addressed and
  authorized, defaulting to OpenAI's own model query parameter and bearer token,
  so the existing service is unchanged.

- **OpenAI Responses API** (`provider/openai/responses`), in two forms.
  `NewHTTPLLM` streams over a request per turn, the same shape as the
  chat-completions service. `NewLLM` holds a WebSocket open for the session and
  adds the API's incremental-context optimization: when the conversation has
  grown from where the previous turn left off, only the new items travel and the
  server recalls the rest by response id. The conversation prefix is fingerprinted
  so a history rewritten behind the service's back falls back to sending
  everything rather than continuing from a copy that no longer matches. Both
  support tool calling, take the system prompt as the API's own instructions
  field rather than a leading message, and default to `store: false` so nothing
  is retained server-side. The two connection-scoped failures (the server having
  dropped the response being continued from, and the connection reaching its
  lifetime limit) reconnect and retry with the full context; an interruption
  does not. A barge-in cancels the response server-side and drains its remaining
  events before the next turn starts, since the API gives no way to tell which
  response a delta belongs to and stale events would otherwise be read as the
  next turn's output.

- **Google Vertex AI** (`provider/google/vertex`), Gemini as served by Vertex
  rather than the Gemini API: `NewLLM` for the cascaded model and `NewS2S` for
  Live speech-to-speech. Both reuse the corresponding Gemini service and change
  only how requests are addressed and authorized, so their behavior, config
  fields and frame contract match. Vertex addresses a model by project and
  location on a regional endpoint and authorizes with a short-lived OAuth token.
  Credentials are explicit: either a service-account key JSON, which is exchanged
  for a token through a signed assertion, or a token source the application built
  itself. jargo reads no environment variables, so Application Default
  Credentials stay opt-in by passing their token source.

- **`gemini.NewShapedLLM` and `live.NewWithConnector`**, the hooks Vertex hangs
  off. A `gemini.RequestShaper` chooses how a generateContent request is
  addressed and authorized; a `live.Connector` does the same for the Live
  WebSocket, and also names the model's resource path in the setup message. Both
  default to the Gemini API's own URL layout and api-key auth, so the existing
  services are unchanged.

- **Azure AI Speech speech-to-text** (`provider/azure/speech.NewSTT`), streaming
  continuous recognition. It speaks Azure's recognition WebSocket protocol
  directly rather than binding the native Speech SDK, so the cgo-free build is
  unaffected: messages carry Azure's header framing, each turn opens with a RIFF
  header describing the PCM that follows, and the recognition parameters travel
  as query parameters. Azure ends a turn after every utterance, so the service
  opens the next one itself, which is what makes recognition continuous rather
  than one-shot. Hypotheses surface as interim transcriptions and a successful
  phrase as a finalized one. It supports a recognition locale, the profanity
  policy (non-English deployments often want `raw`), a custom speech model by
  deployment id, and a private endpoint or sovereign-cloud host. Note this is
  Azure AI Speech, distinct from `provider/azure/openai.NewSTT`, which is Azure
  OpenAI's Whisper.

- **Soniox text-to-speech** (`provider/soniox.NewTTS`), streaming synthesis over
  Soniox's real-time WebSocket API. Each sentence opens a synthesis stream and
  emits audio as it is generated. Word timings are on by default: Soniox reports
  a start offset per character, which is assembled into words to drive the
  word-aligned text path. Chinese and Japanese are written without spaces
  between words, so their characters are reported individually rather than
  assembled. It accepts a stock voice name or a cloned voice's id, a language, a
  speaking rate, and any of Soniox's supported sample rates.
  `provider/soniox` previously offered speech-to-text only.

- **`utils/context.CharAccumulator`**, which assembles per-character timings
  into whole words, carrying a word split across two batches into the next one.
  Several synthesizers report a start offset for every character rather than per
  word; `CharsAsWords` covers the languages written without spaces, where there
  is no boundary to split on. The xAI and Soniox TTS services share it.

- **NVIDIA Riva text-to-speech** (`provider/nvidia.NewTTS`), streaming synthesis
  over Riva's gRPC API. Like the Riva STT service it already sits next to, one
  provider serves both NVIDIA's hosted endpoint and a locally deployed Riva/NIM,
  selected through `Server`, `UseSSL` and the auth fields. Each sentence opens a
  `SynthesizeOnline` stream and emits the audio as the server generates it; a
  sentence past the model's request-length limit is split across several requests
  on that one stream, so it is still generated as a single utterance. It supports
  a custom IPA pronunciation dictionary, model-specific options, and the audio
  prompt a zero-shot model clones its voice from. `provider/nvidia` previously
  offered the LLM and STT only.

- **`tts.Closer`**, an optional interface a `tts.Synthesizer` implements when it
  holds a resource open across syntheses. `tts.Base` closes it during cleanup,
  so a provider can reuse one connection for every sentence (rather than paying
  for a fresh handshake each time) without leaking it when the pipeline ends. A
  Synthesizer that does not implement it is unaffected.

- **xAI Realtime speech-to-speech** (`provider/xai/realtime`), a single
  bidirectional WebSocket carrying the whole conversation in place of the
  cascaded STT → LLM → TTS stack. It sits alongside the OpenAI Realtime, Gemini
  Live and Nova Sonic services and behaves the same way downstream: server VAD
  drives turn-taking and emits an `InterruptionFrame` on barge-in, the model's
  spoken reply arrives as audio frames and its transcript as `LLMTextFrame`s,
  and token usage is reported on each completed response. Where xAI differs
  from the OpenAI protocol it follows xAI: the model is chosen on the handshake
  rather than in the session, the session is configured only once the server
  opens the conversation, the audio format is nested per direction, and the
  session runs at a configurable sample rate (24 kHz by default) rather than a
  fixed one. xAI's built-in web, X and file search tools are advertised from the
  config alongside the pipeline's own function tools. As with the other
  speech-to-speech services, executing a function call is not yet wired up.

- **xAI text-to-speech**, in two forms. `provider/xai/grok.NewTTS` streams synthesis
  over a WebSocket, opening a session per sentence and emitting audio chunks as
  they arrive; `provider/xai/grok.NewHTTPTTS` synthesizes each sentence in one request
  against the batch endpoint. The streaming service reports word timings by
  default: xAI sends per-character timings, which are assembled into words
  (carrying a word split across payloads over to the next one) and drive the
  word-aligned text path, so the assistant context records only what was
  actually spoken before an interruption. Set `WordTimestamps` to false to fall
  back to aggregated text frames. Both accept a voice, language, speaking rate,
  latency-optimization level and text normalization, and both emit raw 16-bit
  PCM at the configured rate.

- **xAI speech-to-text** (`provider/xai/grok.NewSTT`), a streaming transcription
  service over xAI's real-time WebSocket API. It holds one connection for the
  whole session, configures itself through handshake query parameters (encoding,
  language, endpointing silence, interim results, diarization and multichannel
  transcription), and surfaces the server's partial results as interim
  transcriptions and its endpointed results as finalized ones. Audio is withheld
  until the server acknowledges the session, and closing signals end of audio so
  the last transcript is flushed. `provider/xai/grok` previously offered only the LLM.

- **Opus inband FEC.** `transport.Params` gained `AudioOutFEC` and
  `AudioOutExpectedPacketLoss`, which enable forward error correction on the
  outgoing stream so receivers can rebuild dropped packets instead of concealing
  them. Worth enabling whenever clients may be on lossy links (mobile radio,
  relayed media); on a clean link the redundancy is never decoded. Honored by
  the `libopus` build; the pure-Go SILK encoder ignores it.

- **Pipeline lifecycle frames.** A processor with no handle on the `Task` can now
  ask it to change the run's lifecycle, and the `Task` converts the request into
  the matching pipeline-wide frame: `frames.CancelWorkerFrame` becomes a
  `CancelFrame`, `frames.StopWorkerFrame` a new `frames.StopFrame` (stop the run
  but leave processors running), and `frames.InterruptionWorkerFrame` an
  `InterruptionFrame`. `frames.EndWorkerFrame` already did this. The four now
  live together in `frames/worker.go`; push them downstream so frames queued
  ahead are processed first, and the `Task` sink returns them upstream.
- **`Task.Flush`**, a drain barrier. It queues a `frames.PipelineFlushFrame`
  probe that travels to the sink, is bounced back to the source, and closes its
  `Done` channel on arrival. When `Flush` returns, every frame queued ahead of
  the probe has been processed — useful to let a pipeline settle after an
  interruption before injecting new work. The probe is uninterruptible, so it
  always completes its round trip and a waiter cannot be stranded.
- **`frames.CancelWorkerFrame` separates cancellation from failure.** Stopping a
  healthy run (the caller hung up, a supervisor asked to stop) no longer has to
  be dressed as a fatal error, and carries a `Reason` instead of surfacing as an
  `ErrorFrame` to observers and clients. `frames.FatalErrorFrame` is its
  counterpart for genuine unrecoverable failures.
- **Runtime LLM context changes.** `frames.LLMSetToolsFrame`,
  `frames.LLMSetToolChoiceFrame` and `frames.LLMMessagesUpdateFrame` change the
  toolset, the tool-choice policy and the whole conversation on a running
  pipeline. The context aggregator applies each to the shared `LLMContext` and
  forwards it downstream. Backing them, `LLMContext` gains `SetMessages`,
  `ToolChoice` and `SetToolChoice`.
- **`frames.DefaultAudioInSampleRate` / `DefaultAudioOutSampleRate`**, the rates
  `NewStartFrame` applies, exported so applications can refer to them.

### Changed

- **Providers are grouped by vendor.** A vendor offering several services now
  has one folder per service under a folder of its own, so the tree reads by
  vendor rather than by a flattened name. Import paths change accordingly:

  | Was | Now |
  |---|---|
  | `provider/openai` | `provider/openai/chat` |
  | `provider/openairealtime` | `provider/openai/realtime` |
  | `provider/azureopenai` | `provider/azure/openai` |
  | `provider/azurespeech` | `provider/azure/speech` |
  | `provider/google` | `provider/google/gemini` |
  | `provider/geminilive` | `provider/google/live` |
  | `provider/xai` | `provider/xai/grok` |
  | `provider/xairealtime` | `provider/xai/realtime` |
  | `provider/bedrock` | `provider/aws/bedrock` |
  | `provider/polly` | `provider/aws/polly` |
  | `provider/transcribe` | `provider/aws/transcribe` |
  | `provider/novasonic` | `provider/aws/novasonic` |

  The package names follow the last path element, so `openai.LLMConfig` becomes
  `chat.LLMConfig`, `google.NewLLM` becomes `gemini.NewLLM`, and so on. Only the
  import path and package name change; every config field, constructor and
  behavior is untouched. A provider offering a single service (`deepgram`,
  `cartesia`, `nvidia`, …) keeps its top-level folder.

- **`opus.NewEncoder` takes an `opus.EncoderConfig`** instead of positional
  `channels, bitrate` arguments, matching the config-struct convention used
  elsewhere and leaving room for the new FEC settings.
- **Speech-to-speech services are told when tools change.** A realtime model
  generates continuously and never re-reads the shared context between turns, so
  a toolset changed with `LLMContext.SetTools` never reached it. Tool changes now
  travel as `frames.LLMSetToolsFrame`, and the OpenAI Realtime service renders
  tools into its initial session configuration and sends a fresh `session.update`
  whenever they change. `SetTools` and `SetToolChoice` remain for seeding a
  context before the pipeline starts.
- **`frames.OutputTransportMessageFrame` is a data frame again**, delivered in
  order with the surrounding audio, for messages that must land in step with what
  the bot is saying. The new `frames.OutputTransportMessageUrgentFrame` is the
  system-frame variant that goes out immediately, ahead of queued audio; the RTVI
  processor uses it. Previously the single frame had urgent semantics under the
  ordered name and the ordered variant was unavailable.
- **`frames.MixerControlFrame` is now the interface for the mixer control
  family**, with `frames.MixerUpdateSettingsFrame` carrying the settings map and
  `frames.MixerEnableFrame` carrying an `Enable` flag, replacing the single
  settings-carrying frame.
- **`frames.EndWorkerFrame` is a control frame** (uninterruptible) rather than a
  system frame, so it is ordered behind the work already queued.
- **`turns.Emitter.Broadcast` takes a frame constructor** rather than a frame,
  and builds one instance per direction, cross-linked by `BroadcastSiblingID`.
  The two directions are processed on separate goroutines, so sharing a single
  frame between them was a latent data race. Observers ignore the upstream half
  of a pair, so a broadcast event is still reported once.
- **`frames.Frame` is a four-method interface** (`String`, `ID`, `Name`,
  `Base`). The optional per-frame state — presentation timestamp, metadata,
  transport source and destination, broadcast sibling id — lives on `BaseFrame`
  and is reached through `Base()`; the accessors are still promoted onto every
  concrete frame, so code holding a concrete type is unaffected.
- `CancelFrame.Reason`, `EndFrame.Reason` and `EndWorkerFrame.Reason` are
  `string` rather than `any`.
- `frames.STTMetadataFrame` embeds `ServiceMetadataFrame`, as
  `LLMServiceMetadataFrame` already did, and moves to `frames/metadata.go`
  alongside it. `FunctionCallCancelFrame` moves to `frames/function.go` with the
  other function-call frames.
- `AudioRawData.NumFrames` is derived on each call instead of cached at
  construction, so it stays correct when `Audio` is replaced.
- The `frames` package documents its ownership rule: a frame has a single owner
  at a time and must not be mutated after it is pushed.

### Fixed

- **The Pion transport sends on every frame boundary for the life of the
  session**, falling back to silence when nothing is queued, instead of writing
  only when the pipeline handed it audio. RTP timestamps advance one frame per
  packet however much wall clock passed, so a sender that went quiet during a gap
  — the pause between one synthesized sentence and the next — and then resumed
  left the audio after it timestamped as though the gap never happened. Receivers
  schedule playout from those timestamps, so the gap read as network delay:
  conceal by repeating the last frame, then compress once packets bunched up
  again, heard as stuttered and clipped words. With the sender writing every
  frame, elapsed frames and elapsed time cannot diverge. This is the shape
  `transport/localaudio` already had, since a device's playback callback pulls on
  its own clock and has to be given something; `transport/pionrtc` could push
  instead, and drifted. `transport/livekit` still pushes and is likely affected,
  but is untested here.
- **The `libopus` encoder rejects wrong-sized frames** instead of reading past
  the end of the caller's buffer: `opus_encode` consumes a fixed frame regardless
  of the slice length, so a short frame read out of bounds. It now returns the
  same error the pure-Go encoder does, as the shared API promises.

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

- **Memory recall failed against mem0 2.x** (`provider/mem0`): a search sent the
  user, agent and run ids as top-level fields, which `search` rejects from 2.0.0
  on — every retrieval failed with a 502 and the bot ran with no long-term
  recall, while storage kept working. They now travel nested under `filters`,
  the shape search expects. Storage still passes them top-level, which `add`
  accepts.

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
