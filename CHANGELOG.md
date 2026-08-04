# Changelog

All notable changes to jargo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Development status.** While jargo is in early development the version stays
> in the `0.0.x` range: the public API is unstable and may change in any
> release, with no backwards-compatibility guarantees. `0.1.0` will mark the
> first release intended for wider use.

## [Unreleased]

### Changed

- **A task traces the session it runs**, so a call is one trace shaped like the
  conversation it recorded: a `conversation` span at the root, a `turn` span per
  turn beneath it, and each turn's STT, LLM and TTS spans beneath that.
  `pipeline.TaskParams` gains `EnableTracing`, `ConversationID`,
  `AdditionalSpanAttributes` and `EnableTurnTracking`; the turn spans carry
  `turn.number`, `turn.duration_seconds`, `turn.was_interrupted` and
  `turn.user_bot_latency_seconds`, and the conversation span carries whatever
  attributes the caller adds — which is where a backend's own grouping keys go
  (`langfuse.session.id`, `langfuse.user.id`). Services parent their spans to the
  turn through the new `tracing.TracingContext`, handed to them at setup as
  `processor.Setup.Tracing` and reachable as `processor.Base.Tracing()`, so a span
  raised away from the frame path — a TTS synthesis played out on the audio queue
  — lands under the turn it belongs to instead of starting a trace of its own.
  `tracing.StartConversation` is gone: the task opens the conversation span now,
  and `observers.TurnTrace` writes it.
- **A websocket service no longer reconnects once its session is over.** A read
  that fails because the context is done would redial on that same finished
  context and fail every attempt, so the end of every call logged a burst of
  connection errors that read like a provider outage.
- **A `MetricsFrame` carries a list of measurements**, so one frame can report
  several kinds, and several processors, at once. Where it had a fixed set of
  optional fields it now has `Data []frames.MetricsData`, whose concrete types
  are `TTFBMetricsData`, `TTFAMetricsData`, `ProcessingMetricsData`,
  `LLMUsageMetricsData`, `STTUsageMetricsData`, `TTSUsageMetricsData`,
  `TextAggregationMetricsData` and `TurnMetricsData`, each carrying the processor
  that measured it and the model it is attributed to. Read them by switching on
  the type. `frames.TurnPrediction` is gone, replaced by `TurnMetricsData`, and
  `NewMetricsFrame` now takes the measurements rather than a processor name. A
  frame that reported one processor could not say what a pipeline-wide report
  needs to say.
- **A TTS service reports what grouping text into sentences costs**, as a
  `TextAggregationMetricsData` on a `MetricsFrame`: the wait from a model's first
  token to the sentence it completes, which is the delay before synthesis of that
  sentence can start. Nothing is reported when the service passes tokens straight
  through, since per-token aggregation time means nothing. `ttstext.Aggregator`
  gains `Type`, reporting how it groups text, which is what says whether there is
  anything to measure.
- **A speech-to-text service reports the audio it was given in band**, as an
  `STTUsageMetricsData` on a `MetricsFrame`, gated on `EnableUsageMetrics` like
  the token usage an LLM reports. It was measured already but only ever reached
  OpenTelemetry, so an in-band consumer such as an RTVI client could bill LLM and
  TTS use but not STT.
- **The RTVI metrics message gained `ttfa`, `stt_usage` and `text_aggregation`**,
  and now reports every measurement a frame carries rather than one processor's.
  Time to the first audible sample was measured and dropped on the way to the
  client.
- **Word timings are reported to the TTS base a batch at a time**, not one token
  at a time: `tts.WordTimestamps.RunTTSTimed` now takes
  `word(words []uctx.WordTiming, opts tts.WordTimingOptions)`, and
  `tts.AudioContextHost` gains `AddWordTimestamps` for a provider delivering on
  its own receive loop. Normalizing a token stream can need the token before it,
  which the base could not see one call at a time.
- **Punctuation merging moved from the providers into the base**, asked for with
  `tts.WordTimingOptions{PreMergeTokens: true}` and off by default. It was done
  by each provider that happened to need it, so of the four reporting word
  timings two merged and two did not, with no way to tell from the outside which.

- **Voice detection gates on volume as well as confidence.** `vad.Params` gains
  `MinVolume`, 0.6 by default, and a frame counts as speech only when the model
  is confident enough about it and it is loud enough to be worth hearing. A
  confident guess at something barely audible no longer opens a turn. Set it to 0
  for the previous confidence-only behaviour.
- **`audio/loudness`**, measuring loudness to ITU-R BS.1770 (EBU R128): the
  K-weighting filter pair, gated block loudness, and normalisation onto a 0..1
  scale. It is what the volume gate reads, so the gate follows how the ear hears
  rather than raw amplitude.
- **Voice detection moved out of the processor into `audio/vad/controller`.**
  The controller owns the detection state, the resampling the detector needs and
  the watch for audio that stops arriving, so anything needing the same detection
  can drive it without going through a pipeline. `processor/vadproc` now just
  hosts one and turns what it reports into frames.
- **The VAD frames are broadcast rather than pushed downstream.**
  `VADUserStartedSpeakingFrame`, `VADUserStoppedSpeakingFrame` and
  `UserSpeakingFrame` now reach processors on both sides of the detector, which
  is what an interruption decision upstream needs.
- **`UserSpeakingFrame` is emitted for every chunk heard as speech**, rather than
  at a fixed period. `vadproc.Config.SpeechActivityPeriod` is gone with the
  throttle it configured.
- **`frames.SpeechControlParamsFrame` carries the parameter sets themselves.** It
  held three flattened turn timings and no VAD parameters at all; it now carries
  `*vad.Params` and `*turn.Params`, either of which may be nil. It is reported
  when the pipeline starts and whenever the parameters change.
- **A barge-in keeps the frames that must survive it.** The output dropped
  everything it had queued when the user interrupted, uninterruptible frames
  included, so a frame marked to always be delivered was not. It now keeps them
  and drops the rest. When nothing queued has to survive and no mixer is running,
  the sender is restarted instead, which cuts short a write already in flight
  rather than leaving the barge-in waiting behind it.
- **The output's audio queue is no longer bounded at 256 frames.** Expressing
  "drop these but keep those" needs a queue rather than a channel, and the queue
  is unbounded, so a producer running ahead of playback is no longer blocked by
  it.
- **`Task.Flush` now waits for queued audio to play**, a consequence of frames
  leaving the output in step with the audio: the probe is queued like anything
  else, so it completes once what was queued ahead of it has been paced out. It
  is uninterruptible, so a barge-in still lets it finish rather than stranding a
  waiter.
- **Frames leaving the output wait for the audio they were queued behind.** A
  downstream frame carrying neither audio nor a presentation timestamp used to be
  forwarded the moment it arrived, overtaking the audio around it by however much
  was buffered. It is now queued behind that audio and forwarded in step with
  playback. System frames still go straight through, as does anything travelling
  upstream.
- **`frames.OutputTransportMessageFrame` is sent in step with the audio around
  it** rather than the moment it arrives, which is what makes it the ordered
  counterpart of the urgent one. The urgent frame is still sent at once, and is
  no longer also forwarded downstream.
- **Mixer control frames stop at the output** rather than being forwarded on.
  They address the mixer of the stream they name and have no meaning past it.
- **`transport/pionrtc` is now `transport/rtc`.** Pion is how the WebRTC
  transport is implemented, not what it is, and the package name said the
  former. Update imports and any `pionrtc.` references to `rtc.`.
- **`transport.OutputDriver.WriteAudio` takes a frame** rather than a bare PCM
  buffer, so a transport can read the outgoing stream a chunk is addressed to
  alongside its samples. Implementations take a `frames.OutputAudioFrame` and
  read the samples from `AudioData().Audio`.
- **The frames carrying output audio are now a family**, `frames.OutputAudioFrame`,
  covering `OutputAudioRawFrame`, `TTSAudioRawFrame` and the new
  `SpeechOutputAudioRawFrame`. Match the interface rather than a concrete type,
  or audio of a kind that was not named goes unhandled. `InputAudioRawFrame` is
  outside the family, since it is never sent.

### Added

- **The Smallest AI transcription carries every option the service accepts**:
  `WordTimestamps`, `FullTranscript`, `SentenceTimestamps`, `RedactPII`,
  `RedactPCI`, `Numerals`, `Diarize`, `Endpointing`, `Keywords` and `Format`,
  alongside the model. Endpointing and formatting are on unless turned off, as
  they are on the service itself.
- **AssemblyAI takes a list of declared languages**, as
  `Config.LanguageCodes`: one pins the transcription to that language, several
  steer towards that set while still letting the speaker switch between them, in
  the order given. Regional variants resolve to their base code, so at most ten
  distinct languages, which `Validate` checks rather than leaving the service to
  close the session over it. The steering is prompt-based, so it is sent to U3
  Pro models alone.
- **`provider/kyutai/pockettts`**, speech synthesis from a local Pocket TTS
  server (`pocket-tts serve`): a small CPU-only model whose audio streams back as
  it is generated, so the first samples arrive well before the sentence is
  finished. The voice is per request, either a name the server knows or a URL of
  a recording to clone; the language belongs to the server, which loads its
  weights per language when it starts. It sits under `provider/kyutai` beside
  the moshi-server models, which are a separate server of the same vendor.
- **The LiveKit transport reads inbound SIP DTMF.** A key pressed by a caller on
  a SIP leg of the room becomes an `InputDTMFFrame`, so the DTMF aggregator works
  on a LiveKit call as it does on a telephony one. `frames.KeypadEntry.Valid`
  reports whether what arrived is a key at all.
- **Azure speech synthesis takes `ForceLocale`**, wrapping the text in SSML's
  `<lang>` element so a multilingual voice speaks in the configured language
  rather than the one it reads out of the text.
- **The OpenAI transcription and synthesis take an `HTTPClient`**, for a longer
  timeout, a proxy, or connection limits of your own. Every provider built on
  them (Groq, Together, Kokoro's server and the rest) takes it too. Leaving it
  nil is unchanged.
- **Deepgram Flux takes `Numerals`**, writing spoken numbers as digits. It is
  fixed when the session opens, which is all Flux accepts for it.
- **ElevenLabs realtime transcription takes `FilterBackgroundAudio`**, having
  the service strip background audio before it transcribes.
- **Gemini takes content-safety filters**, as `gemini.Config.SafetySettings`
  (and the same field on the Vertex AI config): a harm category and the
  threshold at which the model blocks content for it, one entry per category.
  Configuring none leaves every category at the API's default, as before.
- **Nova Sonic reports its token usage.** The service accounts for each turn in
  a `usageEvent` the read loop was dropping, so a speech-to-speech session
  billed by token reported nothing at all. Each event's own tokens are reported
  rather than the running total, which is how the other realtime services report
  usage, and the speech and text split it carries is kept in the per-modality
  counts.

### Changed

- **The Smallest AI transcription opens the Waves v4 session** at
  `/waves/v1/stt/live` and flushes each utterance with a `finalize` message when
  the VAD reports the speech ended, keeping the session open for the next one.
  A `Stream` asks for that by implementing the new `stt.Finalizer`; a provider
  that does its own endpointing is left alone.
- **The Cartesia transcription connects with `Cartesia-Version: 2026-03-01`**,
  the version its TTS and turn-taking services already used.

### Deprecated

- **`provider/xtts` is deprecated** and will be removed. It has no replacement:
  `provider/kokoro` and `provider/piper` are the maintained local speech
  synthesis services.

### Fixed

- **A WebSocket service stops redialing a handshake the server refused.** A
  rejected key or a model an account cannot use was retried three times with a
  backoff in between, delaying the error the caller needed to see. `wsutil.Dial`
  now returns a `*wsutil.HandshakeError` carrying the refused status, and the
  reconnect gives up on a 4xx while still retrying anything that might not
  repeat.
- **Polly escapes the XML-reserved characters in the text it speaks.** An
  ampersand or an angle bracket in a reply broke the SSML document, which Polly
  rejects outright, so the whole sentence went unsaid.
- **The token total an Anthropic model reports counts its cached input.** The
  service reports the input count net of the prompt cache, so adding only the
  prompt and completion counts left the cached tokens out of the total and
  understated a cached turn, sometimes by most of the prompt. AWS Bedrock reads
  the same, since it runs through the same service. The doc on
  `frames.LLMTokenUsage` now says which of its counts are gross and which are
  net.
- **The Sarvam LLM defaults to `sarvam-105b`**, the model the service now
  offers; `sarvam-30b` is gone.
- **A WebSocket close waits at most two seconds for the peer to answer.**
  Disconnecting happens while a service handles the `EndFrame`, before the frame
  carries on downstream, so a peer that never acknowledges the closing handshake
  held up the end of the pipeline for the library's own five seconds, once per
  service. `wsutil.Dial` now returns a `*wsutil.Conn` whose `Close` gives up
  after `wsutil.DefaultCloseTimeout`, adjustable per connection with
  `SetCloseTimeout`. The providers that dialled the library directly go through
  `wsutil.Dial` too, so every provider socket is bounded the same way.
- **A settings update reaches the service a switcher is not currently using.**
  `ServiceSwitcher` gated every non-system frame on the active service, so an
  update addressed to one of its other services never arrived, and
  `ReachInactiveServices` had nothing acting on it: a setting meant to survive a
  switch was lost at the switch. Both now pass the gate, and the merge's
  existing deduplication keeps a single copy leaving the switcher. Any other
  settings update still applies to the active service alone, since a setting is
  usually specific to one provider. Every settings frame satisfies the new
  `frames.SettingsUpdate` interface, which is what lets a processor route one
  without knowing which kind of service it names.
- **A currency amount with more decimals than the currency has is read to
  subunit precision.** The match stopped after two fractional digits, so
  "$5.500" was spoken as "five dollars and fifty cents" with the third digit
  left in the text right behind the word: "cents0". The fraction is now consumed
  whole and read to the subunit, dropping the precision below it.
- **A user-idle timeout update applies at once.** `UserIdleTimeoutUpdateFrame`
  changed the stored timeout and nothing else, so a retune took effect only at
  the next bot turn: a running timer ran out on the old duration, and turning
  idle detection on while the bot was already waiting for the user armed nothing
  at all until it spoke again. The controller now tracks that waiting window and
  restarts (or arms) the timer on the update.
- **The turn analyzer's safety net is anchored to the end of the user's
  speech.** The wait for a transcript was measured from the moment the timer was
  armed, which is after the end-of-turn model has run, so every millisecond of
  inference was added to the budget instead of taken out of it. It is now
  measured from the instant the speech itself ended, which is the VAD's own stop
  window before it reported the stop. `VADUserStoppedSpeakingFrame.Timestamp` is
  a `time.Time` for this: an RFC3339 string resolved to the second, which is
  coarser than the budget it has to be subtracted from.
- **A parallel pipeline holds system frames while it synchronizes a lifecycle
  frame.** It waited for a `StartFrame`, `EndFrame` or `CancelFrame` to reach
  every branch by blocking on the goroutine handling it. An `EndFrame` is a
  control frame, so that left the system-frame path open: an interruption
  arriving mid-shutdown was fanned into the branches, where it flushed the output
  they still had queued to send ahead of the `EndFrame`. The synchronization now
  pauses both paths, which is what the wait was for.
- **A parallel pipeline no longer swallows a lifecycle frame a branch raised
  itself.** Such a frame was never fanned out, so it had no synchronization
  counter and was dropped at the branch sink. A processor inside a branch ending
  the session, an idle monitor or a transport hanging up, could not end the
  pipeline. A frame with no counter is now released like one whose branches have
  all reported.
- **An output transport must return from `WriteAudio` once its context is
  done**, which is now stated on `OutputDriver`. Stopping an output cancels the
  send loop's context and then waits for the loop to finish, and the loop sits
  inside `WriteAudio` whenever it is sending, so a write that blocks past
  cancellation holds the pipeline open for good. With a mixer the loop writes
  whether or not anything is queued, so at shutdown there is nearly always a
  write in flight and a transport that ignores cancellation wedges every
  shutdown rather than an occasional one. Every transport in the tree already
  waited on the context; the contract was simply never written down.
- **The end of a pipeline waits for the audio still in flight.** A bot stopped
  right after queueing its farewell said about half of it: every other frame the
  TTS base emits goes through the serialization queue, which holds it until the
  contexts queued ahead have drained, but the `EndFrame` was pushed straight
  downstream. It reached the output while a provider delivering on its own
  receive loop was still sending audio, so the output drained what little it had
  and stopped. The `EndFrame` now shuts the serialization queue down and waits
  for it, which puts the last of the audio, and anything queued behind it such as
  a closing chime, downstream before the frame that ends the pipeline.

- **A tool call no longer holds up the frames queued behind it.** Handlers ran on
  the goroutine that processes the LLM service's frames, so nothing queued while
  one was in flight moved until it returned — including the speech a bot plays to
  cover the wait, which is defeated by exactly the slow tools it exists for. Each
  call now runs on its own goroutine; the assistant aggregator already holds the
  results until the last one is in, so calls finishing out of order costs nothing,
  and a result arriving after an interruption is dropped rather than added to the
  synthetic one that already balanced the call.

- **Time to first byte is recorded for a turn that only requests tools.** It was
  measured on the first text delta, so a generation whose whole answer is a
  tool-use block reported no TTFB at all — the turns where the wait is longest
  and least visible. `llm.Base` gains `StartTTFBMetrics` and `StopTTFBMetrics`,
  and each LLM service records it around the point its response stream opens.
  For an SDK that retries inside the call that opens the stream, the retry is now
  part of the measurement rather than invisible.

- **A barge-in during playback no longer races the sequencer.** Clearing it when
  the user interrupts runs on the goroutine processing frames, while completing
  its slots runs on the one draining a context's audio, and nothing kept the two
  apart: concurrent map access, which can corrupt the queue or panic outright.

- **The text about to be spoken is announced again.** An `AggregatedTextFrame`
  describing each unit is emitted before the audio for it opens, so a consumer
  that wants the text ahead of hearing it can act on it: an RTVI client starts
  its segment from this frame, and the progress frames that follow refer back to
  it. It was built and handed to the sequencer, which stored it and returned
  nothing, so it never reached the pipeline at all.

- **The assistant's turns reached the conversation again** with a TTS provider
  that delivers its audio on its own receive loop rather than inline, and no word
  timestamps. With no word timings the whole-unit text frame is the only thing
  carrying a turn into the context, and it was emitted only when the provider had
  answered inline, so every turn of such a provider was left out. Each new turn
  then saw nothing but a run of user messages, and the bot repeated itself and
  re-asked what it had just been told.

### Added

- **`service/settings`, the settings a service can be given while the pipeline
  runs**, with `settings.LLM`, `settings.TTS` and `settings.STT` covering the
  model, voice, language and the sampling knobs an LLM exposes. `settings.Apply`
  merges an update into what a service holds and reports which fields moved and
  what they held before, so a service reacts only to a real change. A field
  carries three states rather than two: not given, given a value, and given no
  value, the last being how a caller asks a service to drop a setting rather than
  leave it alone. `settings.FromMap` builds an update from plain names and
  values, resolving aliases and keeping provider-specific keys, for one arriving
  over the wire. The carriers are `LLMUpdateSettingsFrame`,
  `TTSUpdateSettingsFrame` and `STTUpdateSettingsFrame`: control frames, applied
  in order with the speech around them, and uninterruptible, so a barge-in
  arriving at the same moment cannot drop them. `service/stt.StreamService`
  reads them: a `Connector` implementing `stt.SettingsHolder` has updates merged
  into its own store, `stt.SettingsUpdater` is told what changed and may ask for
  the session to be reopened, and `stt.LanguageNamer` names a language the
  provider's way before the update is applied, so a neutral name and the
  provider's own code are not mistaken for a change. `service/tts.Base` and
  `service/llm.Base` read them too, through the same optional interfaces on the
  `Synthesizer` and the `Generator`, and a model that changes relabels the
  metrics it is measured and priced against. Only STT reopens its session for a
  change, because only there does the base own the connection; a synthesizer
  owns its own and reconnects for itself. **Deepgram STT is the first provider
  wired**: `deepgram.Settings` carries the fifteen fields the provider treats as
  changeable, plus the model and language, and a change to any of them reopens
  the session, since Deepgram takes them as query parameters when the session
  opens and has no way to be told afterwards. What is fixed at build time (the
  endpoint, encoding, channels, VAD events, version, tags) stays on `Config`.
  Cartesia STT follows, with the model, the language and keyterms, and Soniox
  STT with the model, the language hints and the nine options it is told in its
  handshake. All three reopen the session on any change, because all three are
  told their configuration only when the session opens. Cartesia's turn-detecting
  service reopens for a keyterm change alone and reports any other change as one
  that cannot take effect. The remaining providers are unchanged and keep taking
  their configuration at construction.
- **Cartesia and Soniox STT gained the options they were missing.** Cartesia
  takes `Keyterm`, capped at the 100 terms and 1200 characters a connection
  accepts and sent only on the ink-2 models that honor them, with spaces
  percent-encoded as Cartesia requires. Soniox takes `LanguageHintsStrict`,
  `Context`, `EnableSpeakerDiarization`, `EnableLanguageIdentification`,
  `MaxEndpointDelayMs`, `EndpointSensitivity`,
  `EndpointLatencyAdjustmentLevel` and `ClientReferenceID`. An option nobody sets
  is left out of the request, so the service applies its own default rather than
  being sent a zero that means something else.
- **A transcription session reopened for a settings change waits for the user to
  stop speaking**, and the audio arriving in between is held and sent on once the
  new session is up. Replacing the session mid-sentence would drop what is being
  said, and the words lost would be the ones the change was meant to transcribe
  better. A change asked for while nobody is speaking takes effect at once.
- **A streaming speech-to-text session can be held open while it carries no
  audio**, for the providers that close an idle one. Silence reaches the service
  whenever nobody is speaking, and a service switched out of the pipeline sends
  nothing at all, so an idle session is ordinary and losing it costs the next
  thing the user says. A `Connector` implementing `stt.Keepaliver` says how long
  a session may go quiet before silence is submitted to hold it open, and how
  often to look. A `Stream` implementing `stt.KeepaliveSender` sends something of
  the provider's own instead, for a service with a protocol message for this.
  Silence submitted as audio counts towards the session's usage, since the
  provider bills for it; a protocol message does not. A `Connector` that
  implements neither gets no keepalive.
- **A streaming speech-to-text session that drops is reopened**, so a network
  blip no longer costs transcription for the rest of the call. The read loop
  ended with the first failed read before, and nothing dialed again: every
  provider that transcribes over a live connection went silent for good. The new
  `service/wsservice` package holds that loop and the reconnection around it,
  retrying with an exponential backoff and reporting each failed attempt as a
  non-fatal `ErrorFrame`. `service/stt.StreamService` drives it, so all 18
  streaming providers get it without a change of their own. A session closed
  normally is not reopened, since it ended because it was meant to. Neither is
  one that keeps dying the instant it opens, which is what a server rejecting
  credentials after the upgrade looks like: `utils/network.QuickFailureTracker`
  spots it and stops, because waiting longer between attempts cannot help.
  `utils/network.ExponentialBackoffTime` computes the wait.
- **`pipeline.NewSyncParallel`**, a parallel pipeline that holds the output of
  each input frame until every branch has finished producing it, so everything
  the branches produced for one input is released together. It sends a
  `pipeline.SyncFrame` in behind each frame to know when a branch is done, which
  needs the last processor of each branch to be synchronous.
  `pipeline.FrameOrder` picks whether the collected frames go out as they arrive
  (`FrameOrderArrival`, the default) or branch by branch (`FrameOrderPipeline`),
  for when the order between branches matters. Use it where output has to stay
  together; where branches are independent, `pipeline.NewParallel` stays the
  lighter choice.
- **A compound processor reports what it contains**: `Processors`,
  `EntryProcessors` and `ProcessorsWithMetrics` on `Processor`, implemented by
  `Pipeline`, `ParallelPipeline` and `SyncParallelPipeline`; every other
  processor reports nothing. `CanGenerateMetrics` marks a processor that reports
  metrics, which the STT, LLM, TTS and speech-to-speech services do.
- **`TaskParams.SendInitialEmptyMetrics`**, sending one `MetricsFrame` once the
  pipeline is ready that carries a zeroed time to first byte and processing time
  for every processor reporting metrics, so a consumer knows which processors to
  expect metrics from before any have been measured. It applies only when
  `EnableMetrics` is set, and nil defaults to true.
- **`tts.Base.SetPauseFrameProcessing`**, pausing a TTS service's frame handling
  from the moment a turn's text has been sent to the provider until the audio for
  it has played, so the next turn cannot be synthesized over it. It is off unless
  asked for. A watchdog force-resumes, and reports a non-fatal error, if nothing
  confirms the audio is playing within `PauseOptions.WatchdogTimeout` (3s by
  default), so a turn that produces no audio cannot pause the service for good.
- **Pausing a processor's frame handling**: `Base.PauseProcessingFrames` and
  `ResumeProcessingFrames` hold data and control frames, and
  `PauseProcessingSystemFrames` and `ResumeProcessingSystemFrames` hold system
  frames. Held frames stay queued, in order, and are handled on the resume.
  `frames.FrameProcessorPauseFrame` and `FrameProcessorResumeFrame` ask for the
  same thing in band, with `FrameProcessorPauseUrgentFrame` and
  `FrameProcessorResumeUrgentFrame` as the system-frame variants that overtake
  the queue. `ParallelPipeline` is the first caller.
- **`frames.VADParamsUpdateFrame`**, changing the detection parameters on a
  running pipeline. It is pushed upstream (by the RTVI processor acting on a
  client request, say), and the analyzer adopts them from the next chunk.
  `vad.Analyzer` gains `SetParams` for it.
- **`vadproc.Config.AudioIdleTimeout`**, ending the user's speech when the audio
  stops arriving mid-utterance. Voice detection only ever hears silence as speech
  ending, so audio that stops outright, a microphone muted part-way through being
  the usual case, left the user speaking for good and the turn never closed. One
  second by default; a negative value disables it.
- **`Params.AudioOutEndSilenceSecs`**, how many seconds of silence follow the
  last of the audio when the pipeline ends, so the closing words are not clipped
  by whatever closes on top of them. Two seconds by default; 0 sends none.
- **`frames.OutputTransportReadyFrame`**, pushed upstream once the output
  transport has opened its media path and can receive frames. A producer that
  must not speak into a connection that is not up yet can wait for it.
- **`transport.OutputDriver.StartWriting`**, the hook a transport implements to
  open its outgoing media path. The base calls it while starting, and only once
  it returns does it start the senders and report the transport ready, so
  nothing is queued for a path that cannot carry it yet. Transports with nothing
  to open need not implement it.
- **Several outgoing audio streams per transport.** `Params.AudioOutDestinations`
  names them, `OutputDriver.RegisterAudioDestination` opens each one, and every
  frame is routed to the stream its `TransportDestination` names. Each stream
  keeps its own buffer, mixer, chunking and bot-speaking state, so a turn on one
  neither shares a buffer with nor silences another. A frame naming a stream that
  was never registered is dropped with a warning.
- **`Params.AudioOutMixers`**, mapping a destination to the mixer serving it.
  `Params.AudioOutMixer` still serves the default stream, and now serves only
  that one rather than every stream at once.

- **`mem0.Config.APIKeyHeader`**, naming the header the API key is sent in. The
  key went out as `Authorization: Token <key>`, which is what the managed mem0
  API expects; a self-hosted server reads `X-API-Key` and ignores that form, so
  a secured self-hosted deployment rejected every search and store. Empty keeps
  the managed default, and `mem0.HeaderXAPIKey` names the self-hosted one.
- **ElevenLabs WebSocket text-to-speech** (`provider/elevenlabs.NewRealtimeTTS`),
  the multi-stream-input protocol over a single connection held open for the
  session. `NewTTS` issues an HTTP request per sentence, and since the TTS base
  synthesizes a sentence at a time, every sentence boundary in a reply paid for
  connection setup before its first audio came back — a pause the listener hears.
  Each synthesis opens a context, sends its text, and closes it, which is what
  makes the final marker arrive immediately after the last audio byte rather than
  after the server waits to see whether more text is coming. Audio is attributed
  by context id, so what the server had already generated for a synthesis
  abandoned by an interruption cannot leak into the next sentence. Optionally
  reports word timing (`WordTimestamps`): ElevenLabs times every character, and
  those are assembled into words with `utils/context.CharAccumulator`, which lets
  the assistant context record what was actually spoken when a turn is cut short.
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

### Added

- **`tts.Starter`**, the counterpart to `tts.Closer` at the other end of the
  pipeline's life: an optional interface a Synthesizer implements when it has
  setup to do before the first sentence, such as dialing the connection it will
  reuse. Upstream every WebSocket TTS service connects in its `start(StartFrame)`
  and none dials lazily, so this closes a gap rather than adding an optimization —
  the Synthesizer contract had no way to take part in the pipeline starting.
  `provider/elevenlabs.NewRealtimeTTS` implements it, having dialed on first use:
  its handshake measured 2.3 s on a session's first synthesis against roughly
  120 ms for every one after, all of it in the silence before the bot's opening
  words.

### Fixed

- **A single WebSocket message larger than a megabyte failed the read.** The
  shared read limit was set on the premise that a provider's base64 audio chunks
  fit comfortably below it; a TTS provider generating a long sentence in one go
  sends well past that, and the synthesis died part-way through a reply with
  `message too big`. The limit exists to stop a server streaming without end, not
  to bound a legitimate message, so it is now 16 MiB.
- **The ElevenLabs WebSocket TTS produced no audio at all.** `auto_mode` was only
  sent when the caller set it, and with it absent the server buffers text until an
  explicit flush — then discards whatever was never flushed when the context
  closes, answering with a final marker and an empty audio field. Every synthesis
  therefore completed, quickly and without error, having emitted nothing.
  `auto_mode` now defaults on, as upstream does; a caller driving generation with
  its own flushes can still turn it off. The fake server in the tests accepted the
  broken sequence, which is why this passed CI: it now generates only when auto
  mode is on or a flush was sent, as the real one does.
- **`MinWordsStart` drops back to the single-word threshold as soon as a turn
  opens.** It tracked the bot's speaking state only from the bot-speaking frames,
  so after a barge-in it kept requiring `MinWords` until the interrupted bot's
  stopped frame arrived, and the rest of that turn was held to the interruption
  bar rather than the ordinary one. It now clears the flag on turn start, matching
  the upstream strategy's `handle_user_turn_started`.
- **`mem0` searches once per user message rather than once per context frame.** A
  tool-calling turn replays the context after every round trip, and each replay
  repeated the same retrieval — latency and load on the critical path before the
  reply, for a result already in hand. Retrieval is now keyed on the user message
  it was performed for. `Config.SearchTimeout` bounds that blocking retrieval
  separately from `Timeout`, since a late memory is worse than a missing one when
  a reply is waiting on it, and `Config.Prewarm` issues one throwaway search when
  the session starts so the first real one does not pay to warm the path.
- **The Pion sender's starvation counter no longer charges every utterance for
  its own ending.** It counted any silence sent within a fixed window after real
  audio, but the window is also exactly what a normal end of speech looks like, so
  the figure tracked how many times the bot spoke rather than how often the writer
  fell behind — it read as a fault on every healthy session. A run of silence is
  now only charged once real audio resumes after it, and the log reports the
  number of gaps alongside the frame count.
- **The Anthropic prompt cache is now read, not only written.** The cache
  breakpoint sat at the end of the system prompt, which folds in the transient
  context a memory service recalls each turn. A cached prefix is only reused
  while it stays byte-identical, so the breakpoint moved on every request: each
  turn wrote a fresh cache entry and read none back, paying the write premium for
  no benefit. It now sits on the part of the system prompt that survives between
  turns, with the recalled context after it where it is free to vary.
  `LLMContext.SystemParts` exposes that split.
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
