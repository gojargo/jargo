# jargo — plan & roadmap

A **WebRTC-native, audio-first conversational-AI framework for Go**.

jargo lets you build a real-time voice agent: audio comes in over WebRTC, flows
through a pipeline of processors (transcription → reasoning → speech), and audio
goes back out — with natural turn-taking and barge-in.

## Vision

One opinionated path, done well:

- **Real-time voice**, not a general media framework.
- **WebRTC end to end** (Pion), because voice needs jitter buffering, loss
  concealment, and native browser/mobile mic access.
- **RTVI** on the data channel, so existing RTVI clients (web, iOS, Android)
  interoperate with a jargo server out of the box.
- **Batteries for one stack**, clean interfaces for the rest.

### Non-goals (explicitly out of scope)

- Non-WebRTC transports (WebSocket, Daily, LiveKit, telephony/SIP).
- Multimodal I/O: vision, images, video, avatars.
- LLM framework bridges (LangChain, etc.), media pipelines (GStreamer).
- Eval harness, CLI, and dev-runner tooling.
- The long tail of service integrations — addable by the community behind
  stable interfaces, not shipped by default.

## Architecture at a glance

**Principle: the engine is general, the implementation is opinionated.** The
core abstractions (`Frame`, `Transport`, `FrameProcessor`, `Pipeline`) stay
transport- and modality-agnostic. jargo ships exactly one transport (Pion
WebRTC) and one modality (audio); the opinion lives in what we ship, not in what
the architecture can express.

| Concern          | Choice                                  |
| ---------------- | --------------------------------------- |
| Transport        | WebRTC via Pion + HTTP signaling (SDP/ICE) |
| Client protocol  | RTVI over the WebRTC data channel       |
| Modality         | Audio in/out; text as the intermediate  |
| STT              | Deepgram                                |
| LLM              | Anthropic                               |
| TTS              | ElevenLabs                              |
| Memory           | mem0                                    |
| VAD              | Silero (ONNX)                           |
| Turn detection   | Smart Turn V3 (ONNX)                    |
| Audio codec      | Opus (pure Go, pion/opus)               |
| Resampling       | soxr (cgo; Phase 4)                     |

The WebRTC stack and the Opus codec are pure Go (Pion / pion/opus). cgo is
confined to what has no pure-Go equivalent yet: soxr resampling (Phase 4) and
the ONNX runtime for VAD/turn detection (Phase 5).

## Target package layout

```
github.com/gojargo/jargo
├── frames/                 # Frame + categories (system/data/control)   [done]
├── clock/                  # pipeline clock for timing                  [done]
├── processor/              # FrameProcessor base + linking/queueing      [done]
├── pipeline/               # Pipeline, Task, Runner                      [done]
├── transport/              # general Transport / Input / Output processors    [done]
│   └── pionrtc/            # Pion WebRTC impl: signaling, tracks               [done]
├── rtvi/                   # RTVI message types + processor                  [done]
├── audio/
│   ├── opus/               # Opus encode/decode (pure Go, pion/opus)          [done]
│   ├── resample/           # linear resampler (pure Go)                       [done]
│   ├── vad/                # Silero VAD (ONNX)                                [done]
│   └── turn/               # Smart Turn V3 (ONNX) + Whisper log-mel features  [done]
├── turntaking/             # VAD + turn processor (speaking/interruption)     [done]
├── internal/onnxrt/        # ONNX runtime boundary (cgo)                      [done]
├── service/
│   ├── deepgram/           # streaming STT (WebSocket)                        [done]
│   ├── anthropic/          # streaming LLM (official SDK, Haiku)              [done]
│   ├── elevenlabs/         # streaming TTS (HTTP)                             [done]
│   └── mem0/               # memory
├── aggregators/            # LLM context (user + assistant)                   [done]
├── metrics/                # hooks only (impls deferred)
└── examples/               # echo + voicebot                                 [done]
```

Names are provisional and may shift as packages land.

## Roadmap

Each phase is a reviewable increment with a concrete milestone. Build order
moves from the engine outward, then lights up one full voice round-trip before
adding turn-taking and memory.

### Phase 0 — Frames (complete)

- [x] Base `Frame` interface, `BaseFrame`, the three categories, the
      `Uninterruptible` marker, concurrency-safe id/name counters.
- [x] Concrete audio/text/control frames needed downstream:
  - Audio: `InputAudioRawFrame`, `OutputAudioRawFrame`, `TTSAudioRawFrame`.
  - Text: `TextFrame`, `TranscriptionFrame`, `InterimTranscriptionFrame`,
    `LLMTextFrame`, `LLMFullResponseStart/EndFrame`.
  - System/control: `StartFrame`, `EndFrame`, `CancelFrame`, `ErrorFrame`,
    `InterruptionFrame`, `UserStarted/StoppedSpeakingFrame`,
    `BotStarted/StoppedSpeakingFrame`, `TTSStarted/StoppedFrame`.

**Milestone:** the frame vocabulary a voice pipeline needs, fully typed.

### Phase 1 — Engine (processor + pipeline) (complete)

- [x] `processor.Processor` + `Base`: link up/downstream, push frames both ways,
      a system-priority input goroutine, an in-order process goroutine, and
      interruption via context cancellation (uninterruptible frames preserved).
- [x] `Pipeline`: linear chain wrapped by a source and a sink; nests as a
      processor.
- [x] `Task` + `Runner`: lifecycle around `StartFrame` / `EndFrame` /
      `CancelFrame`; the runner ends the task on SIGINT/SIGTERM.
- [x] `clock` package: `Clock` interface + monotonic `System` clock.

Port decisions (Go idiom, confirmed): concrete processors embed `*Base` and pass
`self` so the base can dispatch to the overridden `ProcessFrame` (no inheritance/
`super()`); `ProcessFrame` returns an `error` that the base turns into an
upstream `ErrorFrame`; interruption is cooperative `context` cancellation, since
a goroutine can't be force-stopped. Deferred to later phases: metrics hooks,
observers, event handlers, pause/resume frames.

**Milestone:** frames flow end to end through a chain of processors; an echo
processor round-trips a frame, covered by tests.

### Phase 2 — WebRTC transport (Pion) + audio I/O (complete)

- [x] `transport` package: `Params`, `Transport` interface, `BaseInput` and
      `BaseOutput` processors (audio-only port of Pipecat's base transports).
- [x] `transport/pionrtc`: Pion peer connection + SDP offer/answer signaling
      (non-trickle ICE), incoming/outgoing Opus tracks.
- [x] `audio/opus`: pure-Go Opus via `pion/opus` — decode (SILK/CELT/hybrid) and
      encode (CELT-only, 48 kHz, 20 ms).
- [x] Input: RTP Opus → decode → PCM → `InputAudioRawFrame`.
- [x] Output: `OutputAudioRawFrame` → chunk → encode Opus → RTP track.
- [x] `examples/echo`: HTTP signaling server + browser page; a loopback
      integration test (two in-process Pion peers) proves the round-trip.

Decisions: the codec is **pure Go (`pion/opus`)** at the user's direction — no
cgo. The encoder is unreleased (`pion/opus@main`, CELT-only; SILK pending
upstream) and pinned to a commit; revisit for quality/SILK or swap to cgo
libopus if needed. The echo path runs at **48 kHz mono end to end**, so **soxr
resampling is deferred to Phase 4** (needed once STT @16k / TTS @24k land). Audio
output is paced by the input; an explicit pacer/auto-silence is deferred.

**Milestone:** an echo bot over WebRTC — speak into a browser, hear yourself
back, audio crossing the full transport.

### Phase 3 — RTVI on the data channel (complete)

- [x] `rtvi` package: JSON message types (envelope `{label:"rtvi-ai", type, id,
      data}`, protocol `2.0.0`) and an `rtvi.Processor`.
- [x] Handshake: `client-ready` → `bot-ready`.
- [x] Frame → message reporting: transcriptions, user/bot speaking, errors,
      LLM text (the real transcripts arrive with STT in Phase 4).
- [x] Wired over the Pion data channel: `InputTransportMessageFrame` /
      `OutputTransportMessageFrame` carry messages in and out; the `Connection`
      gained data-channel send/receive.
- [x] Loopback test (real data channel: `client-ready` → `bot-ready`) and the
      echo example's browser page acts as a minimal RTVI client.

Decision: jargo **consolidates Pipecat's `RTVIProcessor` + `RTVIObserver` into a
single `FrameProcessor`** — it both handles client messages and converts
pipeline frames to RTVI messages — since the observer infra was deferred and the
processor is already in the pipeline.

**Milestone:** an existing RTVI `small-webrtc` client connects, receives
bot-ready, and renders live transcripts.

### Phase 4 — First conversational vertical (STT → LLM → TTS) (complete)

- [x] `audio/resample`: pure-Go stateful linear resampler (e.g. 24 kHz TTS →
      48 kHz output), wired into `BaseOutput`.
- [x] `service/deepgram`: streaming STT over Deepgram's WebSocket (linear16,
      interim results, endpointing → finalized `TranscriptionFrame`).
- [x] `service/anthropic`: streaming LLM via the official Go SDK (Claude Haiku
      default, system-prompt caching) consuming `LLMContextFrame`.
- [x] `service/elevenlabs`: streaming TTS over the HTTP streaming endpoint, with
      sentence aggregation; emits `TTSAudioRawFrame`s.
- [x] `aggregators`: shared `LLMContext`, user aggregator (transcription → run
      LLM) and assistant aggregator (response → context).
- [x] `examples/voicebot`: the full pipeline + an RTVI browser client.

Decisions: LLM is **Claude Haiku** (latency) via the official SDK; STT and the
WebSocket use **`coder/websocket`**; TTS uses the **HTTP streaming** endpoint
(simpler than the WS for whole-sentence input) at **`pcm_24000`**, resampled to
48 kHz — keeping Phase 4 **cgo-free** (pure-Go resampler). Turn-taking relies on
**Deepgram endpointing** (`speech_final`) for now; Silero VAD + Smart Turn land
in Phase 5. The three service integrations are verified live (API keys), not by
unit tests; the resampler and aggregators are unit-tested. Also fixed a
pre-existing data race in the Pion input transport's read-cancel handling.

**Milestone:** full voice round-trip — speak, get transcribed, reasoned over,
and answered out loud.

### Phase 5 — Turn-taking (VAD + Smart Turn) (complete)

- [x] `internal/onnxrt`: the single cgo boundary — locates `libonnxruntime`
      (via `JARGO_ONNXRUNTIME_LIB`), initializes the runtime once, and creates
      model sessions (`github.com/yalue/onnxruntime_go`).
- [x] `audio/vad`: a confidence-gated state machine (quiet → starting →
      speaking → stopping) plus `Silero`, the embedded Silero VAD ONNX model
      (caller-managed context + recurrent state).
- [x] `audio/turn`: `SmartTurnV3`, the embedded smart-turn-v3 ONNX model, with
      a pure-Go Whisper log-mel feature extractor (DFT + Slaney mel filterbank;
      a constant-frame fast path keeps the silence-padded window cheap).
- [x] `turntaking`: one processor after the input transport that drives both
      analyzers, emitting `UserStartedSpeakingFrame` + `InterruptionFrame` on a
      barge-in and `UserStoppedSpeakingFrame` on a real end-of-turn; a logical
      turn spans pauses the turn model rates incomplete.
- [x] Aggregator gating (`aggregators.WithTurnTaking`): the LLM runs when an
      end-of-turn is reported and a finalized transcript is in hand, rather than
      on STT endpointing alone; the assistant aggregator commits a partial reply
      on interruption.

Decisions: the models are **embedded with `go:embed`** (Silero ~2 MB, Smart
Turn ~8.7 MB); the ONNX runtime is **operator-supplied** at a path, not bundled.
VAD gating is **confidence-only** — jargo trusts the neural model rather than
porting Pipecat's EBU R128 volume gate. jargo ships **one opinionated path**
(Silero + Smart Turn folded into a single `turntaking` processor) rather than
porting upstream Pipecat's newer pluggable `turns/` strategy framework, which is
out of scope. Feature extraction and both models are validated to within 1e-3
of the reference Python implementation by unit tests (the model tests skip when
the runtime is not configured). Turn analysis runs synchronously on the audio
goroutine (~30 ms per turn-end); offloading is a future optimization.

**Milestone:** natural turn-taking — the bot waits for end-of-turn and can be
interrupted mid-sentence.

### Phase 6 — Memory + polish

- mem0 memory service.
- LLM context summarization for long sessions.
- Flesh out metrics hooks; examples and docs.

**Milestone:** a complete, documented reference voice agent.

## Dependencies (anticipated)

- `github.com/pion/webrtc/v4` — WebRTC stack (pure Go). [in use]
- `github.com/pion/opus` — Opus codec, pure Go. [in use; encoder unreleased,
  pinned to a `main` commit, CELT-only]
- ONNX runtime — `onnxruntime-go` (cgo) for VAD + Smart Turn.
- soxr — audio resampling.
- Provider clients — Anthropic (official Go SDK), Deepgram, ElevenLabs, mem0
  (HTTP/WebSocket; official SDK where one exists, otherwise thin clients).

## Open technical decisions

- **Opus encoder quality** — `pion/opus`'s encoder is CELT-only and unreleased;
  evaluate voice quality and revisit (await upstream SILK, or fall back to cgo
  libopus) if it falls short.
- **ONNX runtime distribution** — *resolved:* models are embedded with
  `go:embed`; the `libonnxruntime` shared library is operator-supplied via the
  `JARGO_ONNXRUNTIME_LIB` env var (`internal/onnxrt`).
- **RTVI schema fidelity** — match the message schema closely enough that
  upstream RTVI clients interoperate unchanged.
- **Audio frame sizing** — sample rate, frame duration, and resampling points
  between WebRTC (Opus), VAD/turn (PCM), and the STT/TTS services.

## Conventions

- Comments describe the Go code on its own terms.
- Attribution lives in `LICENSE` and `NOTICE` only; no per-file headers.
- Concurrency-safe by construction — pipelines are genuinely concurrent.
