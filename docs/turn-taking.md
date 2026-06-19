# Turn-taking: VAD + Smart Turn

jargo does natural turn-taking: the bot waits for a real end-of-turn instead of
any pause, and the user can interrupt it mid-sentence (barge-in). Two local ONNX
models drive this:

- **[Silero VAD](https://github.com/snakers4/silero-vad)** — voice activity
  detection: when is the user speaking at all.
- **[Smart Turn v3](https://github.com/pipecat-ai/smart-turn)** — end-of-turn
  detection: has the user actually finished, or just paused mid-thought.

Both models are **embedded in the binary** (`go:embed`), so there is nothing to
download or locate at run time except the ONNX Runtime itself.

## ONNX Runtime setup

The models run on the [ONNX Runtime](https://onnxruntime.ai/), the one part of
jargo that uses cgo. The runtime shared library is **not** bundled; download a
build for your platform from the
[releases page](https://github.com/microsoft/onnxruntime/releases) and point
jargo at it:

```sh
# Linux
export JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so
# macOS
export JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib
# Windows
set JARGO_ONNXRUNTIME_LIB=C:\path\to\onnxruntime.dll
```

If `JARGO_ONNXRUNTIME_LIB` is unset, jargo looks for the library by its
conventional name on the loader's default search path
(`libonnxruntime.so`/`.dylib`/`onnxruntime.dll`). When the runtime cannot be
loaded, the voice bot still runs — it falls back to STT endpointing for
turn-taking and loses barge-in.

## How it fits the pipeline

A single `turntaking.Detector` processor sits just after the input transport. It
runs both analyzers on the incoming audio (resampled to 16 kHz mono) and emits:

- `UserStartedSpeakingFrame` + `InterruptionFrame` when a turn begins — the
  interruption flushes any in-progress bot response so the user can barge in.
- `UserStoppedSpeakingFrame` when the turn is actually complete.

A *logical turn* spans pauses that Smart Turn rates as incomplete, so a pause
mid-sentence does not end the turn.

```go
v, _ := vad.NewSilero()
tr, _ := turn.NewSmartTurnV3()
detector := turntaking.New(turntaking.Config{VAD: v, Turn: tr})

pipe := pipeline.New(
    t.Input(),
    detector,
    stt,
    agg.User(),   // built with aggregators.WithTurnTaking()
    llm,
    tts,
    rtvi.NewProcessor(),
    t.Output(),
    agg.Assistant(),
)
```

With `aggregators.WithTurnTaking()`, the LLM runs when the turn is reported
complete *and* a finalized transcript is in hand — so Smart Turn, not STT
endpointing, decides when the bot responds. See
[`examples/voicebot`](../examples/voicebot) for the full wiring.

## Implementation notes

- VAD gating is **confidence-only**: jargo trusts Silero's neural confidence
  rather than adding a separate volume threshold.
- Feature extraction for Smart Turn is a pure-Go reimplementation of Whisper's
  log-mel features; it and both models are validated to within `1e-3` of the
  reference Python implementation by unit tests.
- Turn analysis currently runs synchronously on the audio goroutine
  (~tens of ms per end-of-turn). See [benchmarks](benchmarks.md) for the
  performance picture and the planned FFT optimization.
