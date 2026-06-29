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

Turn-taking is a small subsystem (ported from Pipecat's `turns/`) split across
two processors:

- A `vadproc.Processor` just after the input transport runs the VAD on incoming
  audio (resampled to 16 kHz mono) and emits `VADUserStartedSpeakingFrame` /
  `VADUserStoppedSpeakingFrame` plus a periodic `UserSpeakingFrame`.
- A `turns.UserTurnProcessor` after the STT consumes those VAD frames and the
  transcripts, runs pluggable **start** and **stop** strategies, and emits the
  turn decisions: `UserStartedSpeakingFrame` + `InterruptionFrame` to open a turn
  (the interruption flushes in-progress bot audio for barge-in), and
  `UserStoppedSpeakingFrame` to close it. It also owns a `UserIdleController`
  (re-engage a silent user) and optional mute strategies.

The default strategies are VAD-or-transcription to start and Smart Turn v3 to
stop, so a pause Smart Turn rates incomplete does not end the turn.

```go
vd, _ := vad.NewSilero()
tr, _ := turn.NewSmartTurnV3()

vadProc := vadproc.New(vadproc.Config{VAD: vd})
turnsProc := turns.NewUserTurnProcessor(turns.Config{
    Strategies: turns.UserTurnStrategies{
        Start: turns.DefaultStartStrategies(), // VAD + transcription
        Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr})},
    },
    // Optional: re-engage a caller who goes silent.
    IdleTimeout: 10 * time.Second,
    OnIdle: func(ctx context.Context, c *turns.UserIdleController) error {
        return c.Push(ctx, frames.NewTTSSpeakFrame("Are you still there?"), processor.Downstream)
    },
})

pipe := pipeline.New(
    t.Input(),
    vadProc,
    stt,
    turnsProc,
    agg.User(), // built with aggregators.WithTurnTaking()
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

### Strategies, idle, mute, and LLM completion

The start/stop chains are pluggable (`turns.StartStrategy` / `turns.StopStrategy`):
VAD, transcription, min-words and wake-phrase starts; Smart-Turn, speech-timeout
and external stops, plus a `deferred` wrapper. `turns.FilterIncompleteUserTurnStrategies`
together with a `turns.CompletionFilter` placed after the LLM add the optional
✓/○/◐ LLM turn-completion gate, where the model itself judges whether the user's
turn is semantically complete (prepend `turns.CompletionInstructions` to the
system prompt). Mute strategies (`turns.NewAlwaysUserMute`, …) suppress user
input while the bot speaks or a tool call runs.

## Implementation notes

- VAD gating is **confidence-only**: jargo trusts Silero's neural confidence
  rather than adding a separate volume threshold.
- Feature extraction for Smart Turn is a pure-Go reimplementation of Whisper's
  log-mel features; it and both models are validated to within `1e-3` of the
  reference Python implementation by unit tests.
- Smart Turn runs inside the `TurnAnalyzerStop` strategy, driven by the turn
  controller (~tens of ms per end-of-turn). See [benchmarks](https://github.com/gojargo/jargo-benchmarks)
  for the performance picture and the planned FFT optimization.
