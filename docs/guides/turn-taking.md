---
title: Turn-taking
weight: 1
---

# Turn-taking: VAD + Smart Turn

jargo does natural turn-taking: the bot waits for a real end-of-turn instead of
any pause, and the user can interrupt it mid-sentence (barge-in). Two local ONNX
models drive this:

- **[Silero VAD](https://github.com/snakers4/silero-vad)** for voice activity
  detection: is the user speaking at all.
- **[Smart Turn v3](https://github.com/pipecat-ai/smart-turn)** for end-of-turn
  detection: has the user actually finished, or just paused mid-thought.

Both models are **embedded in the binary** (`go:embed`), so there is nothing to
download or locate at run time except the ONNX Runtime itself.

## ONNX Runtime setup

The models run on the [ONNX Runtime](https://onnxruntime.ai/), bound through
[purego](https://github.com/ebitengine/purego) and loaded at run time, so it
needs no C toolchain at build time, and the default `CGO_ENABLED=0` build works.
The runtime shared library is **not** bundled; download a
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
loaded, the voice bot still runs. It falls back to STT endpointing for
turn-taking and loses barge-in.

## How it fits the pipeline

Turn-taking is a small subsystem split across two processors:

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
stop, so a pause Smart Turn rates incomplete does not end the turn. Leaving the
chains empty builds that default, which loads the model: a pipeline that cannot
find the ONNX runtime fails to start with the reason (see
[`make deps-onnx`](../../Makefile)). For turn-taking without a model, build the
stop chain explicitly with `turns.NewSpeechTimeoutStop`, which ends the turn on
silence timers alone.

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
    rtvi.NewProcessor(),
    t.Input(),
    vadProc,
    stt,
    turnsProc,
    agg.User(), // built with aggregators.WithTurnTaking()
    llm,
    tts,
    t.Output(),
    agg.Assistant(),
)
```

With `aggregators.WithTurnTaking()`, the LLM runs when the turn is reported
complete *and* a finalized transcript is in hand, so Smart Turn, not STT
endpointing, decides when the bot responds. See
[`examples/voicebot`](../../examples/voicebot) for the full wiring, and
[Interruptions](../concepts/interruptions.md) for what barge-in does to the
pipeline.

### Shared VAD across concurrent streams

In servers handling many simultaneous calls, creating a dedicated VAD session
per stream incurs the ONNX runtime's own session overhead (~10 MB RSS per
stream) and adds setup latency. Callers can instead load a single
`vad.SileroModel` and share it across streams:

```go
// Load the shared Silero model once (e.g. at server startup).
sileroModel, err := vad.NewSileroModel()
if err != nil {
    // handle error
}
defer sileroModel.Close()

// Create stream analyzers reusing the shared model session.
vd, err := vad.NewSilero(vad.WithModel(sileroModel))
```

- **Lifecycle:** A `SileroModel` must outlive every analyzer built from it.
  Closing the model while analyzers are active will cause subsequent inferences
  to fail.
- **Sequential inference:** Inferences on a shared model run sequentially under
  the ONNX runtime session's internal mutex (about 75 to 110 µs per 32 ms
  frame).

### Strategies, idle, mute, and LLM completion

The start/stop chains are pluggable (`turns.StartStrategy` / `turns.StopStrategy`):
VAD, transcription, min-words and wake-phrase starts; Smart-Turn, speech-timeout,
external and eager stops. Smart Turn runs locally by default
(`turn.NewSmartTurnV3`); `turn.NewHTTPSmartTurn` asks a remote service for the
same verdict instead, for a deployment that scores turns somewhere other than in
the bot. Mute strategies (`turns.NewAlwaysUserMute`, …) suppress user input
while the bot speaks or a tool call runs.

Every strategy sees every frame and reports its own decision (`ReportStarted`,
`ReportStopped`); the turn controller combines them. `turns.Config.StrategiesParams`
says how:

- `StartMode` and `StopMode` take a `turns.StrategyMode`. With
  `StrategyModeFirst` the first strategy to decide wins; with `StrategyModeAll`
  every strategy has to agree. The start list defaults to first, so speech or a
  transcript opens a turn on its own, and the stop list to all, so a turn ends
  only once every stop strategy says the user is done. New speech clears the
  stop decisions made so far.
- `StartMergeMode` and `StopMergeMode` take a `turns.MergeMode`. A service with
  its own turn detection can recommend strategies; with `MergeModeReplace`, the
  default, the application's lists are used instead, and with `MergeModeExtend`
  they are added to the service's (or to the defaults, while no service has
  recommended any).

The controller runs the turn's inference itself, once every stop strategy that
listens to the user has decided. A stop strategy that decides from the answer
instead, such as one reading the LLM's turn-completion marker, returns true from
`WaitsForInference` and decides on that inference. If the user speaks again
after the answer started but before the turn ended, the answer is void: the
user aggregator interrupts the bot, or, for a turn that does not interrupt,
cancels a speculative answer with a `UserTurnResponseCancelFrame`. The stop
timeout (`Config.StopTimeout`) ends a turn left open without user activity,
unless a stop strategy has a decision of its own scheduled
(`DecisionScheduled`).

`turns.FilterIncompleteUserTurnStrategies` adds the optional LLM turn-completion
gate. The model starts every response with a marker (● complete, ◐ the user will
go on shortly, ○ the user needs more time), and the service reports each one as
a `UserTurnCompletionFrame`. `LLMTurnCompletionStop` turns the markers on in the
LLM when the pipeline starts and ends the turn on ●, a tool call, or a response
with no marker; ◐ and ○ keep the turn open, and a user who then stays quiet for
`IncompleteShortTimeout` or `IncompleteLongTimeout` is re-prompted. It works
next to any other stop strategy, an eager end of turn included, and on top of a
service's own turn detection with `StopMergeMode: turns.MergeModeExtend`:

```go
aggregators.WithTurns(turns.Config{
    Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{}),
    StrategiesParams: turns.UserTurnStrategiesParams{
        StopMergeMode: turns.MergeModeExtend,
    },
})
```

For an LLM that does not follow the marker instructions well,
`turns.ClassifierUserTurnStrategies` asks a [classifier](../../classifier) instead:
at each pause with new words, `turns.ClassifierStop` asks whether the turn so far
is complete, cut short, or one where the user needs time, and holds the turn open
for `ShortTimeout` or `LongTimeout` on the last two. It runs after Smart Turn by
default, and the turn ends once both have decided.

`turns.NewWakePhraseStart` starts turns on its own once it hears the wake
phrase, so use it as the only start strategy. To combine it with another start
condition, such as `turns.NewMinWordsStart`, put both in a `StrategyModeAll`
list.

### Turns with nothing transcribed

A turn can start on voice activity alone and end with no transcript: a cough,
background noise, or speech the STT could not make out. Nothing is written to
the conversation and the LLM does not run. `turns.Config.EmptyUserTurn` decides
what happens then, and the two cases differ in what silence costs:

- **The turn interrupted the bot**, while it was thinking, speaking or running a
  tool (a turn before the bot has first finished speaking counts too). Left
  unanswered, the bot would stay silent mid-response, so by default the
  aggregator adds a developer message (`turns.DefaultEmptyUserTurnInterruptedPrompt`)
  and runs the LLM once: the bot asks the user to repeat, or picks up where it
  left off.
- **The bot was waiting for the user.** The conversation is not stuck, and
  answering what may be noise would be intrusive, so the turn gets no answer
  unless `IdlePrompt` is set. The idle timer starts again, as it would have had
  the turn not happened.

`MaxConsecutiveRecoveries` (1 by default) bounds how many empty turns in a row
are answered; a transcribed turn resets the count. An empty `InterruptedPrompt`
turns the interrupted case off. The recovery is off with a realtime service,
which hears the user's audio itself.

```go
idle := "The user may have said something, but it was not recognized. Briefly ask them to repeat it."
aggregators.WithTurns(turns.Config{
    EmptyUserTurn: &turns.EmptyUserTurnConfig{IdlePrompt: idle},
})
```

## Implementation notes

- The embedded Silero model is **v6.2**. Its recurrent state is reset about
  every two seconds of audio, but only during clear non-speech: a reset waits
  until the VAD is quiet and the model has scored a short run of windows as
  clearly non-speech. Long silence, noise or music would otherwise shift the
  state until speech that follows goes undetected, and a reset during speech
  could end the user's turn early. Resets are timed by the audio, not the wall
  clock.
- VAD gating is **confidence-only**: jargo trusts Silero's neural confidence
  rather than adding a separate volume threshold.
- Feature extraction for Smart Turn is a pure-Go reimplementation of Whisper's
  log-mel features; it and both models are validated to within `1e-3` of the
  reference Python implementation by unit tests.
- Smart Turn runs inside the `TurnAnalyzerStop` strategy, driven by the turn
  controller (~tens of ms per end-of-turn). See [benchmarks](https://github.com/gojargo/jargo-benchmarks)
  for the performance picture and the planned FFT optimization.
