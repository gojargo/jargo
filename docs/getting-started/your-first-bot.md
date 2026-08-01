---
title: Your first bot
weight: 3
---

# Your first bot

The complete voice agent is about fifteen lines. This page builds it up one piece
at a time, explaining each. The finished result is
[`examples/voice/openai`](../../examples/voice/openai).

## 1. The services

Each service is a plain `Config` struct and a constructor. No environment
variables are read by the library; that is your app's job.

```go
key := os.Getenv("OPENAI_API_KEY")

stt := chat.NewSTT(chat.STTConfig{APIKey: key, SampleRate: opus.SampleRate})
llm := chat.NewLLM(chat.LLMConfig{APIKey: key})
tts := chat.NewTTS(chat.TTSConfig{APIKey: key})
```

Any of the three can be swapped for a different provider independently. They are
all just processors behind a small interface. See
**[Services](../guides/services.md)**.

## 2. The transport

```go
params := transport.DefaultParams()
params.AudioInSampleRate = opus.SampleRate
params.AudioOutSampleRate = opus.SampleRate

t := rtc.NewTransport(conn, params)
```

`t.Input()` and `t.Output()` are the processors that sit at the head and tail of
the pipeline. `conn` is a `*rtc.Connection` you have already answered a WebRTC
offer on.

## 3. The conversation context

```go
convo := frames.NewLLMContext("You are a helpful voice assistant. Keep replies short.")
agg := aggregators.New(convo)
```

`agg.User()` collects transcriptions into a user message and triggers the LLM;
`agg.Assistant()` records what the bot actually said. They share `convo`, which is
how the conversation accumulates across turns. See
**[LLM context](../concepts/llm-context.md)**.

> Keep replies short is not decoration. A model that writes paragraphs sounds
> terrible out loud and costs you latency on every turn.

## 4. Turn-taking

Optional, but it is the difference between a demo and something usable:

```go
vd, _ := vad.NewSilero()
tr, _ := turn.NewSmartTurnV3()

vadProc := vadproc.New(vadproc.Config{VAD: vd})
turnsProc := turns.NewUserTurnProcessor(turns.Config{
    Strategies: turns.UserTurnStrategies{
        Start: turns.DefaultStartStrategies(),
        Stop: []turns.StopStrategy{
            turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr}),
        },
    },
})
```

Both constructors need the ONNX Runtime and return an error without it. Handle
that by running without turn-taking rather than failing. The bot still works; it
just falls back to STT endpointing and loses barge-in.

When you add these, tell the aggregator to wait for them:

```go
agg := aggregators.New(convo, aggregators.WithTurnTaking())
```

## 5. Assemble

Order matters. This is the reference wiring:

```go
task := pipeline.NewTask(pipeline.New(
    t.Input(),          // audio in
    vadProc,            // is the user speaking?
    stt,                // audio → text
    turnsProc,          // has the user finished?
    agg.User(),         // text → context, triggers the LLM
    llm,                // context → response tokens
    tts,                // tokens → audio
    rtvi.NewProcessor(), // client events over the data channel
    t.Output(),         // audio out
    agg.Assistant(),    // record what was actually said
), pipeline.TaskParams{
    AudioInSampleRate:  opus.SampleRate,
    AudioOutSampleRate: opus.SampleRate,
    EnableMetrics:      true,
    EnableUsageMetrics: true,
})
```

Two placements are easy to get wrong:

- **`agg.Assistant()` goes last, after the output transport.** Put it earlier and
  the context will record sentences the user never heard, because an interruption
  truncates the audio but not the text.
- **`turnsProc` goes after `stt`, not next to `vadProc`.** It needs
  transcriptions to decide a turn is over. It reaches the processors behind it by
  pushing frames upstream.

## 6. Run

```go
ctx, cancel := context.WithCancel(context.Background())
go func() { <-conn.Done(); cancel() }()

if err := task.Run(ctx); err != nil {
    slog.Error("task", "err", err)
}
```

`Run` blocks until the pipeline finishes. One task per connection: build the
whole thing inside your `/offer` handler and let it die with the connection.

## Make it speak first

By default the bot waits for the user. To greet on connect, queue a run of the
current context:

```go
task.QueueFrame(frames.NewLLMRunFrame())
```

Or say something fixed, bypassing the LLM entirely:

```go
task.QueueFrame(frames.NewTTSSpeakFrame("Hi, what can I help with?"))
```

## Where to go next

- **[Architecture](../concepts/architecture.md)**: the model behind what you just wired.
- **[Turn-taking](../guides/turn-taking.md)**: tuning how it feels.
- **[Writing a processor](../extending/custom-processor.md)**: adding your own logic to the chain.
