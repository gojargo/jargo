---
title: Pipeline & Task
weight: 4
---

# Pipeline & Task

Three types, with a clean split of responsibility:

| Type | Responsibility |
|---|---|
| **`Pipeline`** | Links processors into a chain. Is itself a processor, so pipelines nest. |
| **`Task`** | Runs one pipeline for one session: sends `StartFrame`, injects frames, shuts down. |
| **`Runner`** | Runs a `Task` and ends it on an interrupt signal. |

## Pipeline

```go
p := pipeline.New(t.Input(), stt, agg.User(), llm, tts, t.Output(), agg.Assistant())
```

`New` links the processors in order and wraps them in a **source** and a **sink**
so frames can be fed in and observed at the edges:

```mermaid
flowchart LR
    Src(["Source"]) --> A["stt"] --> B["llm"] --> C["tts"] --> Snk(["Sink"])
    Snk -. "upstream" .-> C -. "upstream" .-> B -. "upstream" .-> A -. "upstream" .-> Src

    style Src fill:#f1f5f9,stroke:#64748b
    style Snk fill:#f1f5f9,stroke:#64748b
```

Because a `Pipeline` is a `Processor`, it can be an element of another pipeline.
Three composites build on that:

- **`pipeline.NewParallel(branches ...[]processor.Processor)`**: fan a frame out
  to several branches and merge what comes back, each branch running at its own
  pace. Useful for running two services on the same audio.
- **`pipeline.NewSyncParallel(order, branches ...[]processor.Processor)`**: fan a
  frame out the same way, but hold the output of each input frame until every
  branch has finished producing it, so what the branches produced for one input
  stays together. Pass `pipeline.FrameOrderArrival` to release frames as they
  arrive, or `pipeline.FrameOrderPipeline` to release them branch by branch when
  the order between branches matters, an image ahead of the speech describing it.
  It needs the last processor of each branch to be synchronous.
- **`pipeline.NewServiceSwitcher(services, newStrategy)`**: route frames to
  exactly one of several services, switched at runtime by pushing a
  `frames.ManuallySwitchServiceFrame`. Useful for swapping an LLM
  mid-conversation.

A parallel pipeline synchronizes the lifecycle frames (`StartFrame`, `EndFrame`
and `CancelFrame`): it pauses its own frame handling until every branch has
processed one, so a fast branch cannot start emitting before the others have
been started, or shut the pipeline down while a slower branch still has output
to flush.

## Task

```go
task := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
	Params: pipeline.Params{
	    AudioInSampleRate:  16000,   // default
	    AudioOutSampleRate: 24000,   // default
	    EnableMetrics:      true,
	    EnableUsageMetrics: true,
	},
})
err := task.Run(ctx)
```

With `EnableMetrics` set, the task sends one `MetricsFrame` once the pipeline is
ready, carrying a zeroed time to first byte and processing time for every
processor that reports metrics, so a consumer knows which processors to expect
metrics from before any have been measured. Set `SendInitialEmptyMetrics` to
`&false` to skip it. Which processors those are comes from
`pipeline.ProcessorsWithMetrics()`, which walks the chain and every nested
pipeline collecting the processors whose `CanGenerateMetrics()` reports true: the
STT, LLM, TTS and speech-to-speech services.

A `MetricsFrame` carries a list of measurements, so one frame can report several
kinds and several processors at once. Read them by switching on the type:

```go
for _, d := range mf.Data {
    switch m := d.(type) {
    case frames.TTFBMetricsData:
        log.Printf("%s took %v to answer", m.Processor, m.Value)
    case frames.LLMUsageMetricsData:
        log.Printf("%s used %d tokens", m.Processor, m.Value.TotalTokens)
    }
}
```

`Run` blocks until the pipeline finishes. What it does:

```mermaid
sequenceDiagram
    participant R as task.Run
    participant P as Pipeline
    participant S as Sink

    R->>P: Setup(ctx, Setup{Clock})
    R->>P: StartFrame
    P->>S: StartFrame
    S-->>R: startSig
    Note over R: pipeline is ready

    loop until a pipeline-ending frame arrives
        R->>P: frame from the push queue
    end

    R->>P: EndFrame
    P->>S: EndFrame
    S-->>R: endSig
    R->>P: Cleanup(context.Background())
```

Note the last step: cleanup runs on a **fresh context**, so a canceled `ctx` does
not abort goroutine shutdown.

`Run` returns only after an `EndFrame`, `StopFrame` or `CancelFrame` has traveled
the *whole* way through the pipeline, not merely been queued. That round trip is
what guarantees every processor has seen the shutdown.

### Driving a running task

```go
task.QueueFrame(frames.NewTTSSpeakFrame("Hi, how can I help?"))  // inject
task.QueueFrames([]frames.Frame{f1, f2})                          // in order

task.StopWhenDone()        // EndFrame: stop once queued frames flush
task.Cancel(ctx, "why")    // CancelFrame: stop now
task.HasFinished()         // has Run returned?
```

`Flush` is the one that is easy to miss and often exactly what you want:

```go
if err := task.Flush(ctx); err != nil { /* ctx expired */ }
```

It queues a `PipelineFlushFrame` probe and blocks until the probe has traveled
down to the sink **and back up** to the source. When it returns, every frame
queued ahead of it has been processed. Use it to let the pipeline settle (after
an interruption, say) before injecting new work.

### Observing frames

```go
worker := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
    ReachedDownstreamFilter: pipeline.AnyFrame,
    Observers: []pipeline.Observer{
        observers.NewTurnTracking(observers.TurnTrackingConfig{}),
    },
})

events.On(worker.Events(), pipeline.EventFrameReachedDownstream,
    func(ctx context.Context, f frames.Frame) { /* reached the sink */ })
events.On(worker.Events(), pipeline.EventFrameReachedUpstream,
    func(ctx context.Context, f frames.Frame) { /* reached the source */ })
```

A filter selects which frames are reported; nil reports none, which is the
default, because a handler on every frame sits on the path of everything the
pipeline does.

Both the events and the observers see frames only at the pipeline **edges**, not
between every pair of processors. Observers are notified after the callbacks. See
[Observability](../guides/observability.md).

## Worker frames: talking back to the Task

A processor deep in the chain sometimes needs to end the session: a voicemail
detector that decides to hang up, for instance. It cannot reach the `Task`
directly, so it pushes a **worker frame** and the `Task` converts it:

```mermaid
flowchart LR
    P["processor deep<br/>in the chain"] -->|"EndWorkerFrame<br/>(downstream)"| Snk(["Sink"])
    Snk -->|"fresh instance<br/>upstream"| Src(["Source"])
    Src -->|"converts to"| End["EndFrame<br/>queued on the Task"]

    style End fill:#dcfce7,stroke:#16a34a
```

| Push this | Task queues | Effect |
|---|---|---|
| `EndWorkerFrame` | `EndFrame` | Graceful: queued frames flush first. |
| `StopWorkerFrame` | `StopFrame` | Stop, leave processors running. |
| `CancelWorkerFrame` | `CancelFrame` | Immediate, no flush. |
| `InterruptionWorkerFrame` | `InterruptionFrame` | Barge-in. |

Worker frames are pushed **downstream** by default so that frames already queued
ahead of them are processed first. On reaching the sink, a *fresh* instance is
sent back upstream, a new instance rather than the original, so the two
directions never share a frame.

## Runner

```go
runner := pipeline.NewRunner()
err := runner.Run(ctx, task)
```

Runs the task and cancels it on `SIGINT`/`SIGTERM`. For a server that runs one
task per connection, call `task.Run` directly and cancel on connection close,
which is what the examples do:

```go
ctx, cancel := context.WithCancel(context.Background())
go func() { <-conn.Done(); cancel() }()
task.Run(ctx)
```

---

Next: **[Interruptions](interruptions.md)**, the mechanism that makes barge-in
work.
