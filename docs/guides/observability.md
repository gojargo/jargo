---
title: Observability
weight: 6
---

# Observability

Three layers, each answering a different question:

| Layer | Question | Package |
|---|---|---|
| **Observers** | How is the conversation behaving? | `observers/` |
| **Metrics** | How is the system performing, in aggregate? | `telemetry/metrics` |
| **Tracing** | What happened on *this* turn? | `telemetry/tracing` |

## Observers

Observers watch frames without modifying them. They are the cheapest way to get
conversation-level signal.

```go
task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
    Observers: []pipeline.Observer{
        observers.NewTurnTracking(observers.TurnTrackingConfig{
            OnTurnEnded: func(turn int, d time.Duration, interrupted bool) {
                slog.Info("turn", "n", turn, "dur", d, "interrupted", interrupted)
            },
        }),
        observers.NewUserBotLatency(observers.LatencyConfig{
            OnLatency: func(d time.Duration) {
                slog.Info("response latency", "d", d)
            },
        }),
    },
})
```

| Observer | Measures |
|---|---|
| `NewTurnTracking` | Turn count, duration, and whether each was interrupted. |
| `NewUserBotLatency` | User stopped speaking → bot started. **The number users feel**, with a per-service breakdown of where it went. |
| `NewStartupTiming` | What each processor cost to start, and how long the transport took to connect. |
| `NewDebugLog` | Frames and their contents, for working out what a pipeline is doing. |
| `NewLLMLog` | What a model was asked, what it generated, and the tools it called. |
| `NewTranscriptionLog` | What a transcriber heard, interim results included. |
| `NewMetricsLog` | Each measurement the pipeline reports, as it is reported. |

`TurnTracking` has a `TurnEndTimeout` (default 2.5s) so a brief gap between bot
utterances (an HTTP TTS boundary, a tool call) does not split one turn into two.

### Where the response latency went

`UserBotLatency` reports more than the headline figure. Alongside every latency it
reports a `LatencyBreakdown`, and it reports separately on the first thing the bot
says after a client connects, which is the greeting rather than a reply:

```go
observers.NewUserBotLatency(observers.LatencyConfig{
    OnFirstBotSpeechLatency: func(d time.Duration) {
        slog.Info("greeting latency", "d", d)
    },
    OnBreakdown: func(b observers.LatencyBreakdown) {
        for _, line := range b.ChronologicalEvents() {
            slog.Info("latency", "event", line)
        }
    },
})
```

The breakdown accounts for the user turn (the silence window, the transcriber
finalizing, any end-of-turn analyzer), each service's time to first byte, the
sentence aggregation before synthesis, and any tool calls the reply made. It is
built from the `MetricsFrame`s the services emit, so it is empty of measurements
unless `EnableMetrics` is set below.

### What startup cost

A processor does its startup work while handling the `StartFrame`: connecting,
authenticating, loading a model. `StartupTiming` times that per processor and
reports once the pipeline is up.

```go
observers.NewStartupTiming(observers.StartupTimingConfig{
    OnStartupTimingReport: func(r observers.StartupTimingReport) {
        for _, t := range r.ProcessorTimings {
            slog.Info("started", "processor", t.ProcessorName, "took", t.Duration)
        }
    },
})
```

Pass `Track` to narrow the report to the processors worth measuring; by default
it covers everything but the pipeline plumbing.

### Debugging the frame flow

`NewDebugLog` renders the fields of every frame it is given. Unfiltered that is
dozens of audio frames a second per direction, so narrow it:

```go
observers.NewDebugLog(observers.DebugLogConfig{
    Frames: []observers.DebugFrameFilter{
        // Every LLM token, wherever it travels.
        {Frame: &frames.LLMTextFrame{}},
        // User speech, but only where it reaches a streaming transcriber.
        {
            Frame:    &frames.UserStartedSpeakingFrame{},
            Match:    func(p processor.Processor) bool { _, ok := p.(*stt.StreamService); return ok },
            Endpoint: observers.DestinationEndpoint,
        },
    },
})
```

The three specialized loggers need no filter: each already reports one part of the
conversation. `LLMLog` and `TranscriptionLog` report only what actually passed
through a model or a transcriber, so the same frame types travelling elsewhere are
left alone.

### Observers see every handover

An observer is reported to on every hand-off between two processors, not only at
the ends of the pipeline, so it can tell where each frame came from. That is what
lets it distinguish a frame that has been through the output transport, and so
carries real playback timing, from the same frame earlier in the chain. The
`EventFrameReachedDownstream` and `EventFrameReachedUpstream` events are the
narrower thing: those fire only at the pipeline source and sink.

One consequence of watching everything: a turn-taking signal is **broadcast** as
two frames, one per direction. Observers count only the downstream half, using
`BroadcastSiblingID` to recognize the pair. Otherwise every turn would be counted
twice. If you write an observer that reacts to `UserStartedSpeakingFrame`,
`UserStoppedSpeakingFrame` or `InterruptionFrame`, handle that pairing.

Observers run off the frame path, each on a goroutine of its own, so a slow one
falls behind rather than holding up the conversation.

## Metrics

Enable collection on the task, or nothing is measured:

```go
pipeline.WorkerConfig{
	Params: pipeline.Params{
	    EnableMetrics:      true,   // TTFB, processing time
	    EnableUsageMetrics: true,   // tokens, TTS characters, STT audio duration
	},
}
```

The flags travel down the pipeline on the `StartFrame`, which is how each processor
learns whether to measure. They are separate because usage metrics cost money to
compute in provider round trips and TTFB does not.

`telemetry/metrics` exports through OpenTelemetry:

| Recorded | Meaning |
|---|---|
| TTFB | Time to first byte from a service. |
| TTFA | Time to first *audio*, measured from the first audible sample, not the first byte, so leading silence does not flatter the number. |
| Processing time | Wall time inside a service. |
| Tokens | LLM input/output. |
| TTS characters, STT audio seconds | Usage for cost attribution. |

```go
shutdown, err := metrics.Init(ctx, metrics.Config{ /* … */ })
defer shutdown(ctx)
```

Metrics also travel in-band as `MetricsFrame`s, which is how the RTVI processor
forwards them to a client.

## Tracing

Spans per conversation, per turn and per service call, over OpenTelemetry:

```go
shutdown, err := tracing.Init(ctx, tracing.Config{ /* … */ })
defer shutdown(ctx)

task := pipeline.NewWorker(pipe, pipeline.WorkerConfig{
	EnableTracing:  true,
	ConversationID: sessionID, // empty generates one
	// Attributes on the conversation span, which is the root of the trace — where
	// a backend's own keys go (a session id, a user id, tags).
	AdditionalSpanAttributes: []attribute.KeyValue{
		attribute.String("langfuse.session.id", sessionID),
		attribute.String("langfuse.user.id", userID),
	},
})
task.Run(ctx)
```

The session is one trace, shaped like the conversation it recorded:

```
conversation                       conversation.id, conversation.type
└── turn                           turn.number, turn.duration_seconds,
    ├── stt                        turn.was_interrupted,
    ├── llm                        turn.user_bot_latency_seconds
    └── tts
```

So a slow turn can be opened up stage by stage: STT finalize → LLM first token →
TTS first audio. That breakdown is the fastest way to find which provider is
costing you the latency, and the turn span carries the number that matters most —
`turn.user_bot_latency_seconds` — beside it.

Turn spans come from the turn tracking the task runs for them; set
`EnableTurnTracking` to `false` to trace the conversation and its service calls
without them.

Token usage lands on spans as `gen_ai.usage.*` attributes via
`tracing.SetTokenUsage`, with `SetTTSUsage` and `SetSTTUsage` alongside.

[`examples/voicebot`](../../examples/voicebot) wires tracing end to end.

## What to watch

For a voice agent, the number that matters is **user-stopped-speaking → bot
started**. Under roughly 800 ms feels conversational; past about 1.5 s people start
talking over the bot.

When it is too slow, the useful next question is which stage owns it:

- **End-of-turn detection**: often the real cost, and invisible in provider
  metrics. Check `SpeechControlParamsFrame` and see [Turn-taking](turn-taking.md).
- **LLM first token**: usually the largest single component. A smaller model or a
  shorter system prompt beats micro-optimizing elsewhere.
- **TTS first audio**: watch TTFA, not TTFB.

Interruption rate from `TurnTracking` is the other signal worth a dashboard: a rate
that climbs means turn-taking is cutting users off, or the bot is too verbose.
