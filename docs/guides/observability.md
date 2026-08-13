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

Observers watch frames at the pipeline edges without modifying them. They are the
cheapest way to get conversation-level signal.

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
| `NewUserBotLatency` | User stopped speaking → bot started. **The number users feel.** |
| `NewStartupTiming` | Pipeline start → first bot audio. Cold-start latency. |
| `NewLogger` | Every frame, for debugging frame flow. |

`TurnTracking` has a `TurnEndTimeout` (default 2.5s) so a brief gap between bot
utterances (an HTTP TTS boundary, a tool call) does not split one turn into two.

`NewLogger` takes a `Filter`, which is what makes it usable on a real pipeline:

```go
observers.NewLogger(observers.LoggerConfig{
    Filter: func(f frames.Frame) bool {
        _, isAudio := f.(*frames.OutputAudioRawFrame)
        return !isAudio      // everything except the audio firehose
    },
})
```

Unfiltered, it logs every audio frame: dozens per second per direction.

### Observers see only the edges

Both `Observers` and the `EventFrameReachedDownstream` /
`EventFrameReachedUpstream` events fire at the pipeline **source and sink**, not
between every pair of processors. To observe mid-chain, insert a processor.

One consequence: a turn-taking signal is **broadcast** as two frames, one per
direction. Observers count only the downstream half, using `BroadcastSiblingID` to
recognize the pair. Otherwise every turn would be counted twice. If you write an
observer that reacts to `UserStartedSpeakingFrame`,
`UserStoppedSpeakingFrame` or `InterruptionFrame`, handle that pairing.

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
