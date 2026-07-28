---
title: Telephony
weight: 4
---

# Telephony

A phone agent is the same pipeline over a different transport. Instead of WebRTC,
audio arrives as µ-law 8 kHz over a WebSocket from a phone provider.

## The transport

`transport/wsserver` serves the WebSocket endpoint the provider streams to. The
wire format is provider-specific and supplied as a **`Serializer`**, so the
transport itself stays provider-agnostic:

```go
ser := twilio.New(twilio.Config{ /* … */ })

params := transport.DefaultParams()
params.AudioInSampleRate = 8000
params.AudioOutSampleRate = 8000

t, err := wsserver.Accept(w, r, ser, params)
```

| Provider | Package |
|---|---|
| Twilio Media Streams | `transport/wsserver/twilio` |
| Telnyx Media Streaming | `transport/wsserver/telnyx` |
| Plivo Audio Streaming | `transport/wsserver/plivo` |
| Exotel Media Streaming | `transport/wsserver/exotel` |

Each is `New(Config)` returning a `*Serializer`. One serializer serves **one
session**: build it per call, not once at startup; it is not safe to share.

## Run the pipeline at 8 kHz

This is the part that bites people. Telephony audio is µ-law 8 kHz, and the rate
has to be set in **three** places consistently:

```go
const phoneSampleRate = 8000

params.AudioInSampleRate = phoneSampleRate
params.AudioOutSampleRate = phoneSampleRate

tts := elevenlabs.NewTTS(elevenlabs.Config{
    APIKey:     key,
    SampleRate: phoneSampleRate,   // ask the provider for 8 kHz directly
})

task := pipeline.NewTask(pipeline.New(procs...), pipeline.TaskParams{
    AudioInSampleRate:  phoneSampleRate,
    AudioOutSampleRate: phoneSampleRate,
})
```

Asking the TTS provider for 8 kHz avoids synthesizing at 24 kHz and downsampling,
which costs both quality and latency.

## The pipeline

Identical to the WebRTC one, minus the RTVI processor, since there is no data channel
on a phone call:

```go
pipeline.New(
    t.Input(), vadProc, stt, turnsProc,
    agg.User(), llm, tts, t.Output(), agg.Assistant(),
)
```

See [`examples/twiliobot`](../../examples/twiliobot) for the complete server.

## The idle watchdog

Phone calls need this more than browser sessions do: a caller who wanders off
leaves the line open and the meter running. Configure it on the turn processor:

```go
turnsProc := turns.NewUserTurnProcessor(turns.Config{
    Strategies:  strategies,
    IdleTimeout: 10 * time.Second,
    OnIdle: func(ctx context.Context, c *turns.UserIdleController) error {
        return c.Push(ctx, frames.NewTTSSpeakFrame("Are you still there?"),
            processor.Downstream)
    },
})
```

`OnIdle` fires each time the timeout elapses, so escalate and eventually hang up
rather than asking forever. To end the call from inside the callback, push an
`EndWorkerFrame`, the mechanism a processor uses to reach the `Task`:

```go
return c.Push(ctx, frames.NewEndWorkerFrame(), processor.Downstream)
```

Retune the timeout mid-call with `UserIdleTimeoutUpdateFrame`: shorter while
waiting for a yes/no, longer while the caller reads out a number.

## DTMF

Keypresses arrive as `InputDTMFFrame` (a system frame, so they are never dropped
by a barge-in). `processor/dtmf` aggregates digits into complete entries, so an
account number typed at speed arrives as one value rather than eight frames.

Play tones outbound with `OutputDTMFFrame`.

For menu navigation, `processor/ivr` handles the traversal, and
`processor/voicemail` detects an answering machine so the bot does not hold a
conversation with a recording.

## Practical notes

- **8 kHz µ-law hurts STT accuracy.** Expect a real drop versus wideband audio and
  budget for it in prompts. Confirm important values back to the caller.
- **Turn-taking matters more on the phone.** There is no video, no visual
  backchannel, and callers expect the rhythm of a phone conversation. Tune it.
- **Recording is usually regulated.** Consent requirements vary by jurisdiction.
