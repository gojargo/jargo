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

params := wsserver.DefaultParams()
params.AudioInSampleRate = 16000
params.AudioOutSampleRate = 16000

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

### Origins

`Params.AllowedOrigins` names the origins a browser may open the socket from.
Empty, the default, allows every origin, which is what a phone provider needs: it
is not a browser and sends no `Origin` header at all.

Set it whenever the endpoint is one a browser connects to (the `rtviws`
serializer, say). Without it, a page on any other site can open the socket in a
visitor's browser and hold a conversation as them. `Accept` refuses a request
whose origin is not listed, and one carrying no origin at all, with
`wsserver.ErrOriginNotAllowed` and no reply written, leaving the endpoint to
choose what to tell it:

```go
params := wsserver.DefaultParams()
params.AllowedOrigins = []string{"https://app.example"}

t, err := wsserver.Accept(w, r, ser, params)
if errors.Is(err, wsserver.ErrOriginNotAllowed) {
    http.Error(w, "forbidden", http.StatusForbidden)
    return
}
```

## Sample rates

Telephony audio is µ-law 8 kHz on the wire, and always will be. The pipeline
does not have to run at that rate: the serializer converts at each edge, so set
the pipeline rate to whatever suits its services and let it follow.

```go
const pipelineSampleRate = 16000

params.AudioInSampleRate = pipelineSampleRate
params.AudioOutSampleRate = pipelineSampleRate

tts := elevenlabs.NewTTS(elevenlabs.Config{
    APIKey:     key,
    SampleRate: pipelineSampleRate,
})

task := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
	Params: pipeline.Params{
	    AudioInSampleRate:  pipelineSampleRate,
	    AudioOutSampleRate: pipelineSampleRate,
	},
})
```

Running the whole pipeline at 8 kHz works and saves two conversions, but it
hands 8 kHz to the transcriber and asks the voice for 8 kHz back. Both are
audibly worse than converting once on the way in and once on the way out, so
16 kHz is the better default. Ask the TTS provider for the pipeline rate
directly, so its audio is not synthesized at 24 kHz and downsampled twice.

Two knobs on `wsserver.AudioConfig` cover the rest:

```go
ser := twilio.New(twilio.Config{
    Audio: wsserver.AudioConfig{
        SampleRate:          24000, // override the pipeline rate for this leg
        ResamplerClearAfter: -1,    // never clear the resampler history
    },
})
```

A stream resampler that has sat idle starts the next chunk fresh, so the tail of
one utterance is not filtered into the start of the next. Providers whose chunks
arrive at irregular intervals want that off, since those gaps are gaps in
delivery rather than gaps in the audio.

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
  budget for it in prompts. Confirm important values back to the caller. Running
  the pipeline at 16 kHz does not recover what the wire never carried: it only
  stops the transcriber and the voice from working at telephone bandwidth on top
  of that.
- **Turn-taking matters more on the phone.** There is no video, no visual
  backchannel, and callers expect the rhythm of a phone conversation. Tune it.
- **Recording is usually regulated.** Consent requirements vary by jurisdiction.
