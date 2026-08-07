---
title: Services
weight: 2
---

# Services

STT, LLM and TTS providers are all processors behind small interfaces. Swapping
one is a one-line change, because nothing in the pipeline depends on which
provider you picked.

## The pattern

Every provider is a `Config` struct plus a constructor:

```go
stt := deepgram.NewSTT(deepgram.Config{APIKey: key})
llm := anthropic.NewLLM(anthropic.Config{APIKey: key, Model: "claude-sonnet-5"})
tts := cartesia.NewTTS(cartesia.Config{APIKey: key, VoiceID: id})
```

Library packages read **no environment variables** and take no functional options.
Configs are validated with `go-playground/validator` tags, so a bad config fails
at construction rather than mid-call. Reading env vars, flags or config files is
your app's job. See `examples/`.

Most fields have sensible defaults: an empty `Model` picks the provider's
recommended one, and `SampleRate: 0` inherits the transport's rate.

> **The config type name varies by provider.** A provider that offers one service
> names it plainly `Config`; one that offers several qualifies the extras. So it
> is `deepgram.Config` for STT but `deepgram.TTSConfig` for TTS, `cartesia.Config`
> for TTS but `cartesia.STTConfig` for STT, and `chat.STTConfig` /
> `chat.LLMConfig` / `chat.TTSConfig` for all three. Check the package, or
> let the compiler tell you.

## Providers

Pick any per category.

| Category | Providers |
|---|---|
| **STT** | Deepgram, AssemblyAI, Gladia, Speechmatics, Soniox, Whisper (OpenAI/Groq/local), Azure, xAI, ElevenLabs, Cartesia, NVIDIA |
| **LLM** | Anthropic (direct + Bedrock), OpenAI (chat + Responses), Gemini (direct + Vertex), Groq, Together, Fireworks, DeepSeek, Cerebras, Perplexity, OpenRouter, xAI, Ollama, NVIDIA, Mistral, Nebius, SambaNova, Qwen, Azure OpenAI |
| **TTS** | ElevenLabs, Cartesia, Rime, LMNT, Kokoro, Piper, Pocket TTS, Deepgram, OpenAI, Azure, Hume, Fish, MiniMax, xAI, NVIDIA, Soniox |
| **Speech-to-speech** | OpenAI Realtime (direct + Azure), Gemini Live (direct + Vertex), AWS Nova Sonic, xAI Realtime |
| **Memory** | mem0 |

Each lives in `provider/<name>`, or `provider/<vendor>/<name>` when a vendor
offers several (`provider/azure/speech`, `provider/openai/chat`,
`provider/aws/bedrock`). There are more in the tree than listed above; browse
[`provider/`](../../provider) for the current set. Per-provider runnable
examples are in [`examples/voice/`](../../examples/voice).

Coverage is uneven: the providers used by the examples get the most exercise, and
some of the others are thinly tested. Bug reports naming a specific provider are
especially useful.

## The interfaces

You rarely implement these, but knowing their shape explains what a provider can
and cannot do.

### STT

Two flavors. **Streaming** is what you want for conversation:

```go
type Connector interface {
    Connect(ctx context.Context, sampleRate int) (Stream, error)
}

type Stream interface {
    Send(audio []byte) error
    Recv() ([]Result, error)
    Close() error
}
```

`stt.StreamService` wraps a `Connector` into a processor: audio in, interim and
final `TranscriptionFrame`s out.

**Segment** STT transcribes a finished buffer instead, for providers with no
streaming API:

```go
type Transcriber interface {
    Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error)
}
```

Segment STT emits no interim transcriptions. It still works with
`WithTurnTaking()`, which gates on a *finalized* transcript, but start strategies
that key on partial transcripts never fire, so turn starts fall back to VAD alone.
`chat.NewSTT` is segment-based; `deepgram.NewSTT` streams.

### LLM

```go
type Generator interface {
    Generate(ctx context.Context, convo *frames.LLMContext, emit Emit) error
}
```

Stream deltas to `emit` until done or `ctx` is canceled. **Cancellation is an
interruption**: honor it, or barge-in stalls for up to three seconds.

Tool-capable providers implement `ToolGenerator` as well:

```go
type ToolGenerator interface {
    GenerateWithTools(ctx context.Context, convo *frames.LLMContext, sink Sink) error
}
```

`llm.Base` runs the tool loop automatically when the context carries tools and the
generator supports them.

### TTS

```go
type Synthesizer interface {
    SampleRate() int
    Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error
}
```

Providers that return word timings also implement `WordTimestamps`, which is what
lets `TTSTextFrame`s align to the audio actually being spoken, and therefore what
lets an interrupted response be recorded truncated rather than whole.

## Tool calling

Register a handler by name, then advertise the tool on the context:

```go
llm.RegisterFunction("get_order_status", func(ctx context.Context, p llm.FunctionCallParams) error {
    var in struct{ OrderID string `json:"order_id"` }
    if err := json.Unmarshal(p.Arguments, &in); err != nil {
        return err
    }
    status, err := lookup(ctx, in.OrderID)
    if err != nil {
        return err
    }
    return p.Result(ctx, status, nil)
})

convo.SetTools([]frames.Tool{{
    Name:        "get_order_status",
    Description: "Look up the status of an order by its ID.",
    Parameters:  json.RawMessage(`{
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"]
    }`),
}})
```

`Parameters` is a raw JSON-Schema object. A handler reports what it produced
through `p.Result` rather than returning it, because a call can have more than
one thing to say. A returned error is reported as a non-fatal pipeline error and
puts nothing in the tool's mouth, so a failure the model should see belongs in
the result instead.

A handler that blocks **must honor `ctx`**, the same interruption rule as
`Generate`: the call's context is canceled when the user barges in.

The call and the message answering it are written together the moment the call
starts, so the conversation is valid at every instant, and the result replaces
that placeholder in place. See
[LLM context](../concepts/llm-context.md#tool-calls).

A tool can carry its own handler instead, which saves keeping two lists in step:

```go
convo.SetTools([]frames.Tool{{
    Name:        "get_order_status",
    Description: "Look up the status of an order by its ID.",
    Parameters:  schema,
    Handler:     lookupOrder,
}})
```

The handler is registered when the toolset is advertised and dropped when it
stops being advertised, so what the model can call and what answers are the same
set. A handler registered by hand always wins and is never dropped.

### Tuning how calls run

| Option | What it does |
|---|---|
| `llm.WithCancelOnInterruption(false)` | Registers an **asynchronous** tool: the model carries on rather than waiting, the call survives a barge-in, and every result the handler reports reaches the model on a later turn as a developer message. Call `p.Result` with `IsFinal` false for the ones before the last. |
| `llm.WithTimeout(d)` | Bounds one function's calls. |
| `llm.WithFunctionCallTimeout(d)` | Bounds every call. One that overruns is given up on: it records as completed rather than answering on the tool's behalf. |
| `llm.WithSequentialFunctionCalls()` | Runs the calls of one response one after another, for tools that share something not safe to use concurrently. |
| `llm.WithUngroupedFunctionCalls()` | Re-runs generation per result instead of once the batch finishes. |
| `llm.WithAsyncToolCancellation()` | Offers the model a built-in `cancel_async_tool_call`, while any asynchronous tool is registered, so it can abandon background work it no longer needs. |

Register the empty name for a **catch-all** that takes any call no named handler
claims. `UnregisterFunction` withdraws a handler, `HasFunction` asks whether a
call would be claimed, and `OnFunctionCallsStarted` / `OnFunctionCallsCanceled`
report which calls a response started and which an interruption took away.

## Speech-to-speech

A single model replaces the STT → LLM → TTS trio:

```go
s2s := realtime.New(realtime.Config{APIKey: key})

pipeline.New(t.Input(), s2s, t.Output())
```

The three implementations are `provider/openai/realtime`,
`provider/google/live` and `provider/aws/novasonic`.

Lower latency and better prosody, at the cost of the per-stage control you get
from three separate services. You cannot inspect the transcript before the model
answers, or swap just the voice.

## Switching at runtime

`pipeline.NewServiceSwitcher` routes to one of several services, changed mid-call
by pushing a `SwitchServiceFrame`:

```go
sw, err := pipeline.NewServiceSwitcher(
    []processor.Processor{fastLLM, smartLLM},
    pipeline.SwitchManual,   // or pipeline.SwitchFailover
)
...
task.QueueFrame(pipeline.NewSwitchServiceFrame(smartLLM))
```

`SwitchFailover` additionally moves to the next service when the active one
reports a non-fatal error, which is the cheap way to survive a provider outage.

Useful for escalating to a stronger model when a conversation gets hard, without
rebuilding the pipeline.

---

See **[Writing a service](../extending/custom-service.md)** to add a provider.
