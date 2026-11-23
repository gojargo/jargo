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
| **TTS** | ElevenLabs, Cartesia, Rime, Kokoro, Piper, Pocket TTS, Deepgram, OpenAI, Azure, Fish, MiniMax, xAI, NVIDIA, Soniox, AssemblyAI, HeyGen, Zoom AI |
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
streaming API or through a provider's batch API:

```go
type Transcriber interface {
    Transcribe(ctx context.Context, audio []byte, sampleRate int) (string, error)
}
```

A provider whose result carries more than text (one transcript per speaker of a
diarized result, say) implements `SegmentRunner` instead, yielding the frames
itself, and is built with `stt.NewSegmentRunner`.

Segment STT emits no interim transcriptions. It still works with
`WithTurnTaking()`, which gates on a *finalized* transcript, but start strategies
that key on partial transcripts never fire, so turn starts fall back to VAD alone.
`chat.NewSTT` and `deepgram.NewSegmentedSTT` (Deepgram's pre-recorded API, one
request per segment) are segment-based; `deepgram.NewSTT` streams.

#### Rewriting transcripts

Both STT services take text transforms, as a TTS service does. Each transcript
is rewritten before the service pushes it, so the context, observers such as
the RTVI client's transcript and the STT tracing span see only the rewritten
text: for example, to redact a card number a caller reads out.

```go
svc.AddTextTransformer(stt.TextTransformer{
    TextType:  frames.AnyTextType, // or stt.TranscriptionFinal, stt.TranscriptionInterim
    Transform: maskCardNumbers,
})
```

A transform sees a final transcript as `stt.TranscriptionFinal` and an interim
one as `stt.TranscriptionInterim`. A transcription whose text a transform
changed loses its `Result`, the provider's response with the text as heard, and
one whose transform failed is not pushed at all. `SetTextTransformers` replaces
the list, which is also how a transform is withdrawn.

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

#### Sentences in the language spoken

A TTS service is given the model's output a sentence at a time, and where a
sentence ends depends on the language: "bzw." ends nothing in German, and an
abbreviation list is what tells "Dr. Smith" from the end of a sentence. The
service finds its boundaries in the language its settings name, English when
they name none, with rules for some forty languages; a
`TTSUpdateSettingsFrame` changing the language applies from the next text, and
keeps what is already buffered. An aggregator built by hand takes its language
too, as `text.NewSimpleAggregator(frames.TextTypeSentence, "de")`.

#### Pronunciations

A voice guesses a name or a term it does not know from its spelling.
`PronunciationTransformIPA` builds a text transform from a word-to-IPA mapping,
and each provider writes the IPA in its own markup: Cartesia as inline phoneme
blocks, ElevenLabs as SSML `<phoneme>` tags, Inworld as IPA between slashes and
Deepgram Aura-2 as an inline pronunciation object. One mapping therefore works
with any of them. A word the provider cannot use is reported once, when the
transform is built, and spoken as written.

```go
pronounce := t.PronunciationTransformIPA(map[string]string{"Metformin": "mɛtˈfɔɹmɪn"})
t.SetTextTransformers(stripMarkdown, pronounce) // last, so nothing rewrites its markup
```

Register it last, so no later transform rewrites the markup it inserts. The
ElevenLabs tags are read only by `eleven_flash_v2` and `eleven_turbo_v2`, and on
the WebSocket service only with `EnableSSMLParsing` on; with any other setup
the transform is skipped with a warning and the words are spoken as written.
Deepgram Flux has no pronunciation markup.

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
| `llm.WithCancelOnInterruption(false)` | Registers an **asynchronous** tool: the model carries on rather than waiting, and the call survives a barge-in. A result that arrives before the conversation moves on settles into the call's placeholder like any other; one that arrives after reaches the model on a later turn as a developer message. Call `p.Result` with `IsFinal` false for the ones before the last. The call is background work: the service broadcasts a `BackgroundWorkStartedFrame` as it starts and a `BackgroundWorkFinishedFrame` once it settles, so a user quiet meanwhile counts as [waiting](turn-taking.md#idle-and-waiting), not idle. |
| `llm.WithTimeout(d)` | Bounds one function's calls. |
| `llm.WithFunctionCallTimeout(d)` | Bounds every call. One that overruns is given up on: it records as completed rather than answering on the tool's behalf. |
| `llm.WithSequentialFunctionCalls()` | Runs the calls of one response one after another, for tools that share something not safe to use concurrently. |
| `llm.WithUngroupedFunctionCalls()` | Re-runs generation per result instead of once the batch finishes. |
| `llm.WithCancellableByLLM(true)` | Lets the model stop an asynchronous tool's calls while they run, through a built-in `cancel_<name>` tool advertised beside it. Pair it with `WithCancelOnInterruption(false)`. |

Register the empty name for a **catch-all** that takes any call no named handler
claims. `UnregisterFunction` withdraws a handler, `HasFunction` asks whether a
call would be claimed, and `OnFunctionCallsStarted` / `OnFunctionCallsCanceled`
report which calls a response started and which an interruption took away.
`RunningFunctionCalls` lists the calls in flight, and `CancelFunctionCall`
cancels one by its `tool_call_id`.

While an asynchronous tool is registered, the service adds guidance on such
tools to its system instruction, and while a cancellable one is, the guidance on
canceling. Both come from the service's async-tool extension, which every LLM
service holds (see [LLM extensions](#llm-extensions)).

## LLM extensions

An LLM extension is a pluggable feature of an LLM service. It embeds
`llm.BaseLLMExtension` and overrides what it needs:

| Method | What it does |
|---|---|
| `SystemInstruction()` | Standing instructions, composed onto the service's system instruction after the application's prompt. |
| `BuiltinTools(registered)` | Tools added to the service's built-in tools, given the registered ones. Their names are reserved, and their calls are announced only when `AnnounceCallStart` is set. |
| `ProcessFrame` | Watches every frame the service processes, before the service handles it. |
| `FilterPushFrame` | Filters every frame the service pushes: passes it on, holds it back, drops it, or adds frames. |
| `Order()` | Where the extension sits among the others. |

Extensions wrap the service the way middleware wraps a handler. An incoming frame
goes through them from the outside in and an outgoing frame from the inside out,
and a frame an extension pushes itself, with `PushFrame` or `BroadcastFrame`,
goes through the extensions outside it. A service holds one extension of each
type; `llm.GetExtension[*T](service)` finds it.

A component that has the service in hand attaches one with `AddExtension`
before the pipeline runs. A running service takes one over an
`LLMUpdateSettingsFrame` whose `settings.LLM` delta lists it in `Extensions`; an
entry that is a `func() llm.LLMExtension` builds one for each service the update
reaches, which is what an update reaching every LLM behind a switcher needs,
since an extension belongs to one service.

jargo's own features are extensions: turn completion
(`llm.LLMTurnCompletionLLMExtension`, installed by the turn strategy, see
[Turn-taking](turn-taking.md)), the async-tool guidance and cancel tools
(`llm.AsyncToolLLMExtension`), and the pairing of a frontend with its backend
(`llmwithbackend.BackendPairingLLMExtension`).

## Speech-to-speech

A single model replaces the STT → LLM → TTS trio:

```go
s2s := realtime.New(realtime.Config{APIKey: key})

pipeline.New(t.Input(), s2s, t.Output())
```

The three implementations are `provider/openai/realtime`,
`provider/google/live` and `provider/aws/novasonic`.

Each takes its session's configuration in its `Config`. For OpenAI Realtime that
is `SessionProperties`: the voice, the transcription of the user's speech, turn
detection (server or semantic VAD, or turned off so the pipeline's own turn
detection drives the session), noise reduction and reasoning. An
`LLMUpdateSettingsFrame` carrying a `realtime.Settings` delta changes them while
the pipeline runs.

Lower latency and better prosody, at the cost of the per-stage control you get
from three separate services. You cannot inspect the transcript before the model
answers, or swap just the voice.

## A frontend with a backend

`llmwithbackend.New` pairs a fast conversational LLM, the **frontend**, with a
slower, stronger one that has the tools, the **backend**. It is alpha: its API is
likely to change between releases. Put it where the LLM goes:

```go
convo := frames.NewLLMContext("You fix failing tests in the repository.")
convo.SetTools([]frames.Tool{readFile, runTests})
backend, err := llmworker.NewBackend(llmworker.BackendConfig{LLM: anthropicLLM, Context: convo})
llm, err := llmwithbackend.New(llmwithbackend.Config{Frontend: openaiLLM, Backend: backend})

pipeline.New(t.Input(), stt, agg.User(), llm, tts, t.Output(), agg.Assistant())
```

The frontend gets one built-in tool, `delegate`, which hands the request to the
backend and returns at once, so the frontend acknowledges while the backend
works. Everything the backend writes comes back as a message in the frontend's
conversation marked `Backend:`. Its model begins a message with `>>` when the
user should hear it, and the frontend then runs on it; everything else it writes
is kept as notes on the work. The user can keep talking meanwhile: a second
request joins the work in progress, and a correction changes or stops it.

The frontend's system instruction says who the assistant is, the backend's says
what work it does, and each gets a pairing instruction that explains how the two
work together. On the frontend, the pairing instruction and `delegate` come from
a `llmwithbackend.BackendPairingLLMExtension` the pairing installs on it, and a
call of `delegate` is announced as a call of the application's own tools is. `llmwithbackend.ConnectorConfig` changes the wording, how
requests are composed, and whether each exchange is traced to the client as an
RTVI server message. The frontend has to carry built-in tools: the OpenAI chat
completions, Anthropic and Gemini services do.

A text frontend spends a whole inference deciding to call `delegate` before the
backend can start. `llmwithbackend.PredictiveDelegation` can save that time: a
fast yes/no classifier, such as `classifier/jev`, races the frontend on each
user turn, and a "yes" that comes before the frontend's first output cancels its
inference and delegates the turn at once. The cost is money, not time: the
classifier runs on every turn, and an early delegation abandons a frontend
request already under way.

```go
judge, err := jev.New(jev.Config{APIKey: jevKey})
pd, err := llmwithbackend.NewPredictiveDelegation(llmwithbackend.PredictiveDelegationConfig{Classifier: judge})
llm, err := llmwithbackend.New(llmwithbackend.Config{
    Frontend: openaiLLM, Backend: backend, PredictiveDelegation: pd,
})
```

It needs a text frontend and a request strategy whose `delegate` call takes no
arguments, such as the default transcript strategy.

## Switching at runtime

`pipeline.NewServiceSwitcher` routes to one of several services, changed mid-call
by pushing a `ManuallySwitchServiceFrame`:

```go
sw, err := pipeline.NewServiceSwitcher(
    []processor.Processor{fastLLM, smartLLM},
    pipeline.NewManualStrategy, // or pipeline.NewFailoverStrategy
)
...
task.QueueFrame(frames.NewManuallySwitchServiceFrame(smartLLM))
```

The second argument is the strategy that decides when the active service
changes. `NewManualStrategy` changes it only when asked. `NewFailoverStrategy`
additionally moves to the next service when the active one reports a non-fatal
error, which is the cheap way to survive a provider outage. Implement
`pipeline.SwitcherStrategy` for anything else, and pass its constructor: the
switcher builds the strategy over the services it manages, so the two cannot
drift apart.

Every service is started and kept warm, but only the active one receives data
and only the active one is heard from: the metadata a service broadcasts about
itself reaches the rest of the pipeline only while it is in use, and a switch
asks the new service to describe itself again.

A switch request naming a service this switcher does not manage travels on, so
several switchers can sit in one pipeline and each pick up its own.

Useful for escalating to a stronger model when a conversation gets hard, without
rebuilding the pipeline.

---

See **[Writing a service](../extending/custom-service.md)** to add a provider.
