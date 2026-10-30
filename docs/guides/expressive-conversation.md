---
title: Expressive conversation
weight: 7
---

# Expressive conversation

> **Alpha.** The tag syntax and the types behind it are likely to change between
> releases.

An expressive conversation lets the bot answer in kind: the LLM writes
expression into its reply as inline tags, and the TTS service performs them
instead of reading them aloud.

```text
<<warm>> It's [[laugh]] ^^back^^ up. [[pause 300]] Your code is <<slow: A B 7>>.
```

| Tag | Meaning |
|---|---|
| `<<name>>` | The delivery of the words after it, until the next delivery tag or the end of the reply. `<<neutral>>` returns to the voice's baseline. |
| `<<name: words>>` | The delivery of just those words, added to the delivery around them. |
| `^^words^^` | Stress those words. |
| `[[name]]` | A sound at one point, such as `[[laugh]]`. |
| `[[pause N]]` | A silence of N milliseconds. |

## Turning it on

Set it on the pipeline. Every processor learns it at setup:

```go
task := pipeline.NewWorker(pipeline.New(transport.Input(), stt, userAgg, llm, tts,
	transport.Output(), assistantAgg), pipeline.WorkerConfig{
	Params: pipeline.Params{EnableExpressiveConversation: true},
})
```

With it on, the TTS service:

- keeps each tag whole as text streams in, however the LLM splits it;
- sends each tag in its provider's markup, such as ElevenLabs' `[warmly]` or
  Cartesia's `<emotion value="content"/>`, and leaves out what the provider
  cannot perform, warning once per tag;
- shows clients the words alone, and reports the delivery of each word it
  speaks and each sound and pause it performs (see [RTVI](rtvi.md));
- broadcasts a `frames.TTSMetadataFrame` at start, both ways, whose
  `Expression` says what the service can perform, so the tags the LLM is taught
  can match.

The conversation keeps the reply as the LLM wrote it, tags included, so the
model reads its own expression back. What the assistant turn is reported to
have said (`aggregators.AssistantTurnStopped` and `LLMContextAssistantTurnFrame`)
is the words alone.

Teaching the LLM to write the tags is up to the prompt for now.

## Universal names

Each delivery and sound has one universal name, the same on every service:
`expressive.ExpressionDelivery` (such as `warm`, `calm`, `slow`) and
`expressive.ExpressionSound` (such as `laugh`, `sigh`, `breath`). A universal
name in a tag is sent by the provider's own name for it, through its
capabilities' `ServiceVocabulary`, while everything reported keeps the name as
written. A name that is not universal is sent as written, to a provider that
declares it or accepts free text.

## Providers

Each provider performs expression through an expression adapter, in a package
under `adapter/tts`. A provider without one speaks only the words.

| Provider | Performs expression on | Delivery |
|---|---|---|
| Cartesia (WebSocket and HTTP) | every model | emotions, between sentences; slow and fast as a speed |
| ElevenLabs (HTTP) | Eleven v3 and v4 | free-text audio tags; emphasis in capitals |
| Fish Audio | S2 models | free-text tags; emphasis |
| MiniMax | speech-2.8 models | emotion and speed with each request |
| Inworld | `inworld-tts-2` (and sounds and pauses on `inworld-tts-2-flash`) | free-text instructions; emphasis in capitals |
| Soniox | `tts-rt-v2` | its tags; emphasis |
| Mistral | Voxtral 3 (`voxtral-mini-tts-3*`) | emotion and pace tags |
| Azure | DragonHD and DragonHD Omni voices | style markers |

The Gemini 3.8 adapter (`adapter/tts/gemini`) is there for a Gemini API TTS
service, which jargo does not have yet.

A provider whose expression depends on its settings chooses the adapter for
them, and the service broadcasts its metadata again when a settings change
alters what it can perform.

## Writing an adapter

A provider of your own implements `tts.ExpressionAdapterProvider` on its
synthesizer, returning an adapter that embeds `BaseExpressionAdapter` from
`adapter/tts` and overrides what the provider performs: `Capabilities`, and the methods that
format words in their delivery, sounds, pauses and the end of a request. One that
takes delivery as a request setting returns it from `RequestSettings`, and reads
it back with `tts.Base.ExpressionSettings` while it builds the request.
