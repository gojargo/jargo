---
title: Expressive conversation
weight: 7
---

# Expressive conversation

> **Alpha.** The tag syntax and the types behind it are likely to change between
> releases.

An expressive conversation lets the bot answer in kind: the LLM hears how the
user spoke, writes expression into its reply as inline tags, and the TTS
service performs them instead of reading them aloud.

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
model reads its own expression back, unless a [tagger](#a-tagger-after-the-llm)
wrote them. What the assistant turn is reported to
have said (`aggregators.AssistantTurnStopped` and `LLMContextAssistantTurnFrame`)
is the words alone.

## Teaching the LLM

With it on, every LLM service teaches itself the tags the TTS service it feeds
can perform, in its system instruction, after the application's prompt. It
reads them from the `frames.TTSMetadataFrame` that TTS service broadcasts, so
the prompt does not describe them. The instruction follows the active TTS
service: a `ServiceSwitcher` switch, or a settings change such as a new model,
broadcasts the metadata again, and the instruction is rebuilt from it. An LLM
held inactive in an `LLMSwitcher` is taught too.

The pipeline's `expressive.ExpressionPolicy` says which tags are offered and
when to use them:

```go
Params: pipeline.Params{
	EnableExpressiveConversation: true,
	ExpressionPolicy: &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{
			expressive.GuidanceMatchStyle,
			expressive.GuidanceLine("Never laugh while discussing billing."),
		},
		Delivery: []expressive.ExpressionDelivery{
			expressive.DeliveryCalm, expressive.DeliveryHappy,
			expressive.DeliverySympathetic, expressive.DeliveryExcited,
		},
	},
},
```

- `Delivery` and `Sounds` narrow the tags offered, by their universal names.
  Left nil, the core ones are offered (`expressive.CoreDelivery` and
  `expressive.CoreSounds`), those the TTS service performs; an empty list
  offers none. An LLM starts inventing names when offered more than about 15
  from a fixed list. Tags the TTS service cannot perform are left out, with a
  warning.
- `Guidance` lines say when to express, joined in order: the presets
  `GuidanceMatchStyle` (answer in the user's mood and energy, but stay calm
  with hostility), `GuidanceSlowForDetails` (say just what the user may need
  to write down slowly, in a span such as `<<slow: A B 7>>`) and
  `GuidanceBreathe` (an occasional breath before a long or difficult
  sentence, which is also what offers `[[breath]]`), and lines of your own,
  as `GuidanceLine`.

`expressive.ExpressionInstructions` returns the instruction for a service's
capabilities and a policy, to read it or to build a prompt of your own, and
`expressive.ExpressionPolicyWarning` what the policy asks for that the service
cannot perform.

## Perception

A perception producer, such as an STT service or a processor after it, reads
how the user spoke and sets the reading on the transcription it passes along,
in `Perception`: signals such as a tone, each with its likeliest labels and how
sure the reading is, or a measure such as speech rate. It declares the signals
it reports in a `frames.PerceptionMetadataFrame` at start.

The user aggregator writes the turn's final reading at the head of its user
message:

```text
([tone frustrated=0.86, neutral=0.10]) Can you tell me where my order is?
```

and the LLM's instruction explains how to read it, from the producer's
metadata: what the confidences mean, that a reading below 0.5 is a guess, and
each signal. A reading changes the bot's delivery on its own, but changes what
it says only when the policy's guidance names a reaction, such as a
`GuidanceLine("If the user sounds frustrated at 0.7 or above, apologize once.")`.
A reading of an interim transcription is provisional and never reaches the
context. One of an eager transcription is written into the message an early
reply is generated from.

The application gets the turn's reading in `aggregators.UserTurnStopped.Perception`,
processors after the user aggregator in `frames.LLMContextUserTurnFrame.Perception`,
and clients in the `perception` field of the transcription's `user-input`
message (see [RTVI](rtvi.md)), which waits for a producer after the STT to read
it.

jargo ships no perception producer yet: implement one as a processor after
the STT that broadcasts its `PerceptionMetadataFrame` at start and sets
`Perception` on each final `frames.TranscriptionFrame`.

## A tagger after the LLM

A processor between the LLM and the TTS service can write the tags itself, from
plain replies. It broadcasts a `frames.ExpressionMetadataFrame` at start. The
LLM is then taught no tags, keeping how to read perception and the policy's own
guidance lines when there is a reading to act on, while the presets are left
to the tagger. The assistant aggregator writes each reply to the context as its
words alone, so the LLM reads back what it wrote.

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
