---
title: Extending
weight: 4
---

# Extending

Two extension points cover nearly everything.

- **[Writing a processor](custom-processor.md)**: add your own logic to the chain,
  such as redacting text before it reaches the LLM, logging transcripts, hanging
  up on a keyword, or counting words. A struct embedding `*processor.Base` and one
  method.
- **[Writing a service](custom-service.md)**: add an STT, LLM or TTS provider.
  Usually one interface method; the shared base handles frames, metrics, tracing,
  interruption and the tool loop.

Read **[Processors](../concepts/processors.md)** first, particularly the part about
the two goroutines. It is the thing that catches people out.
