---
title: Concepts
weight: 2
---

# Concepts

A jargo bot is a **chain of processors** that passes **frames** to each other.
These six pages are that sentence, unpacked.

The engine is small: `frames`, `processor` and `pipeline` are about 2,000 lines
between them. It is worth reading properly once rather than guessing at it
later. Everything else in jargo, transports and services included, is a processor
plugged into it.

Read in order:

1. **[Architecture](architecture.md)**: the whole system on one page, and a full
   turn traced end to end.
2. **[Frames](frames.md)**: the three categories, why they exist, and what all 62
   frame types are for.
3. **[Processors](processors.md)**: the two goroutines inside every processor, and
   the reason there are two.
4. **[Pipeline & Task](pipeline.md)**: building the chain and driving it.
5. **[Interruptions](interruptions.md)**: barge-in as a mechanism rather than a
   metaphor.
6. **[LLM context](llm-context.md)**: how the conversation accumulates, and what
   decides when the LLM runs.

If you only read two, read *Frames* and *Interruptions*. Frame category is the
property that decides everything else, and interruptions are where a design that
looks over-engineered turns out not to be.
