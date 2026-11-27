---
title: Compatibility
weight: 7
---

# Versioning and compatibility

jargo follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until
`1.0.0` a minor release may still change the public API, and the
[changelog](https://github.com/gojargo/jargo/blob/main/CHANGELOG.md) marks every
change that does. From `1.0.0` on, this page is the promise.

## What 1.x promises

Within 1.x, code that compiles against one release compiles against every later
one, and keeps its meaning:

- **No exported identifier is removed or renamed**, and no signature changes in
  a way that breaks a caller. A breaking change waits for 2.0.
- **Deprecation comes first.** A deprecated identifier says so in its doc
  comment (`Deprecated:`), names its replacement, and keeps working for at least
  two minor releases. It is removed only in the next major release.
- **Config structs only grow.** A new field's zero value means "the default",
  so code that does not set it behaves as before. A field is never made
  required after the fact.
- **Interfaces you implement only grow in ways that cannot break you.** An
  interface meant to be implemented outside jargo is one of two kinds:
  - **Sealed**: you implement it by embedding its base type, and a method
    added later comes with its default in the base. These are
    `processor.Processor` (`processor.Base`), `transport.Transport`
    (`transport.BaseTransport`), `vad.Analyzer` (`vad.Base`), `turn.Analyzer`
    (`turn.Base`), `text.Aggregator` (`text.BaseAggregator`),
    `classifier.Classifier` (`classifier.Base`) and `pipeline.SwitcherStrategy`
    (`pipeline.BaseSwitcherStrategy`).
  - **Frozen**: every other exported interface. Its method set does not
    change in 1.x, and a new capability arrives as a separate optional
    interface, which is how the STT, LLM and TTS service hooks already work.
- **Interfaces jargo hands to you may grow.** A few interfaces exist for jargo
  to give your code something to call, not for you to implement:
  `processor.Running`, `tts.AudioContextHost`. They may gain methods in a minor
  release; implement them only as test doubles.

## What it does not cover

- **Experimental APIs.** A package or identifier whose doc comment says
  *Experimental* or *Alpha* may change in a minor release, and the changelog
  says so. These are the parts that still follow a fast-moving upstream design:
  flows, the workers and the bus (with the network buses), LLM-with-backend,
  the eval harness, and the expressive-conversation and perception features.
  An experimental API leaves that status in a minor release, never the other
  way round.
- **Provider behaviour.** A provider package's Go API is covered. What the
  vendor does is not: a provider's default model, voice or endpoint may change
  in a minor release to follow the vendor, and the changelog says so. Set them
  explicitly where it matters to you. When a vendor retires a service, its
  package is deprecated and keeps compiling until the next major release,
  though calls to the retired service fail.
- **Log messages and tracing details.** Log text, log attributes, and the names
  and attributes of tracing spans may change in any release. Metric names are
  covered (see below).
- **Internal packages**, anything under `internal/`, and the examples.

## Beyond the Go API

These are also covered by the 1.x promise:

- the `JARGO_*` environment variables and the development runner's flags and
  routes;
- metric names (`jargo.ttfb`, `jargo.processing`, ...);
- the bus's wire names (`jargo:bus`);
- the RTVI protocol, which follows its own versioning: a client of an older
  protocol version is answered in its own version.

## Upstream changes

jargo is a port of [Pipecat](https://github.com/pipecat-ai/pipecat) and keeps
following it. A breaking change upstream reaches jargo 1.x in one of three
ways, in order of preference:

1. adapted behind the existing jargo API, when the behaviour can follow without
   the API changing;
2. added alongside it, with the old API deprecated as above;
3. in an experimental package, as a breaking change the changelog calls out.

## Go versions

jargo supports the two most recent Go releases, as the Go project does. Raising
the minimum Go version is not a breaking change.

## Security fixes

Only the latest release receives security fixes. See
[`SECURITY.md`](https://github.com/gojargo/jargo/blob/main/SECURITY.md).
