# Benchmarks

This module measures the overhead jargo itself adds to a voice agent: moving
frames through a pipeline, starting one, and running many at once. It also runs
the same scenarios against [Pipecat](https://github.com/pipecat-ai/pipecat), the
Python framework jargo is a port of, under the same CPU and memory limits, so the
two can be compared on equal terms.

It is a Go module of its own (`github.com/gojargo/jargo/benchmarks`) that builds
against the library in the tree beside it, so the library takes on none of its
dependencies.

## Layout

| Path | What it holds |
|---|---|
| `go.mod` | The module, with `replace github.com/gojargo/jargo => ../` and `benchstat` as a tool. |
| `out/` | Results of local runs. Ignored by git: numbers are never committed. |

## Checking it still builds

```sh
make bench-check
```

CI runs this on every pull request, so a change to the library's API that breaks
the suite fails there rather than the next time someone runs it.
