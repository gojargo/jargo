# Benchmarks

This module measures the overhead jargo itself adds to a voice agent: moving
frames through a pipeline, starting one, and running many at once. It also runs
the same scenarios against [Pipecat](https://github.com/pipecat-ai/pipecat), the
Python framework jargo is a port of, under the same CPU and memory limits, so the
two can be compared on equal terms.

It is a Go module of its own (`github.com/gojargo/jargo/benchmarks`) that builds
against the library in the tree beside it, so the library takes on none of its
dependencies.

## What is measured

Every pipeline is a chain of pass-through processors that forward each frame
and do nothing else, so what is timed is the framework: the queue, the handoff
and the dispatch at every hop. Each pipeline runs the way an application runs a
bot, a runner holding one pipeline worker, and frames are seen arriving through
the worker's reached-downstream event. The worker is configured the same way on
both sides: no turn tracking, no idle timeout, no RTVI, no observers, logging
silenced.

| Kind | Sub-benchmarks | What one sample is |
|---|---|---|
| `latency` | `depth=1,4,16,64` | Closed loop: queue one frame, wait for it at the end, then the next. Warm-up frames and startup are excluded. Reports `ns/frame`, `ns/hop` and the median and 99th percentile per frame. |
| `throughput` | `depth=1,4,16,64` | Open loop: queue every frame at once, time from the first queued to the last arrived. Startup is excluded. Reports `ns/frame` and `frames/s`. |
| `startup` | `depth=1,4,16,64` | From running a pipeline to its StartFrame reaching the end (`ns/start`), and building it beforehand (`build-ns/start`). |
| `sessions` | `sessions=1,10,100` (and `1000` in the full profile) | Many depth-4 pipelines at once: startup per session, then rounds of one frame into every pipeline. Reports `ns/round`, `ns/frame`, `frames/s`, `startup-ns/session` and the resident memory per session (`B/session`). |

Every sample has a time budget. A sample that runs past it prints a
`--- TIMEOUT` line instead of a result and the larger sizes of that benchmark
are skipped, so a run ends rather than hangs.

## Layout

| Path | What it holds |
|---|---|
| `scenarios/` | The scenario files, read by both sides: `quick.json` for a laptop, `full.json` for CI. |
| `internal/scenario` | The scenario types and their loader. |
| `internal/run` | One implementation per kind, on jargo's public API. |
| `cmd/jbench` | Runs a scenario and prints Go benchmark format. |
| `bench_test.go` | The same kinds as `testing.B` benchmarks, for jargo-only regressions. |
| `out/` | Results of local runs. Ignored by git: numbers are never committed. |

## Running jargo's side

```sh
# One scenario, printed in Go benchmark format.
go -C benchmarks run ./cmd/jbench run -count 6 scenarios/quick.json

# The testing.B benchmarks, to compare two commits of jargo with benchstat.
go -C benchmarks test -run '^$' -bench . -count 6 ./...
```

The config lines that open jbench's output say what was measured:
`framework`, `framework_version` (the jargo revision), `runtime`, the machine,
and the scenario.

## Checking it still builds

```sh
make bench-check
```

CI runs this on every pull request, so a change to the library's API that breaks
the suite fails there rather than the next time someone runs it.
