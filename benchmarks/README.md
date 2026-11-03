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
| `pipecat/bench.py` | The same kinds against Pipecat, printing the same format. |
| `pipecat/Dockerfile` | Python 3.12 with the locked Pipecat release, by digest. |
| `pipecat/requirements.in`, `requirements.txt` | The pinned `pipecat-ai` release, and its lock with every package's hashes. |
| `out/` | Results of local runs. Ignored by git: numbers are never committed. |

## Comparing the two

```sh
make bench-compare                       # quick profile on CPUs 2,3 with 1g
make bench-compare BENCH_PROFILE=full BENCH_CPUSET=0-3 BENCH_MEM=4g
```

This builds both images (`make bench-images`), runs jargo's and then Pipecat's
side, one after the other, each with `--cpuset-cpus=$(BENCH_CPUSET)`,
`--memory=$(BENCH_MEM)` and no swap (GOMAXPROCS is set to the size of the
cpuset), and writes `out/jargo.txt`, `out/pipecat.txt` and the benchstat
summary `out/compare.txt`. Other knobs: `BENCH_COUNT` (samples per
sub-benchmark, 6 by default) and `BENCH_RUN` (a regexp selecting
sub-benchmarks, on both sides).

See [docs/benchmarks.md](../docs/benchmarks.md) for the method, the fairness
rules and how to read the result.

`make bench` runs the `testing.B` benchmarks on the host, throttled, into
`out/bench.txt`.

In CI, `.github/workflows/bench.yml` runs the full profile weekly, on demand,
and on pull requests that change `pipecat/`, and uploads `out/` as an
artifact.

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

## Running Pipecat's side

The Python side runs in its image, never on the host:

```sh
docker build -f benchmarks/pipecat/Dockerfile -t jargo-bench-pipecat:local benchmarks/pipecat
docker run --rm --cpuset-cpus=2,3 --memory=1g \
  -v "$PWD/benchmarks/scenarios:/scenarios:ro" jargo-bench-pipecat:local \
  run -count 6 /scenarios/quick.json
```

`bench.py` mirrors `internal/run` kind for kind, and the two have to be kept in
step: a change to how one side measures is a change to both. It uses nothing
beyond the standard library and `pipecat-ai`.

To move to a new Pipecat release, change the pin in `pipecat/requirements.in`
and run `make bench-lock`, which re-resolves `requirements.txt` with hashes
inside the same Python image.

## Checking it still builds

```sh
make bench-check
```

CI runs this on every pull request, so a change to the library's API that breaks
the suite fails there rather than the next time someone runs it.

## Next steps

Not measured yet, and planned:

- **Session density with real-time audio.** Many sessions each fed 20 ms audio
  frames in real time, reporting how many a set of CPUs holds before frames
  start arriving late.
- **End to end.** A full voice pipeline against a mock AI server (STT, LLM, TTS
  with fixed, scripted latencies), driven by a load generator speaking the
  protobuf WebSocket protocol, so the time from the end of user speech to the
  first bot audio is measured through the transports as well.
