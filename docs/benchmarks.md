---
title: Benchmarks
weight: 6
---

# Benchmarks

jargo is a Go port of [Pipecat](https://github.com/pipecat-ai/pipecat). The
benchmark suite in [`benchmarks/`](https://github.com/gojargo/jargo/tree/main/benchmarks)
measures the overhead the framework itself adds to a voice agent, and runs the
same measurements against Pipecat under the same limits so the two can be
compared on equal terms.

What it measures is the framework, not a conversation. The latency a caller
hears is dominated by the round trips to the speech-to-text, LLM and
text-to-speech services, which are the same whichever framework calls them.
What differs between the two is everything around those calls: moving frames
between processors, starting a session, and running many sessions in one
process. Those are the numbers here.

This page gives the method and how to run it. It deliberately quotes no results:
numbers depend on the machine, and the ones that matter are the ones you measure
on yours. The README shows one CI run of the deployment benchmark, with the
machine it ran on.

## What is measured

Every pipeline is a chain of pass-through processors, each forwarding every frame
and doing nothing else, so all that is timed is the framework: the queue, the
handoff and the dispatch at each hop. Each pipeline is run the way an application
runs a bot, by a runner holding one pipeline worker, and a frame is seen
arriving through the worker's reached-downstream event.

| Kind | Sizes | One sample |
|---|---|---|
| Latency | chain depth 1, 4, 16, 64 | Closed loop: one frame is queued, and the next only once it has reached the end. Reports the mean per frame and per hop, and the median and 99th percentile per frame. |
| Throughput | chain depth 1, 4, 16, 64 | Open loop: every frame is queued at once, timed from the first queued to the last arrived. Reports time per frame and frames per second. |
| Startup | chain depth 1, 4, 16, 64 | From asking a pipeline to run to its StartFrame reaching the end, when it is ready for frames. Building the pipeline beforehand is reported apart. |
| Sessions | 1, 10, 100 sessions (and 1,000 in the full profile) | That many depth-4 pipelines at once: the startup per session, then rounds of one frame into every pipeline and all of them back out, and the resident memory each session holds. |
| Deployment | 10, 100 sessions (100 and 1,000 in the full profile) | The same sessions served the way each framework is deployed to use every CPU it is given (see below): what the whole deployment holds idle and with every session running, the memory each session adds, and the frames all of them move. |

Latency and throughput exclude starting the pipeline and a number of warm-up
frames. Every sample has a time budget: one that runs past it is reported as
timed out and the larger sizes of that benchmark are skipped, so a run that
overwhelms one side ends with that fact recorded rather than hanging.

## How the two sides are kept equal

- **One definition.** Both sides read the same scenario files
  (`benchmarks/scenarios/*.json`), which hold the sizes, sample lengths, warm-up
  and budgets. The Go side (`jbench`, over `benchmarks/internal/run`) and the
  Python side (`benchmarks/pipecat/bench.py`) implement each kind the same way,
  step for step: the closed loop really waits for each frame, every frame is
  built fresh on both sides, and the samples are interleaved the same way so a
  slow drift in the machine spreads across all of them.
- **The same configuration.** The worker runs with turn tracking, the idle
  timeout, RTVI and every observer off, and logging silenced, on both sides.
  Both use their framework's defaults otherwise: Pipecat runs on the standard
  asyncio event loop, as it does unless told otherwise.
- **The same limits.** `make bench-compare` runs each side in its own container,
  one after the other and never at once, pinned to the same CPUs with
  `--cpuset-cpus`, held to the same memory with swap off, and with GOMAXPROCS set
  to the number of CPUs in the set.
- **Pinned versions.** The Pipecat side is Python 3.12 (the version Pipecat
  recommends and tests on) in an image pinned by digest, with `pipecat-ai`
  pinned to one release and every package installed from a lock checked by hash.
  The jargo side is built from the tree it runs in. Each side's output names
  its framework version and runtime, along with the CPU model, the CPUs it was
  allowed, their frequency governor, the kernel, the cgroup version and the
  memory limit.

### Sessions and deployment

The two answer different questions. Sessions runs every pipeline in the
benchmark's own process, which is how the framework's per-session overhead is
seen most directly. It leaves out two things a deployment pays for:

- **Using every CPU.** A Go program uses every CPU it is given from one process.
  A Python one is held to one core at a time by its global interpreter lock, so
  in Sessions Pipecat runs on one of the CPUs while jargo runs on all of them,
  and serving the same load on every CPU takes one Python process per CPU.
- **The process itself.** Sessions reports the memory each session adds, not
  what the process holds before the first one: the interpreter and Pipecat's
  imports on one side, the Go runtime and the jargo library on the other. The
  benchmark's process has also run the samples before it by then, and memory
  they freed is reused, which makes a session look cheaper than it is in a fresh
  process. Python's allocator keeps more of it than Go's does.

Deployment measures that. The benchmark starts fresh serving processes, one for
jargo and one per CPU for Pipecat, splits the sessions evenly between them, and
drives them over their standard input and output (the protocol is in
`benchmarks/internal/run/deploy.go`, and both sides speak it). Memory is each
serving process's proportional set size, from `/proc/<pid>/smaps_rollup`, after
it has handed its freed memory back to the system. Proportional means pages
shared between processes, such as a shared library's, are split between them
rather than counted once per process, so the total is what the deployment holds.
The benchmark's own process is not counted on either side.

What is not measured: the Python interpreter's own startup and import time, and
anything a real service, transport or model would add.

The resident memory per session is read from `/proc` after both sides have
collected garbage and handed freed memory back to the system, so it counts what
the running sessions hold. It is only meaningful at the larger session counts;
at one session it is mostly the noise of the allocator.

## Running it locally

On Linux with Docker:

```sh
make bench-compare
```

The defaults suit a laptop that is also in use: the quick profile, CPUs `2,3`
and 1 GiB (`BENCH_CPUSET=2,3`, `BENCH_MEM=1g`, `BENCH_PROFILE=quick`), six
samples of every sub-benchmark (`BENCH_COUNT=6`). The full profile adds a
thousand concurrent sessions and is meant for a dedicated machine:

```sh
make bench-compare BENCH_PROFILE=full BENCH_CPUSET=0-3 BENCH_MEM=4g
```

`BENCH_RUN` selects sub-benchmarks by regexp on both sides, for example
`BENCH_RUN=Latency`. The raw results land in `benchmarks/out/jargo.txt` and
`benchmarks/out/pipecat.txt`, and the benchstat summary in
`benchmarks/out/compare.txt`. That directory is ignored by git: results are
never committed.

For jargo on its own, to compare two commits:

```sh
make bench                    # writes benchmarks/out/bench.txt
go -C benchmarks tool benchstat old.txt new.txt
```

### Docker caveats

- **Pin with `--cpuset-cpus`, not `--cpus`.** `--cpus` is a CFS quota: the
  container may run on every core in short bursts and is then throttled until
  the next period, which shows up as tail latency that belongs to the quota, not
  to the framework. The Makefile only ever pins.
- **Docker Desktop numbers are not comparable.** On macOS and Windows the
  containers run inside a virtual machine, whose scheduling and timers sit
  between the benchmark and the hardware. Compare the two sides measured there
  with each other if you must, never with numbers from native Linux.
- **The governor matters.** A `powersave` governor, or a laptop on battery,
  lowers and varies the clock. The governor is recorded in the output; for
  numbers you mean to keep, use `performance` and a machine that is otherwise
  idle.

## In CI

The `Benchmarks` workflow (`.github/workflows/bench.yml`) runs
`make bench-compare` with the full profile on a GitHub-hosted runner every week,
on demand, and on pull requests that change the Python side. Both sides are
pinned to every CPU of the runner but the first, which is left to the runner
agent and the Docker daemon, with 4 GiB each. The job uploads `benchmarks/out`
(both raw outputs, the benchstat summary and a description of the runner) as an
artifact and shows the summary on the run's page.

Hosted runners are shared virtual machines, so their absolute numbers move from
one run to the next; only the ratio between the two sides measured in the same
job, on the same runner, means anything. Never compare numbers from different
jobs.

## Reading the result

`compare.txt` is benchstat's table, one per unit, with a column per framework:

```text
                         │    jargo    │              pipecat               │
                         │   sec/hop   │   sec/hop    vs base               │
PlumbingLatency/depth=4    ...
```

Each cell is the median of the samples with its 95% confidence interval, and the
`vs base` column is Pipecat's median relative to jargo's, with the p-value of the
difference (`~` when it could be noise). A `vs base` of +900% is a ratio of 10.
benchstat needs at least six samples per side for an interval (fewer show
`± ∞`), which is why `BENCH_COUNT` defaults to 6.

To quote a comparison, quote it as a ratio of the medians together with both
confidence intervals, the number of samples, and the machine spec from the
config lines. In the shape of: "at depth 4, Pipecat's median time per hop was
R times jargo's (jargo X ± a%, Pipecat Y ± b%, n=6, p=P; 2 CPUs of an Intel
Core Ultra 7, governor performance, cgroup v2)". A single multiplier with no
interval and no machine is not a result.

## Next steps

Two kinds of measurement are planned and not yet in the suite: session density
with real-time audio (how many sessions, each fed 20 ms audio frames in real
time, a set of CPUs holds before frames arrive late), and an end-to-end run
against a mock AI server with scripted latencies, driven by a load generator
speaking the protobuf WebSocket protocol.
