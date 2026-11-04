"""Pipecat side of the jargo benchmark comparison.

Reads the same scenario files as jbench (benchmarks/scenarios/*.json), runs
the same kinds the same way, and prints the same Go benchmark format, so
benchstat can set the two side by side:

    python bench.py run [-count N] [-run REGEXP] scenarios/quick.json

A deployment benchmark starts `bench.py serve` processes of its own, one per
CPU, which hold the sessions and are driven over their standard input and
output in the protocol of benchmarks/internal/run/deploy.go.

Every kind mirrors benchmarks/internal/run: a chain of pass-through processors
in a PipelineWorker held by a WorkerRunner, frames seen arriving through
on_frame_reached_downstream, and the worker configured as on the Go side (no
turn tracking, no idle timeout, no RTVI, no observers, logging silenced). Keep
the two in step: a change to how one side measures is a change to both.

Standard library only, beyond pipecat-ai itself.
"""

import argparse
import asyncio
import ctypes
import gc
import json
import math
import os
import platform
import re
import sys
import time
from importlib.metadata import version as dist_version

# Pipecat logs through loguru, at debug level as frames go through. Removing the
# handlers before importing it keeps the logger off the measured path, as
# discarding the records does on the Go side.
from loguru import logger

logger.remove()

from pipecat.frames.frames import TextFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import PipelineWorker  # noqa: E402
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402
from pipecat.workers.runner import WorkerRunner  # noqa: E402

STOP_GRACE = 10.0  # seconds a session told to stop is waited for
TEXT = "x"  # the payload of every frame, as on the Go side
KINDS = ("latency", "throughput", "startup", "sessions", "deployment")
FIELDS = {"name", "kind", "depths", "depth", "sessions", "warmup", "frames",
          "rounds", "repeats", "budget_seconds"}
NAME = re.compile(r"^[A-Z][A-Za-z0-9]*$")


class BudgetExceeded(Exception):
    """A sample ran past its budget."""


class Passthrough(FrameProcessor):
    """Forwards every frame and does no work of its own."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class Session:
    """One bot, run the way an application runs one: a WorkerRunner holding a
    PipelineWorker whose pipeline is a chain of pass-through processors. Every
    TextFrame reaching the end is reported on the arrival queue."""

    def __init__(self, depth, arrived):
        self.worker = PipelineWorker(
            Pipeline([Passthrough() for _ in range(depth)]),
            enable_turn_tracking=False,
            enable_rtvi=False,
            idle_timeout_secs=None,
        )
        self.worker.set_reached_downstream_filter((TextFrame,))
        self.runner = WorkerRunner(handle_sigint=False)
        self.started = asyncio.Event()
        self.task = None

        @self.worker.event_handler("on_pipeline_started")
        async def on_started(_worker, _frame):
            self.started.set()

        @self.worker.event_handler("on_frame_reached_downstream")
        async def on_arrived(_worker, _frame):
            arrived.put_nowait(None)

    async def start(self):
        """Add the worker to the runner and run it in the background."""
        await self.runner.add_workers(self.worker)
        self.task = asyncio.create_task(self.runner.run())

    async def wait_started(self):
        """Wait for the StartFrame to have reached the end of the pipeline."""
        waiter = asyncio.ensure_future(self.started.wait())
        try:
            await asyncio.wait({waiter, self.task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if not self.started.is_set():
            raise RuntimeError("session finished before it started")

    async def send(self, frame):
        await self.worker.queue_frame(frame)

    async def stop(self):
        """End gracefully, and cancel if that takes longer than STOP_GRACE."""
        await self.worker.stop_when_done()
        if not await self._wait():
            await self.cancel()

    async def cancel(self):
        await self.runner.cancel(reason="benchmark budget exceeded")
        if not await self._wait():
            self.task.cancel()
            raise RuntimeError("session did not finish")

    async def _wait(self):
        done, _ = await asyncio.wait({self.task}, timeout=STOP_GRACE)
        return bool(done)


async def start_one(depth, arrived, group):
    s = Session(depth, arrived)
    group.append(s)
    await s.start()
    await s.wait_started()
    return s


async def abandon(group):
    """Cancel the sessions of a sample that ran out of time."""
    live = [s for s in group if s.task and not s.task.done()]
    await asyncio.gather(*(s.cancel() for s in live), return_exceptions=True)


def percentile(samples, p):
    """Nearest-rank percentile, as on the Go side. Sorts samples in place."""
    samples.sort()
    rank = math.ceil(p * len(samples))
    return samples[max(0, min(rank - 1, len(samples) - 1))]


async def latency(depth, warmup, frames, group):
    arrived = asyncio.Queue()
    s = await start_one(depth, arrived, group)
    for _ in range(warmup):
        await s.send(TextFrame(text=TEXT))
        await arrived.get()
    samples = []
    for _ in range(frames):
        frame = TextFrame(text=TEXT)
        start = time.perf_counter_ns()
        await s.send(frame)
        await arrived.get()
        samples.append(time.perf_counter_ns() - start)
    await s.stop()
    mean = sum(samples) / frames
    return frames, [
        (mean, "ns/frame"),
        (mean / depth, "ns/hop"),
        (percentile(samples, 0.50), "p50-ns/frame"),
        (percentile(samples, 0.99), "p99-ns/frame"),
    ]


async def throughput(depth, warmup, frames, group):
    arrived = asyncio.Queue()
    s = await start_one(depth, arrived, group)
    for _ in range(warmup):
        await s.send(TextFrame(text=TEXT))
        await arrived.get()
    batch = [TextFrame(text=TEXT) for _ in range(frames)]
    start = time.perf_counter_ns()
    for frame in batch:
        await s.send(frame)
    for _ in range(frames):
        await arrived.get()
    elapsed = time.perf_counter_ns() - start
    await s.stop()
    return frames, [(elapsed / frames, "ns/frame"), (frames / (elapsed / 1e9), "frames/s")]


async def startup(depth, repeats, group):
    build = run = 0
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        s = Session(depth, None)
        t1 = time.perf_counter_ns()
        group.append(s)
        await s.start()
        await s.wait_started()
        build += t1 - t0
        run += time.perf_counter_ns() - t1
        await s.stop()
    return repeats, [(run / repeats, "ns/start"), (build / repeats, "build-ns/start")]


def release_memory():
    """Collect garbage and hand freed memory back to the system, as
    debug.FreeOSMemory does on the Go side."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def resident_bytes():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


async def sessions(depth, count, warmup, rounds, group):
    release_memory()
    before = resident_bytes()
    arrived = asyncio.Queue()
    group.extend(Session(depth, arrived) for _ in range(count))

    start = time.perf_counter_ns()
    for s in group:
        await s.start()
    for s in group:
        await s.wait_started()
    started = time.perf_counter_ns() - start

    async def one_round():
        for s in group:
            await s.send(TextFrame(text=TEXT))
        for _ in group:
            await arrived.get()

    for _ in range(warmup):
        await one_round()
    start = time.perf_counter_ns()
    for _ in range(rounds):
        await one_round()
    elapsed = time.perf_counter_ns() - start

    release_memory()
    after = resident_bytes()
    await asyncio.gather(*(s.stop() for s in group))
    moved = rounds * count
    metrics = [
        (elapsed / rounds, "ns/round"),
        (elapsed / moved, "ns/frame"),
        (moved / (elapsed / 1e9), "frames/s"),
        (started / count, "startup-ns/session"),
    ]
    # Resident memory is read from /proc, so it is only reported where there is one.
    if before is not None and after is not None:
        metrics.append(((after - before) / count, "B/session"))
    return rounds, metrics


async def send_round(group, arrived):
    """Send one frame into every session of group and wait for all of them."""
    for s in group:
        await s.send(TextFrame(text=TEXT))
    for _ in group:
        await arrived.get()


async def serve():
    """A serving process of a deployment: holds the sessions the coordinator
    asks for, on commands read from standard input, one answer per line on
    standard output. The protocol is benchmarks/internal/run/deploy.go's."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    def answer(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    release_memory()
    answer("ready")
    group, arrived = [], asyncio.Queue()
    try:
        while line := await reader.readline():
            fields = line.decode().split()
            if not fields:
                continue
            command = fields[0]
            if command == "start":
                n, depth = int(fields[1]), int(fields[2])
                group.extend(Session(depth, arrived) for _ in range(n))
                for s in group:
                    await s.start()
                for s in group:
                    await s.wait_started()
                answer("started")
            elif command == "round":
                await send_round(group, arrived)
                answer("done")
            elif command == "settle":
                release_memory()
                answer("settled")
            elif command == "stop":
                await asyncio.gather(*(s.stop() for s in group))
                answer("stopped")
                return
            else:
                raise ValueError(f"unknown command {command!r}")
    except Exception as exc:  # noqa: BLE001
        answer(f"error {exc!r}")
        await abandon(group)
        raise


def proportional_bytes(pid):
    """A process's proportional set size, from /proc/<pid>/smaps_rollup."""
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no Pss line in smaps_rollup")


class Server:
    """A serving process as the coordinator sees it."""

    def __init__(self, proc):
        self.proc = proc

    async def call(self, command, want):
        if command:
            self.proc.stdin.write((command + "\n").encode())
            await self.proc.stdin.drain()
        line = (await self.proc.stdout.readline()).decode().strip()
        if line.startswith("error "):
            raise RuntimeError(f"server failed: {line[6:]}")
        if line != want:
            raise RuntimeError(f"deployment protocol: got {line!r}, want {want!r}")


async def deployment(depth, count, warmup, rounds, servers):
    """Serve count sessions the way Pipecat is deployed to use every CPU it is
    given: a process per CPU, since the global interpreter lock holds each to
    one core. Mirrors run.Deployment: the same metrics, read the same way, with
    the coordinator (this process) not counted."""
    processes = len(os.sched_getaffinity(0))
    for _ in range(processes):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", os.path.abspath(__file__), "serve",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        servers.append(Server(proc))

    async def every(make):
        await asyncio.gather(*(s.call(*make(i)) for i, s in enumerate(servers)))

    def pss():
        return sum(proportional_bytes(s.proc.pid) for s in servers)

    await every(lambda i: ("", "ready"))
    baseline = pss()

    def share(i):
        return count // processes + (1 if i < count % processes else 0)

    start = time.perf_counter_ns()
    await every(lambda i: (f"start {share(i)} {depth}", "started"))
    started = time.perf_counter_ns() - start

    for _ in range(warmup):
        await every(lambda i: ("round", "done"))
    start = time.perf_counter_ns()
    for _ in range(rounds):
        await every(lambda i: ("round", "done"))
    elapsed = time.perf_counter_ns() - start

    await every(lambda i: ("settle", "settled"))
    total = pss()
    await every(lambda i: ("stop", "stopped"))
    moved = rounds * count
    return rounds, [
        (elapsed / rounds, "ns/round"),
        (elapsed / moved, "ns/frame"),
        (moved / (elapsed / 1e9), "frames/s"),
        (started / count, "startup-ns/session"),
        (baseline, "baseline-B"),
        (total, "B/deployment"),
        ((total - baseline) / count, "B/session"),
    ]


async def stop_servers(servers, kill):
    """Close the servers' input, kill them when told to, and reap them."""
    for s in servers:
        if s.proc.stdin and not s.proc.stdin.is_closing():
            s.proc.stdin.close()
        if kill and s.proc.returncode is None:
            s.proc.kill()
    await asyncio.gather(*(s.proc.wait() for s in servers), return_exceptions=True)


async def sample(bench, size):
    """Take one sample within the benchmark's budget. Every session a kind
    starts goes into group, so the ones still running can be canceled when the
    budget runs out."""
    group, servers, ok = [], [], False
    try:
        async with asyncio.timeout(bench["budget_seconds"]):
            kind, warmup = bench["kind"], bench.get("warmup", 0)
            if kind == "deployment":
                result = await deployment(bench["depth"], size, warmup, bench["rounds"], servers)
                ok = True
                return result
            if kind == "latency":
                return await latency(size, warmup, bench["frames"], group)
            if kind == "throughput":
                return await throughput(size, warmup, bench["frames"], group)
            if kind == "startup":
                return await startup(size, bench["repeats"], group)
            return await sessions(bench["depth"], size, warmup, bench["rounds"], group)
    except TimeoutError:
        await abandon(group)
        raise BudgetExceeded from None
    finally:
        # A sample that failed may have left a server mid-command, so it is
        # killed rather than waited for.
        await stop_servers(servers, not ok)


def validate(sc):
    """Mirror of internal/scenario's checks, so both sides refuse the same files."""
    if not sc.get("name") or not sc.get("benchmarks"):
        raise ValueError("scenario needs a name and benchmarks")
    seen = set()
    for b in sc["benchmarks"]:
        name = b.get("name", "")
        if unknown := set(b) - FIELDS:
            raise ValueError(f"benchmark {name!r}: unknown fields {sorted(unknown)}")
        if name in seen or not NAME.match(name):
            raise ValueError(f"benchmark {name!r}: duplicate or malformed name")
        seen.add(name)
        if b.get("kind") not in KINDS:
            raise ValueError(f"benchmark {name!r}: unknown kind {b.get('kind')!r}")
        if b.get("budget_seconds", 0) <= 0 or b.get("warmup", 0) < 0:
            raise ValueError(f"benchmark {name!r}: bad budget_seconds or warmup")
        multi = b["kind"] in ("sessions", "deployment")
        sizes = b.get("sessions" if multi else "depths") or []
        if not sizes or min(sizes) < 1:
            raise ValueError(f"benchmark {name!r}: sizes must be a non-empty list of positive numbers")
        need = {"latency": "frames", "throughput": "frames", "startup": "repeats",
                "sessions": "rounds", "deployment": "rounds"}[b["kind"]]
        if b.get(need, 0) < 1 or (multi and b.get("depth", 0) < 1):
            raise ValueError(f"benchmark {name!r}: {need} (and depth for sessions) must be at least 1")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    return value.strip()
    except OSError:
        pass
    return "unknown"


def read_value(path):
    """The trimmed content of a one-line file, or "unknown"."""
    try:
        with open(path) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"


def cpus_allowed():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    return line.partition(":")[2].strip()
    except OSError:
        pass
    return "unknown"


def governor(allowed):
    first = re.match(r"\d+", allowed)
    if not first:
        return "unknown"
    return read_value(f"/sys/devices/system/cpu/cpu{first.group()}/cpufreq/scaling_governor")


def config_lines(scenario_name):
    """The same keys jbench prints, read the same way."""
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    allowed = cpus_allowed()
    cgroup = "v2" if os.path.exists("/sys/fs/cgroup/cgroup.controllers") else "v1"
    return [
        ("framework", "pipecat"),
        ("framework_version", dist_version("pipecat-ai")),
        ("runtime", f"{platform.python_implementation().lower()}{platform.python_version()}"),
        ("goos", sys.platform),
        ("goarch", arch),
        ("cpu", cpu_model()),
        ("cpus", str(len(os.sched_getaffinity(0)))),
        ("cpus_allowed", allowed),
        ("governor", governor(allowed)),
        ("kernel", read_value("/proc/sys/kernel/osrelease")),
        ("cgroup", cgroup),
        ("memory_limit", read_value("/sys/fs/cgroup/memory.max")),
        ("scenario", scenario_name),
    ]


def units(sc, pattern):
    out = []
    for g, b in enumerate(sc["benchmarks"]):
        multi = b["kind"] in ("sessions", "deployment")
        label, sizes = ("sessions", b["sessions"]) if multi else ("depth", b["depths"])
        for i, n in enumerate(sizes):
            name = f"{b['name']}/{label}={n}"
            if pattern.search(name):
                out.append((b, g, i, name, n))
    return out


async def execute(sc, count, pattern):
    """Interleave the samples, every sub-benchmark once per pass, as jbench does."""
    todo = units(sc, pattern)
    if not todo:
        raise SystemExit(f"bench.py: -run {pattern.pattern!r} matches no sub-benchmark")
    skip_from, failed = {}, False
    for _ in range(count):
        for b, g, i, name, size in todo:
            if g in skip_from and i >= skip_from[g]:
                continue
            try:
                n, metrics = await sample(b, size)
            except BudgetExceeded:
                print(f"--- TIMEOUT: Benchmark{name} exceeded its {b['budget_seconds']:g}s budget; "
                      "larger sizes skipped", flush=True)
                skip_from[g] = i
                continue
            except Exception as exc:  # noqa: BLE001
                # Any other failure is reported, and the run goes on.
                print(f"--- FAIL: Benchmark{name}: {exc!r}", flush=True)
                skip_from[g], failed = i, True
                continue
            line = "".join(f"\t{v:.2f} {unit}" for v, unit in metrics)
            print(f"Benchmark{name}\t{n}{line}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(prog="bench.py")
    if sys.argv[1:] == ["serve"]:
        asyncio.run(serve())
        return
    parser.add_argument("command", choices=["run"])
    parser.add_argument("-count", type=int, default=1, help="samples of each sub-benchmark")
    parser.add_argument("-run", default="", help="only sub-benchmarks whose full name matches")
    parser.add_argument("scenario")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("-count must be at least 1")
    with open(args.scenario) as f:
        sc = json.load(f)
    validate(sc)
    for key, value in config_lines(sc["name"]):
        print(f"{key}: {value}")
    failed = asyncio.run(execute(sc, args.count, re.compile(args.run)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
