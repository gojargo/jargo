"""Pipecat-side pipeline plumbing & concurrency benchmarks, the counterpart to
jargo's pipeline/bench_test.go.

Unlike the turn-taking benchmarks (which route through the same ONNX runtime on
both sides and so measure glue, not language), these measure the frame-transport
architecture itself, where jargo (goroutines) and Pipecat (a single asyncio
event loop) genuinely differ:

  - plumbing:    latency of one frame through a chain of N pass-through
                 processors (Go: BenchmarkFramePlumbing).
  - concurrency: aggregate frame throughput as the number of simultaneous
                 pipelines scales (Go: BenchmarkConcurrentSessions).

Each pass-through does no work of its own, so the numbers are the cost of the
plumbing. Run jargo's side with:

    go test -run '^$' -bench 'FramePlumbing|ConcurrentSessions' -benchmem ./pipeline/

Usage:

    pip install -r requirements.txt   # plus: pip install pipecat-ai
    python bench_pipeline.py

This targets a recent pipecat-ai. The FrameProcessor/PipelineTask API has
changed across releases; if import or run fails, pin the version you compare
against and note it next to the numbers.
"""

import asyncio
import time

try:
    from loguru import logger
    from pipecat.frames.frames import EndFrame, TextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameProcessor

    # Pipecat logs every frame at DEBUG; silence it so the timing measures the
    # pipeline, not the logger (and the output stays readable).
    logger.remove()

    HAVE_PIPECAT = True
except ImportError as exc:  # pragma: no cover - scaffold guard
    HAVE_PIPECAT = False
    _IMPORT_ERROR = exc
    FrameProcessor = object  # so the class below still defines when pipecat is absent


class Passthrough(FrameProcessor):
    """A data-forwarding processor with no work of its own — the Python twin of
    jargo's passthrough. Isolates per-hop plumbing cost from real processing."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def build_task(depth: int) -> "PipelineTask":
    """A PipelineTask whose pipeline is `depth` pass-through processors."""
    pipeline = Pipeline([Passthrough() for _ in range(depth)])
    return PipelineTask(pipeline)


async def run_with_frames(task: "PipelineTask", n_frames: int) -> None:
    """Queue n_frames text frames plus an EndFrame and run the task to completion."""
    runner = PipelineRunner(handle_sigint=False)
    for _ in range(n_frames):
        await task.queue_frame(TextFrame("x"))
    await task.queue_frame(EndFrame())
    await runner.run(task)


async def bench_plumbing(results: dict) -> None:
    """Mean per-frame latency through a chain, at several depths. Reported as
    ns/hop to line up with the Go benchmark's ns/hop metric."""
    n_frames = 2000
    for depth in (1, 4, 16, 64):
        task = build_task(depth)
        start = time.perf_counter()
        await run_with_frames(task, n_frames)
        elapsed = time.perf_counter() - start
        ns_per_hop = elapsed / n_frames / depth * 1e9
        results[f"plumbing depth={depth}"] = ns_per_hop
        print(f"plumbing depth={depth:<3}  {ns_per_hop:10.0f} ns/hop")


async def bench_concurrency(results: dict) -> None:
    """Aggregate frame throughput as simultaneous pipelines scale. All tasks run
    on the one event loop via gather; this is the ceiling jargo's goroutines are
    meant to beat across cores."""
    depth = 4
    n_frames = 200
    # 1000 sessions on one asyncio loop does not finish in any reasonable time —
    # that ceiling is itself the result; cap at 100 so the run completes.
    for sessions in (1, 10, 100):
        tasks = [build_task(depth) for _ in range(sessions)]
        start = time.perf_counter()
        await asyncio.gather(*(run_with_frames(t, n_frames) for t in tasks))
        elapsed = time.perf_counter() - start
        rate = sessions * n_frames / elapsed
        results[f"concurrency sessions={sessions}"] = rate
        print(f"concurrency sessions={sessions:<4} {rate:12.0f} frames/s")


async def main() -> None:
    if not HAVE_PIPECAT:
        print(f"SKIP: pipecat-ai not importable ({_IMPORT_ERROR}); pip install pipecat-ai")
        return
    results: dict = {}
    await bench_plumbing(results)
    print()
    await bench_concurrency(results)


if __name__ == "__main__":
    asyncio.run(main())
