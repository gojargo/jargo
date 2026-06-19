# Benchmarks

Micro-benchmarks comparing jargo (Go) with [Pipecat](https://github.com/pipecat-ai/pipecat)
(Python). Two families:

1. **Turn-taking** — the CPU-bound pieces both implement locally (Whisper
   features, Silero VAD, Smart Turn). Model-backed work routes through the same
   ONNX runtime, so it measures glue, not language.
2. **Pipeline plumbing & concurrency** — the frame-transport architecture, where
   jargo (goroutines) and Pipecat (one asyncio event loop) genuinely differ.
   This is where a language/runtime comparison is meaningful.

See [`docs/benchmarks.md`](../docs/benchmarks.md) for methodology, what is and
isn't comparable, and a results discussion.

## Pipeline plumbing & concurrency

```sh
# jargo (Go) — frame latency through a processor chain, and throughput scaling
go test -run '^$' -bench 'FramePlumbing|ConcurrentSessions' -benchmem ./pipeline/

# Pipecat (Python) — the matching harness (needs pipecat-ai installed)
cd benchmarks/python && pip install -r requirements.txt && pip install pipecat-ai
python bench_pipeline.py
```

`BenchmarkFramePlumbing` reports `ns/hop` (per-processor latency); `bench_pipeline.py`
reports the same. `BenchmarkConcurrentSessions` reports aggregate `frames/s` as
the number of simultaneous pipelines scales (1 → 1000).

## Turn-taking

The Go benchmarks live next to the code they measure. They need the ONNX
runtime for the model-backed ones (see [`docs/turn-taking.md`](../docs/turn-taking.md)):

```sh
# jargo (Go)
JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so \
  go test -run '^$' -bench 'ComputeLogMel|Silero|SmartTurnPredict' -benchmem \
  ./audio/turn/ ./audio/vad/

# Pipecat (Python) — loads the same ONNX model files jargo embeds
cd benchmarks/python && pip install -r requirements.txt
PIPECAT_SRC=/path/to/pipecat python bench.py
```

`PIPECAT_SRC` points at a Pipecat checkout; it is used only to time Pipecat's
own feature-extraction code. The Silero benchmark needs no checkout.
