# Benchmarks: jargo vs Pipecat

jargo is a Go port of [Pipecat](https://github.com/pipecat-ai/pipecat). A natural
question is how the Go implementation compares to the Python original. This page
sets out what is fair to compare, how to reproduce it, and what the numbers say.

There are two families:

1. **Turn-taking** — the CPU-bound work both projects compute locally (Whisper
   features, Silero VAD, Smart Turn). Model-backed work routes through the same
   ONNX Runtime on both sides, so it mostly measures glue; only the feature
   extraction is a true language-to-language comparison.
2. **Pipeline plumbing & concurrency** — the frame-transport architecture, where
   jargo (goroutines) and Pipecat (one asyncio event loop) genuinely differ.
   This is where the runtime comparison is most meaningful.

> Real end-to-end conversational latency is dominated by network round-trips to
> the STT, LLM and TTS services, which are identical for both projects. These are
> component micro-benchmarks, not end-to-end latency.

## Running them

See [`../benchmarks/README.md`](../benchmarks/README.md). In short:

```sh
# jargo (Go) — turn-taking
JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so \
  go test -run '^$' -bench 'ComputeLogMel|Silero|SmartTurnPredict' -benchmem \
  ./audio/turn/ ./audio/vad/

# jargo (Go) — pipeline plumbing & concurrency
go test -run '^$' -bench 'FramePlumbing|ConcurrentSessions' -benchmem ./pipeline/

# Pipecat (Python)
cd benchmarks/python && pip install -r requirements.txt && pip install pipecat-ai
PIPECAT_SRC=/path/to/pipecat python bench.py            # turn-taking
python -u bench_pipeline.py                             # pipeline & concurrency
```

## Results

Indicative figures from one 16-core x86-64 Linux machine (Intel Ultra 7 255H),
ONNX Runtime 1.26, numpy 2.4. Micro-benchmarks are noisy; treat these as orders
of magnitude and re-run on your own hardware.

### Turn-taking

| Component | Pipecat (Python) | jargo (Go) | |
| --- | ---: | ---: | --- |
| Whisper log-mel features (2 s utterance) | ~41 ms | ~30 ms | **jargo ~1.4× faster** |
| Silero VAD (per 32 ms frame) | ~0.63 ms | ~0.73 ms | tie |
| Smart Turn, end-to-end (features + ONNX) | ~210 ms | ~222 ms | tie |

- **Feature extraction — jargo wins.** This is the one true language comparison:
  the same math, jargo in Go vs Pipecat in numpy. jargo computes the 400-point
  real DFT with [gonum](https://gonum.org)'s FFT, projects onto a **sparse**
  mel filterbank (each triangular filter touches ~8 of 201 bins, so the dense
  `×0` multiplies are skipped), and stores the filterbank mel-major so the
  projection's inner loop is contiguous. Those three changes took the extractor
  from ~195 ms (a naive `O(N²)` DFT with a dense mel matmul) to ~30 ms — faster
  than numpy's `WhisperFeatureExtractor` on the same machine.

- **Silero VAD — a tie.** Both bind the same ONNX Runtime; the per-frame glue is
  negligible. This is the common case for anything model-backed: jargo neither
  wins nor loses on raw inference, because it is the same C++ engine.

- **Smart Turn end-to-end — a tie.** A single end-of-turn decision is dominated
  by the ONNX inference (~190 ms here), which is runtime-bound on both sides. The
  feature-extraction win is small relative to the model.

### Pipeline plumbing & concurrency

Frame transport through a chain of pass-through processors with no work of their
own — pure framework overhead.

**Per-hop latency** (one frame through one processor):

| Chain depth | Pipecat (Python) | jargo (Go) |
| --- | ---: | ---: |
| 1 | ~547 µs/hop | ~14 µs/hop |
| 4 | ~235 µs/hop | ~8.8 µs/hop |
| 16 | ~134 µs/hop | ~8.2 µs/hop |
| 64 | ~109 µs/hop | ~7.5 µs/hop |

jargo's per-hop cost is goroutine wakeup latency; Pipecat's is an `await`
through each processor's asyncio queue plus its worker bookkeeping. jargo is
~15–40× faster per hop.

**Aggregate throughput** as simultaneous sessions scale (4-processor chain):

| Sessions | Pipecat (Python) | jargo (Go) |
| --- | ---: | ---: |
| 1 | ~1,100 frames/s | ~26,000 frames/s |
| 10 | ~1,000 frames/s | ~41,000 frames/s |
| 100 | ~1,000 frames/s | ~92,000 frames/s |
| 1000 | *did not finish* | ~123,000 frames/s |

This is the headline. Pipecat's throughput is **flat** — one asyncio event loop
runs on one core, so adding sessions adds no capacity (it slightly *declines*
under scheduling overhead, and 1000 sessions did not complete in 280 s). jargo's
throughput **rises** with sessions as goroutines spread across cores: ~92× more
aggregate throughput at 100 sessions, and it keeps scaling to 1000.

## What this means for a voice agent

The old assumption — "Go can't beat numpy, so the wins are only operational" — is
wrong on both counts:

- jargo now **wins the one fair compute comparison** (feature extraction), after
  replacing a naive DFT and a dense mel projection with an FFT and a sparse,
  cache-friendly filterbank.
- The decisive win is **concurrency**: a single jargo process scales many
  simultaneous voice sessions across cores, where a Pipecat process is capped by
  one event loop and the GIL. For a server hosting many concurrent calls, that is
  the difference between one box and many.

The operational properties still hold and still matter: low, predictable memory
(no interpreter/numpy/torch resident set — the footprint is dominated by the ONNX
model), fast startup (no import graph), and a deployment that is a Go binary plus
its native libraries rather than a Python environment to reconcile.
