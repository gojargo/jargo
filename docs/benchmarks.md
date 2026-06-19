# Benchmarks: jargo vs Pipecat

jargo is a Go port of [Pipecat](https://github.com/pipecat-ai/pipecat). A natural
question is how the Go implementation compares to the Python original. This page
sets out what is fair to compare, how to reproduce it, and what the numbers say.

> These are micro-benchmarks of individual components, not end-to-end voice
> latency. Real conversational latency is dominated by the network round-trips to
> the STT, LLM and TTS services, which are identical for both projects.

## What is comparable

Most of a voice agent's work is I/O to remote services, where the language makes
no difference. The locally-computed, CPU-bound work in turn-taking is where a
language comparison is meaningful — and only some of it is a true
language-to-language comparison:

| Component | Comparison | Notes |
| --- | --- | --- |
| Whisper log-mel feature extraction | **Fair language comparison** | Same math, jargo in Go vs Pipecat in numpy. |
| Smart Turn inference | Shared runtime | Both call the same ONNX Runtime (C++); measures glue, not language. |
| Silero VAD inference | Shared runtime | Same as above, per audio frame. |

Both sides load the **same ONNX model files** (the ones jargo embeds), so the
models are identical and only the surrounding code differs.

## Running them

See [`../benchmarks/README.md`](../benchmarks/README.md). In short:

```sh
# jargo (Go)
JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so \
  go test -run '^$' -bench 'ComputeLogMel|Silero|SmartTurnPredict' -benchmem \
  ./audio/turn/ ./audio/vad/

# Pipecat (Python)
cd benchmarks/python && pip install -r requirements.txt
PIPECAT_SRC=/path/to/pipecat python bench.py
```

## Results

Indicative figures from one 16-core x86-64 Linux machine, ONNX Runtime 1.26,
numpy 2.x. Micro-benchmarks are noisy; treat these as orders of magnitude, not
exact ratios, and re-run on your own hardware.

| Component | Pipecat (Python) | jargo (Go) |
| --- | ---: | ---: |
| Whisper log-mel features (2 s utterance) | ~5–10 ms | ~44 ms |
| Silero VAD (per 32 ms frame) | ~0.15 ms | ~0.15 ms |
| Smart Turn, end-to-end (features + ONNX) | ~105–120 ms | ~100 ms |

### Reading the numbers

- **Silero VAD: a tie.** Both bind the same ONNX Runtime, and the per-frame glue
  is negligible. This is the common case for anything model-backed: jargo neither
  wins nor loses on raw inference, because it is the same C++ engine.

- **Smart Turn end-to-end: comparable.** A single end-of-turn decision is
  dominated by the ONNX inference (~50–100 ms), which is runtime-bound and
  sensitive to thread settings on both sides. The feature-extraction difference
  is small relative to the model.

- **Feature extraction: jargo is currently slower.** numpy's `rfft` is an
  `O(N log N)` FFT with vectorized SIMD; jargo's feature extractor uses a naive
  `O(N²)` DFT, so it trails by roughly an order of magnitude. A constant-frame
  fast path keeps the silence-padded part of the 8-second window cheap, but the
  speech frames still pay the DFT cost. **Replacing the DFT with a real FFT is
  the top performance item for the turn package.**

## So why port to Go at all?

Raw single-threaded compute is not where a Go port pays off — for the
model-backed work, jargo and Pipecat call the same runtime. The wins are
operational, and they matter for shipping and running a voice agent:

- **One static binary.** No Python, no virtualenv, no system packages to match
  at deploy time. `scp` it and run it; container images are tiny.
- **Low, predictable memory.** No interpreter heap or numpy/torch resident set;
  a jargo agent's footprint is dominated by the ONNX model, not the runtime.
- **Fast startup.** No interpreter or import graph to warm up.
- **True concurrency.** Goroutines handle many simultaneous sessions on one
  process without a GIL or per-worker process pool.

jargo trades numpy's tuned numerics — recoverable with a better FFT — for these
deployment and concurrency properties.
