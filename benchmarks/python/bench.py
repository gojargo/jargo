"""Pipecat-side micro-benchmarks, for comparison against jargo's Go benchmarks.

These time the CPU-bound, locally-computed pieces of turn-taking that both
projects implement, so the comparison is apples-to-apples:

  - Whisper log-mel feature extraction (the fair language comparison: numpy
    vs Go, same math).
  - Silero VAD inference per frame (shared ONNX runtime; measures glue).
  - Smart Turn end-to-end: features + ONNX inference.

The same ONNX model files jargo embeds are loaded here, so only the surrounding
code differs. Run jargo's side with:

    JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so \\
        go test -run '^$' -bench 'ComputeLogMel|Silero|SmartTurnPredict' \\
        ./audio/turn/ ./audio/vad/

Usage:

    pip install -r requirements.txt
    PIPECAT_SRC=/path/to/pipecat python bench.py

PIPECAT_SRC points at a Pipecat checkout (its repository root or src/ dir); it
is used only for the feature-extraction module. If unset, that one benchmark is
skipped.
"""

import importlib.util
import os
import time

import numpy as np
import onnxruntime as rt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SILERO_MODEL = os.path.join(REPO_ROOT, "audio", "vad", "silero_vad.onnx")
SMART_TURN_MODEL = os.path.join(REPO_ROOT, "audio", "turn", "smart-turn-v3.2-cpu.onnx")

SAMPLE_RATE = 16000


def gen_audio(n: int) -> np.ndarray:
    """The same deterministic signal jargo's benchmarks use."""
    i = np.arange(n)
    return (
        0.3 * np.sin(2 * np.pi * 150 * i / SAMPLE_RATE)
        + 0.2 * np.sin(2 * np.pi * 440 * i / SAMPLE_RATE)
        + 0.1 * np.sin(2 * np.pi * 900 * i / SAMPLE_RATE)
    ).astype(np.float32)


def bench(fn, iters: int, warmup: int) -> float:
    """Return mean milliseconds per call."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1000


def load_whisper_features():
    """Load Pipecat's feature extractor by file path, or return None."""
    src = os.environ.get("PIPECAT_SRC")
    if not src:
        return None
    for candidate in (
        os.path.join(src, "src", "pipecat", "audio", "turn", "smart_turn", "_whisper_features.py"),
        os.path.join(src, "pipecat", "audio", "turn", "smart_turn", "_whisper_features.py"),
    ):
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("wf", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def bench_features(results):
    wf = load_whisper_features()
    if wf is None:
        print("features:    SKIP (set PIPECAT_SRC to a Pipecat checkout)")
        return
    audio = gen_audio(32000)
    ms = bench(lambda: wf.compute_whisper_log_mel_features(audio, do_normalize=True), iters=50, warmup=5)
    results["features (2s utterance)"] = ms
    print(f"features:    {ms:8.3f} ms")


def bench_silero(results):
    session = rt.InferenceSession(SILERO_MODEL, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, 64), np.float32)
    frame = (0.5 * np.sin(2 * np.pi * 300 * np.arange(512) / SAMPLE_RATE) * 32767).astype(np.int16)

    def once():
        nonlocal state, context
        x = (frame.astype(np.float32) / 32768.0).reshape(1, -1)
        x = np.concatenate((context, x), axis=1)
        out = session.run(None, {"input": x, "state": state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        state = out[1]
        context = x[:, -64:]

    ms = bench(once, iters=500, warmup=20)
    results["silero (per 32ms frame)"] = ms
    print(f"silero:      {ms:8.3f} ms")


def bench_smart_turn(results):
    wf = load_whisper_features()
    if wf is None:
        print("smart turn:  SKIP (set PIPECAT_SRC to a Pipecat checkout)")
        return
    session = rt.InferenceSession(SMART_TURN_MODEL, providers=["CPUExecutionProvider"])
    audio = gen_audio(32000)

    def once():
        log_mel = wf.compute_whisper_log_mel_features(audio, do_normalize=True)
        session.run(None, {"input_features": np.expand_dims(log_mel, 0)})

    ms = bench(once, iters=50, warmup=5)
    results["smart turn (end-to-end)"] = ms
    print(f"smart turn:  {ms:8.3f} ms")


def main():
    print(f"numpy {np.__version__}, onnxruntime {rt.__version__}\n")
    results: dict[str, float] = {}
    bench_features(results)
    bench_silero(results)
    bench_smart_turn(results)


if __name__ == "__main__":
    main()
