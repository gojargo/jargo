# Benchmarks

Micro-benchmarks comparing jargo (Go) with [Pipecat](https://github.com/pipecat-ai/pipecat)
(Python) on the parts of turn-taking that both implement locally and on the
CPU, so the comparison is fair. See [`docs/benchmarks.md`](../docs/benchmarks.md)
for methodology, what is and isn't comparable, and a results discussion.

## jargo (Go)

The Go benchmarks live next to the code they measure. They need the ONNX
runtime for the model-backed ones (see [`docs/turn-taking.md`](../docs/turn-taking.md)):

```sh
JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so \
  go test -run '^$' -bench 'ComputeLogMel|Silero|SmartTurnPredict' -benchmem \
  ./audio/turn/ ./audio/vad/
```

## Pipecat (Python)

The Python side loads the same ONNX model files jargo embeds, so only the
surrounding code differs.

```sh
cd benchmarks/python
pip install -r requirements.txt
PIPECAT_SRC=/path/to/pipecat python bench.py
```

`PIPECAT_SRC` points at a Pipecat checkout; it is used only to time Pipecat's
own feature-extraction code. The Silero benchmark needs no checkout.
