# Quickstart

Two example bots live in [`examples/`](../examples):

- **echo** — speak into the browser, hear yourself back. No API keys.
- **voicebot** — a full voice agent: STT → LLM → TTS, with turn-taking and barge-in.

Run them the easy way (the container image, no host setup) or with a local Go
toolchain.

## Run with Docker (recommended)

The image bundles both native dependencies — libsoxr and the ONNX Runtime — so
there is nothing to install on the host.

```sh
docker build -t jargo-voicebot .
docker run --rm -p 8080:8080 \
  -e DEEPGRAM_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  -e ELEVENLABS_API_KEY=... \
  jargo-voicebot
```

Then open <http://localhost:8080>, click start, and allow the microphone.

Build the echo bot instead with `--build-arg EXAMPLE=echo` (it needs no keys).
Multi-arch images build with buildx:

```sh
docker buildx build --platform linux/amd64,linux/arm64 -t jargo-voicebot .
```

## Run locally

### Prerequisites

jargo uses cgo and two native libraries:

- **libsoxr** — high-quality audio resampling (linked at build time).
- **ONNX Runtime** — VAD and turn detection (loaded at run time).

```sh
# Debian/Ubuntu
sudo apt-get install -y libsoxr-dev      # libsoxr0 at run time

# ONNX Runtime: download the shared library and point jargo at it
export JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so
```

Get the ONNX Runtime library from the
[onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) — the
`onnxruntime-linux-*` archive contains `lib/libonnxruntime.so`.

### Echo bot — no keys

```sh
go run ./examples/echo           # then open http://localhost:8080
```

### Voice bot

Set the provider API keys, then run:

```sh
export DEEPGRAM_API_KEY=...       # STT
export ANTHROPIC_API_KEY=...      # LLM
export ELEVENLABS_API_KEY=...     # TTS
go run ./examples/voicebot        # then open http://localhost:8080
```

Pick different providers per stage with the `STT`, `LLM` and `TTS` env vars:

```sh
STT=assemblyai LLM=openai TTS=cartesia go run ./examples/voicebot
```

See [Services](../README.md#services) for the full provider list, and
[Turn-taking](turn-taking.md) for tuning end-of-turn detection and barge-in.
