# Quickstart

The example bots live in [`examples/`](../examples):

- **echo** — speak into the browser, hear yourself back. No API keys.
- **voicebot** — the full voice agent: STT → LLM → TTS with turn-taking and
  barge-in, plus long-term memory and tracing.
- **voice/** — one headless backend per provider, each wiring its STT/LLM/TTS
  explicitly in Go and exposing only the WebRTC `/offer` endpoint
  (`go run ./examples/voice/<provider>`); drive it from a browser client.

Run them with Docker (no host setup) or with a local Go toolchain.

## Run with Docker

jargo publishes a build base and a distroless runtime base image, so you can
containerise a bot without installing the native (cgo) dependencies on the host.
The **[Deploy with Docker](deploy-with-docker.md)** guide has a copyable
two-stage Dockerfile for the example bots and the run command
(`-e DEEPGRAM_API_KEY=…` etc., then open <http://localhost:8080>).

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

The voicebot runs a fixed Deepgram + Anthropic + ElevenLabs stack. To try a
different provider, run one of the per-provider examples under
[`examples/voice`](../examples/voice) — one self-contained file each, with the
provider wired explicitly in Go:

```sh
go run ./examples/voice/cartesia     # Deepgram STT, Anthropic LLM, Cartesia TTS
go run ./examples/voice/openai       # OpenAI STT + LLM + TTS
go run ./examples/voice/groq         # Groq STT + LLM, ElevenLabs TTS
```

These are **headless backends**: they expose the WebRTC `/offer` endpoint and no
web UI. Point a browser client at `http://localhost:8080` — the `nextjs-voicebot`
example in [jargo-client-react](https://github.com/gojargo/jargo-client-react),
with `NEXT_PUBLIC_JARGO_URL=http://localhost:8080`. Each example's doc comment
lists the API keys it needs. See [Providers](../README.md#providers) for the full
list, and [Turn-taking](turn-taking.md) for tuning end-of-turn detection and
barge-in.
