---
title: Installation
weight: 1
---

# Installation

```sh
go get github.com/gojargo/jargo
```

jargo needs Go 1.26 or newer.

## The default build needs no C toolchain

```sh
CGO_ENABLED=0 go build ./...
```

That works out of the box. Opus encode/decode and resampling are **pure Go**, and
the two native runtimes are bound through
[purego](https://github.com/ebitengine/purego), located and loaded at *run* time
from their shared libraries, with nothing required at build time.

This is the property that makes jargo deployable as a single static binary. Keep
it unless you have a measured reason not to.

## Runtime libraries

| Library | Needed for | Env var | Without it |
|---|---|---|---|
| **ONNX Runtime** | Silero VAD + Smart Turn v3 | `JARGO_ONNXRUNTIME_LIB` | Bot still runs; falls back to STT endpointing, loses barge-in. |
| **RNNoise** | Optional input noise reduction | `JARGO_RNNOISE_LIB` | Bot runs without denoising. |

Both are located by their conventional name on the loader's default search path
when the variable is unset.

### ONNX Runtime

Download a build for your platform from the
[onnxruntime releases](https://github.com/microsoft/onnxruntime/releases); the
`onnxruntime-linux-*` archive contains `lib/libonnxruntime.so`:

```sh
export JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so     # Linux
export JARGO_ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib  # macOS
set JARGO_ONNXRUNTIME_LIB=C:\path\to\onnxruntime.dll        # Windows
```

The VAD and turn models themselves are **embedded in the binary** with
`go:embed`, so there is nothing else to download or locate.

### RNNoise

```sh
sudo apt-get install -y librnnoise0     # or build from source
export JARGO_RNNOISE_LIB=/path/to/librnnoise.so
```

Enable it per transport rather than globally:

```go
params := transport.DefaultParams()
if filter, err := rnnoise.New(); err == nil {
    params.AudioInFilter = filter
}
```

`rnnoise.New` returns an error when the library is missing, which is the
idiomatic way to make denoising optional. See `examples/voice/openai`.

## Optional cgo build tags

Two higher-quality C backends are available. Both are **opt-in** and both need a C
toolchain; the pure-Go defaults are used otherwise.

| Tag | Replaces | Install |
|---|---|---|
| `libsoxr` | Pure-Go resampler with the SoX Resampler (highest quality) | `sudo apt-get install -y libsoxr-dev` |
| `libopus` | Pure-Go SILK encoder with the C Opus encoder (better speech) | `sudo apt-get install -y libopus-dev` |

```sh
go build -tags libsoxr ./...
go build -tags libsoxr,libopus ./...
```

These are the only cgo in the tree.

## Docker

The published base images bundle all of the above, so you do not have to think
about any of it:

- **`gojargo/jargo-build`**: build stage, with the toolchain and C headers.
- **`gojargo/jargo`**: distroless runtime, carrying the ONNX Runtime, RNNoise,
  libsoxr, libgomp and libopus.

See **[Deploy with Docker](../deploy/docker.md)** for a copyable Dockerfile.

## Verify

```sh
go run ./examples/echo    # then open http://localhost:8080
```

The echo bot needs no API keys and no ONNX Runtime. If you hear yourself back,
the audio path works. Then continue to
**[Your first bot](your-first-bot.md)**.
