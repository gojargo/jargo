---
title: Installation
description: "Install jargo with go get. The default build needs Go 1.27 and no C toolchain; the ONNX Runtime, RNNoise and libmoonshine are loaded at run time."
weight: 1
---

# Installation

```sh
go get github.com/gojargo/jargo
```

jargo needs Go 1.27 or newer.

## The default build needs no C toolchain

```sh
CGO_ENABLED=0 go build ./...
```

That works out of the box. Opus encode/decode and resampling are **pure Go**, and
the native runtimes are bound through
[purego](https://github.com/ebitengine/purego), located and loaded at *run* time
from their shared libraries, with nothing required at build time.

This is the property that makes jargo deployable as a single static binary. Keep
it unless you have a measured reason not to.

## Runtime libraries

| Library | Needed for | Env var | Without it |
|---|---|---|---|
| **ONNX Runtime** | Silero VAD, Smart Turn v3, the tone reader and tagger | `JARGO_ONNXRUNTIME_LIB` | Bot still runs; falls back to STT endpointing, loses barge-in. |
| **RNNoise** | Optional input noise reduction | `JARGO_RNNOISE_LIB` | Bot runs without denoising. |
| **libmoonshine** | The Moonshine STT provider (`provider/moonshine`) | `JARGO_MOONSHINE_LIB` | `moonshine.NewSTT` returns `moonshine.ErrLibraryNotAvailable`. |

Each is located by its conventional name on the loader's default search path
when nothing points at it.

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
idiomatic way to make denoising optional.

### libmoonshine

The Moonshine STT provider transcribes on the CPU with Moonshine's native
library, `libmoonshine.so` (`libmoonshine.dylib` on macOS, `moonshine.dll` on
Windows). The library runs the models on its own bundled ONNX Runtime, which has
to sit where the library expects it: the `moonshine-voice` wheel on PyPI ships
both in that layout. On Linux, `make deps-moonshine` unpacks them from the wheel
under `.native/moonshine/` and prints the path to export:

```sh
make deps-moonshine
export JARGO_MOONSHINE_LIB=$PWD/.native/moonshine/moonshine_voice/libmoonshine.so
```

The provider takes the path in its configuration rather than reading the
environment; `JARGO_MOONSHINE_LIB` is the variable the eval harness reads, and
the one to use in your own application:

```go
stt, err := moonshine.NewSTT(moonshine.STTConfig{
    LibraryPath: os.Getenv(moonshine.LibPathEnv), // empty searches the default path
})
```

The models are not in the binary. Each is downloaded once, on first use, from
Moonshine's model CDN into `jargo/moonshine` under the user's cache directory,
and checked against its published size and CRC32C. `STTConfig.CacheDir` moves
the cache, and `STTConfig.ModelPath` loads a model already on disk instead.

## No cgo

There is no C toolchain in the picture at all. The Opus codec and the resampler
are pure Go, and the native runtimes above are loaded at run time rather
than linked, so `CGO_ENABLED=0 go build ./...` is the ordinary build and no
build tag changes that.

## Docker

The published base images bundle the ONNX Runtime and RNNoise, so you do not
have to think about them:

- **`gojargo/jargo-build`**: build stage, with the pinned Go toolchain.
- **`gojargo/jargo`**: distroless runtime, carrying the ONNX Runtime and
  RNNoise. It does not carry libmoonshine.

See **[Deploy with Docker](../deploy/docker.md)** for a copyable Dockerfile.

## Verify

```sh
go run ./examples/echo    # then open http://localhost:8080
```

The echo bot needs no API keys and no ONNX Runtime. If you hear yourself back,
the audio path works. Then continue to
**[Your first bot](your-first-bot.md)**.
