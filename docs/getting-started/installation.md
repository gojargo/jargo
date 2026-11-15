---
title: Installation
description: "Install jargo with go get. The default build needs Go 1.27 and no C toolchain; the ONNX Runtime, RNNoise, libmoonshine, libespeak-ng and the video codecs are loaded at run time."
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
| **libespeak-ng** | The local Kokoro TTS provider (`provider/kokoro`), which also needs the ONNX Runtime | `JARGO_ESPEAK_LIB` | `kokoro.NewTTS` returns `kokoro.ErrEspeakNotAvailable`. |
| **libvpx** | Video over the WebRTC transport (`transport/rtc`): VP8 in both directions | none | No VP8 video is received or sent; an error is logged once per track. |
| **libopenh264** | Video over the WebRTC transport: H.264 sent by the client | none | No H.264 video is received; an error is logged once per track. |

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

### libespeak-ng

The local Kokoro TTS provider runs the Kokoro model on the ONNX Runtime above,
and phonemizes the text with espeak-ng first. Its library is a distribution
package, so there is no `make` target for it; install the runtime package, which
brings the espeak-ng data with it:

```sh
sudo apt-get install -y libespeak-ng1     # or your system's espeak-ng package
```

It is found by its conventional names (`libespeak-ng.so.1`, then
`libespeak-ng.so`) on the loader's default search path. The provider takes
another path in its configuration rather than reading the environment;
`JARGO_ESPEAK_LIB` is the variable the eval harness and the `voice-kokoro`
example read, and the one to use in your own application:

```go
tts, err := kokoro.NewTTS(kokoro.Config{
    Voice:  "af_heart",
    Espeak: kokoro.EspeakConfig{LibPath: os.Getenv(kokoro.EspeakLibPathEnv)},
})
```

`EspeakConfig.DataPath` points at an espeak-ng data directory when the library
was not built to find its own.

The model is not in the binary either. The Kokoro v1.0 model and voices files
(about 350 MB together) are downloaded once, on first use, into
`jargo/kokoro-onnx` under the user's cache directory. `Config.CacheDir` moves
the cache, and `Config.ModelPath` and `Config.VoicesPath` load files already on
disk instead.

### Video codecs

The WebRTC transport decodes the video a client sends (its camera and its
screen share) and encodes the video the bot shows. VP8 goes through libvpx and
H.264 through OpenH264, both distribution packages, so there is no `make`
target for them either:

```sh
sudo apt-get install -y libvpx12 libopenh264-8   # or your release's versions
```

They are found by their conventional names on the loader's default search path:
`libvpx.so.12`, then the older `libvpx.so.11`, `libvpx.so.9` and so on, and
`libopenh264.so.8`, then `libopenh264.so.7` and so on. Neither is loaded until
video needs it, so an audio-only bot runs without them. The transport takes
other paths in its configuration rather than reading the environment:

```go
conn, err := rtc.NewConnection(rtc.WithVideoCodecs(rtc.VideoCodecConfig{
    VPXLibPath:      "/opt/vpx/lib/libvpx.so.12",
    OpenH264LibPath: "/opt/openh264/lib/libopenh264.so.8",
}))
```

`rtc.RequestHandlerConfig.VideoCodecs` does the same for the connections a
request handler creates.

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
  RNNoise. It does not carry libmoonshine, libespeak-ng or the video codecs.

See **[Deploy with Docker](../deploy/docker.md)** for a copyable Dockerfile.

## Verify

```sh
go run ./examples/echo    # then open http://localhost:8080
```

The echo bot needs no API keys and no ONNX Runtime. If you hear yourself back,
the audio path works. Then continue to
**[Your first bot](your-first-bot.md)**.
