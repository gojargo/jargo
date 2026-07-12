# AGENTS.md

Guidance for coding agents (and humans) working in this repository. Keep it
short; the README and `docs/` cover usage in depth.

## What this is

jargo is a WebRTC-native, audio-first conversational-AI framework for Go — a
port of [Pipecat](https://github.com/pipecat-ai/pipecat). Library code lives at
the module root; runnable bots live in `examples/`.

## Build, test, lint

The default build is **cgo-free**: `CGO_ENABLED=0 go build ./...` works. The
resampler and Opus encoder are pure Go, and the native runtimes (ONNX Runtime
for VAD/turn, RNNoise for denoising) are bound through `ebitengine/purego` —
loaded at run time from their shared libraries, with no C toolchain at build
time. The only cgo in the tree is the optional, higher-quality libsoxr and
libopus backends, compiled in only when you build with their tag:

```sh
sudo apt-get install -y libsoxr-dev   # for -tags libsoxr
sudo apt-get install -y libopus-dev   # for -tags libopus
```

- `go build ./...` — build everything with the pure-Go / purego defaults.
- `go test ./...` — run tests. Add `-race` as CI does.
- `go build -tags libsoxr ./...` — opt into libsoxr resampling (SoX Resampler,
  highest quality; the default is the pure-Go `github.com/gojargo/go-resample`
  converter). Needs cgo.
- `go build -tags libopus ./...` — opt into the C Opus encoder (better speech;
  the default is a pure-Go encoder). Needs cgo.
- The **ONNX Runtime** and **RNNoise** shared libraries are located at run time.
  Point at non-standard installs with `JARGO_ONNXRUNTIME_LIB` and
  `JARGO_RNNOISE_LIB`.

Formatting and lint are enforced by **golangci-lint** (`.golangci.yml`):

- `golangci-lint run` — the full linter set; CI fails on any finding.
- `golangci-lint fmt` — apply the configured formatters (`gofmt -s`, `gofumpt`,
  `goimports`). Code must be clean under all three.

## Conventions

- **No upstream references in code.** Keep Pipecat/Python out of `.go` comments;
  attribution lives only in `LICENSE`, `NOTICE`, and `README.md`. No per-file
  copyright headers.
- **Config is explicit.** Library packages read no environment variables: take a
  plain `Config` struct and validate it with go-playground `validate` tags.
  Prefer structs over functional options. Env/flags/Viper belong in the app
  (see `examples/`), not the library.
- **Commits** follow Conventional Commits (`feat:`, `fix:`, `ci:`, `docs:`, …).
- Record notable changes in [`CHANGELOG.md`](CHANGELOG.md). The project is in
  `0.0.x`: the public API is unstable and may change in any release.

## Layout

- `frames/`, `pipeline/`, `processor/` — the streaming engine.
- `transport/` — Pion WebRTC, plus WebSocket/Twilio.
- `service/` + `provider/` — STT/LLM/TTS/S2S interfaces and their providers.
- `turns/`, `audio/` — turn-taking (VAD + Smart Turn) and audio handling.
- `rtvi/`, `aggregators/`, `metrics/`, `tracing/` — RTVI, context, observability.

## Security

Report vulnerabilities privately — see [`SECURITY.md`](SECURITY.md).
