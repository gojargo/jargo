# AGENTS.md

Guidance for coding agents (and humans) working in this repository. Keep it
short; the README and `docs/` cover usage in depth.

## What this is

jargo is a WebRTC-native, audio-first conversational-AI framework for Go — a
port of [Pipecat](https://github.com/pipecat-ai/pipecat). Library code lives at
the module root; runnable bots live in `examples/`.

## Build, test, lint

The build is **cgo-free**, with no build tag that changes that. The resampler
and the Opus encoder are pure Go, and the native runtimes (ONNX Runtime for
VAD/turn, RNNoise for denoising) are bound through `ebitengine/purego`, loaded
at run time from their shared libraries with no C toolchain at build time.

- `go build ./...` — build everything. `CGO_ENABLED=0 go build ./...` is the
  same build; nothing in the tree links C.
- `go test ./...` — run tests. Add `-race` as CI does.
- The **ONNX Runtime** and **RNNoise** shared libraries are located at run time.
  Point at non-standard installs with `JARGO_ONNXRUNTIME_LIB` and
  `JARGO_RNNOISE_LIB`. The Moonshine STT provider's **libmoonshine** is too; it
  takes its path in its config, which the eval harness fills from
  `JARGO_MOONSHINE_LIB`, and so is the local Kokoro TTS provider's
  **libespeak-ng** (a distribution package, `libespeak-ng1`), from
  `JARGO_ESPEAK_LIB`, and the in-process Piper TTS provider's **libpiper**,
  from `JARGO_PIPER_LIB`.

Formatting and lint are enforced by **golangci-lint** (`.golangci.yml`):

- `golangci-lint run` — the full linter set; CI fails on any finding.
- `golangci-lint fmt` — apply the configured formatters (`gofmt -s`, `gofumpt`,
  `goimports`). Code must be clean under all three.

Everything above is a one-liner and stays here. Anything longer lives in the
[`Makefile`](Makefile), which the workflows call for the steps they share, so a
CI failure can be reproduced locally. `make help` lists the targets; the ones
worth knowing:

- `make build-matrix`: compile with cgo off and on, as CI does.
- `make cover`: the coverage run and its total, the profile CI uploads. It
  measures across the whole module, because a package is often covered by the
  tests of the one that drives it, and measuring package by package reads as
  untested code that is not. Codecov applies the `ignore:` rules in
  `codecov.yml` (`examples/`, generated protobuf), so the total it reports reads
  roughly 9 points above the local one. Break it down with `make cover-func` or
  `make cover-html`.
- `make generate`: regenerate the Riva protobuf clients with pinned tool
  versions. CI runs `make generate-check` and fails if the tree is stale.
- `make docs-check`: fail on a relative link in `docs/` or the README that points
  at nothing (`TestDocsLinks`, also part of `go test ./...`).
- `make deps`, `make deps-onnx`, `make deps-rnnoise`, `make deps-moonshine`,
  `make deps-piper`: install the cgo headers and the native runtimes above. The
  last four need no root: they install under `.native/` and print the
  `JARGO_*_LIB` values to export. `deps-piper` builds libpiper from source and
  needs CMake.
- `make vuln`, `make secrets`: the govulncheck and gitleaks scans CI gates on.
- `make sast`: the Semgrep CE scan; CI reports its findings without gating.

## Conventions

- **No upstream references in code.** Keep Pipecat/Python out of `.go` comments;
  attribution lives only in `LICENSE`, `NOTICE`, and `README.md`. No per-file
  copyright headers. The one exception is the comparison suite: under
  `benchmarks/` and in `docs/benchmarks.md`, Pipecat is what is being measured,
  so it is named there in code, file names and results.
- **Config is explicit.** Library packages read no environment variables: take a
  plain `Config` struct and validate it with go-playground `validate` tags.
  Prefer structs over functional options. Env/flags/Viper belong in the app
  (see `examples/`), not the library. The one exception is `runner`, the
  development runner: it is an application's entry point, so it reads flags,
  the environment and a `.env` file.
- **Faithful design, Go packaging.** Port Pipecat's behaviour, its tests, the
  layer that owns each piece of machinery, and the names of its concepts and
  frames. Do not port its Python module tree: package layout and naming follow
  Go conventions (next section).
- **Commits** follow Conventional Commits (`feat:`, `fix:`, `ci:`, `docs:`, …).
- Record notable changes in [`CHANGELOG.md`](CHANGELOG.md). The project is
  pre-1.0: the public API may still change in any minor release, and a change
  that breaks it says so in its entry and carries a `!` in its commit type.

## Package layout and naming

A Go package name is part of every identifier a user writes, and two packages
with the same name collide in user code. So:

- No `utils`-style grab-bag packages, and no package named like a standard
  library one (`context`, `errors`).
- No two public packages with the same name. Code that serves only one provider
  lives in that provider's package or under `internal/`.
- Package names are short, lower-case and singular, and say what they provide.
- No stutter: an identifier does not repeat its package name (`llm.Service`, not
  `llm.LLMService`). Frame types are the exception and keep their upstream names
  (`frames.TranscriptionFrame`), since users search for them.
- Adding a method to an exported interface breaks every outside implementation.
  Add new capabilities as an optional interface, or as a method with a default
  in an embedded base type.

## Layout

- `frames/`, `pipeline/`, `processor/`: the streaming engine.
- `transport/`: Pion WebRTC, plus WebSocket/telephony and LiveKit.
- `service/` + `provider/`: STT/LLM/TTS/S2S interfaces and their providers.
- `classifier/`: typed yes/no, choice and score questions about a state,
  answered through an LLM service, Jev, or Google's or Zoom's Decisions API.
- `audio/`: Opus, resampling, mixing, VAD and Smart Turn analyzers.
- `processor/turns`, `processor/aggregators`, `processor/rtvi`: turn-taking,
  conversation context, RTVI.
- `observers/`, `telemetry/`: frame observers, metrics and tracing.
- `docs/`: the documentation sources. The site at https://gojargo.dev
  (repository `gojargo/gojargo.dev`) imports them at build time. Edit `docs/`,
  never a copy.

## Security

Report vulnerabilities privately — see [`SECURITY.md`](SECURITY.md).
