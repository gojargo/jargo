# Contributing to jargo

Thanks for your interest in jargo. Bug reports, fixes, new providers, examples
and documentation are all welcome.

## Before you start

- **Bugs and small fixes**: open a pull request directly, or an issue first if
  you are not sure it is a bug.
- **New features and larger changes**: open an issue first so we can agree on
  the shape before you write the code.
- **Security issues**: do not open a public issue. Follow
  [`SECURITY.md`](SECURITY.md).
- **Questions**: use [GitHub Discussions](https://github.com/gojargo/jargo/discussions)
  rather than an issue.

Issues labelled
[`good first issue`](https://github.com/gojargo/jargo/labels/good%20first%20issue)
are a good place to start.

## Design follows Pipecat

jargo is a Go port of [Pipecat](https://github.com/pipecat-ai/pipecat), and
Pipecat is the reference for its design. When you add a feature or a provider
that Pipecat already has, port what Pipecat does (its behaviour, the names of its
concepts and frames, and the layer it lives in) rather than inventing a new
design, and port its tests alongside the code. Where the Go version has to
differ, say why in the pull request.

Package layout is the exception: it follows Go conventions, not Pipecat's Python
module tree. See [Package layout and naming](AGENTS.md#package-layout-and-naming)
in `AGENTS.md`.

Keep Pipecat and Python out of `.go` comments and commit messages, though:
attribution lives only in `LICENSE`, `NOTICE` and the README.

## Setting up

You need a recent Go (see `go.mod`) and nothing else to build: the default build
is cgo-free.

```sh
git clone https://github.com/gojargo/jargo
cd jargo
go build ./...
go test ./...
```

To run a voice bot with turn-taking you also need the ONNX Runtime;
`make deps-onnx` installs it under `.native/` without root. See
[Installation](docs/getting-started/installation.md) and the
[Quickstart](docs/getting-started/quickstart.md).

## Before you open a pull request

CI runs the same checks, and fails on any of them:

```sh
go test -race ./...     # or: make test
golangci-lint run       # the full linter set
golangci-lint fmt       # gofmt -s, gofumpt and goimports
make docs-check         # no broken relative links in docs/ or the README
```

`make help` lists the other targets. [`AGENTS.md`](AGENTS.md) describes the
build, the layout and the conventions in more detail.

## Conventions

- **Config is explicit.** Library packages read no environment variables. Take a
  plain `Config` struct with go-playground `validate` tags; prefer structs over
  functional options. Environment variables and flags belong in `examples/`.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `ci:`, ...). A change that breaks the public API
  carries a `!` in its type, such as `feat(service)!:`.
- **Changelog.** Record notable changes under `[Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md), and call out breaking changes there.
- **Tests** come with the code they cover. A new provider ships with tests that
  run without network access or API keys.
- **Docs.** The pages in [`docs/`](docs) are also published on
  [gojargo.dev](https://gojargo.dev/docs/). Edit them there, in this repository.

## Adding a provider

A new STT, LLM or TTS provider lives under `provider/<name>/` and implements the
interfaces in `service/`. [Writing a service](docs/extending/custom-service.md)
walks through it. Add the provider to the list in the README, and an example
under `examples/voice/` if it has a voice to hear.

## License

By contributing, you agree that your contributions are licensed under the
project's [BSD 2-Clause License](LICENSE).

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
