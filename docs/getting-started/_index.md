---
title: Getting started
weight: 1
---

# Getting started

```sh
go get github.com/gojargo/jargo
go run ./examples/echo      # then open http://localhost:8080
```

The echo bot needs no API keys and no native libraries. If you hear yourself back,
the audio path works.

- **[Installation](installation.md)** — the cgo-free default build, the two
  optional runtime libraries, and the `libsoxr` / `libopus` build tags.
- **[Quickstart](quickstart.md)** — run the example bots, locally or with Docker.
- **[Your first bot](your-first-bot.md)** — the full STT → LLM → TTS pipeline, built
  up line by line.

Then read **[Architecture](../concepts/architecture.md)** for the model behind what
you just wired.
