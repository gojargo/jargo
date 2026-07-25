---
title: Deploy with Docker
weight: 1
---

# Deploy with Docker

The default build is cgo-free, but a bot still uses native libraries at run
time — the ONNX Runtime and RNNoise are loaded through purego, and the optional
`-tags libsoxr` / `-tags libopus` builds link C libraries. Rather than a single
image, two **base images** take the native-dependency pain out of containerising
a bot:

| Image | Purpose |
| --- | --- |
| `gojargo/jargo-build` | **Build base** — the Go toolchain plus the cgo dev libraries for the optional `-tags libsoxr` / `-tags libopus` builds (libsoxr, libopus, pkg-config). Compile your bot here. |
| `gojargo/jargo` | **Runtime base** — a [distroless](https://github.com/GoogleContainerTools/distroless) image (no shell, no package manager, non-root) carrying the native runtime libraries: the ONNX Runtime and RNNoise (loaded via purego), plus libsoxr, libgomp and libopus for the `-tags` builds. Ship your bot here. |

Both live on [Docker Hub](https://hub.docker.com/u/gojargo). amd64 only for now.

## Build and run a bot image

A two-stage Dockerfile — compile on the build base, ship on the runtime base:

```dockerfile
# syntax=docker/dockerfile:1
FROM gojargo/jargo-build AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# -tags libopus links the C Opus encoder (optional; default is pure-Go SILK).
RUN go build -tags libopus -ldflags="-s -w" -o /out/bot ./path/to/your/bot

FROM gojargo/jargo
COPY --from=build /out/bot /usr/local/bin/bot
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/bot"]
```

```sh
docker build -t my-bot .
docker run --rm -p 8080:8080 \
  -e DEEPGRAM_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  -e ELEVENLABS_API_KEY=... \
  my-bot
```

The runtime base sets `JARGO_ONNXRUNTIME_LIB` and `JARGO_RNNOISE_LIB`, so VAD,
turn detection and noise reduction work with no extra configuration.

## Try an example bot

Point the build at one of jargo's [examples](../../examples), from a jargo
checkout:

```dockerfile
# syntax=docker/dockerfile:1
FROM gojargo/jargo-build AS build
WORKDIR /src
COPY . .
ARG EXAMPLE=voicebot
RUN go build -tags libopus -ldflags="-s -w" -o /out/bot ./examples/${EXAMPLE}

FROM gojargo/jargo
COPY --from=build /out/bot /usr/local/bin/bot
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/bot"]
```

Build the echo bot (no API keys) instead with `--build-arg EXAMPLE=echo`.

## Notes

- **Distroless & non-root.** The runtime base has no shell and runs as an
  unprivileged user, so there is nothing to exec into and the attack surface is
  small. There is no `HEALTHCHECK` (no shell); health-check at the orchestrator
  instead — e.g. a Kubernetes HTTP liveness probe on `:8080`.
- **Pin for production.** Pin the base images to a released tag
  (`gojargo/jargo:0.0.2`) or a digest rather than `latest`.
- **Architecture.** amd64 only today; arm64 is planned.
