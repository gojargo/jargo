# Deploy with Docker

jargo is cgo — it links libsoxr and libopus and loads the ONNX Runtime — so
rather than a single image, jargo publishes two **base images** that take the
native-dependency pain out of containerising a bot:

| Image | Purpose |
| --- | --- |
| `ghcr.io/gojargo/jargo-build` | **Build base** — the Go toolchain plus the cgo dev libraries (libsoxr, libopus, pkg-config). Compile your bot here. |
| `ghcr.io/gojargo/jargo` | **Runtime base** — a [distroless](https://github.com/GoogleContainerTools/distroless) image (no shell, no package manager, non-root) carrying only the native runtime libraries (libsoxr, libgomp, libopus) and the ONNX Runtime. Ship your bot here. |

Both are mirrored on Docker Hub as `gojargo/jargo-build` and `gojargo/jargo`.
amd64 only for now.

## Build and run a bot image

A two-stage Dockerfile — compile on the build base, ship on the runtime base:

```dockerfile
# syntax=docker/dockerfile:1
FROM ghcr.io/gojargo/jargo-build AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# -tags libopus links the higher-quality Opus encoder (optional).
RUN go build -tags libopus -ldflags="-s -w" -o /out/bot ./path/to/your/bot

FROM ghcr.io/gojargo/jargo
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

The runtime base sets `JARGO_ONNXRUNTIME_LIB`, so VAD and turn detection work
with no extra configuration.

## Try an example bot

Point the build at one of jargo's [examples](../examples), from a jargo
checkout:

```dockerfile
# syntax=docker/dockerfile:1
FROM ghcr.io/gojargo/jargo-build AS build
WORKDIR /src
COPY . .
ARG EXAMPLE=voicebot
RUN go build -tags libopus -ldflags="-s -w" -o /out/bot ./examples/${EXAMPLE}

FROM ghcr.io/gojargo/jargo
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
  (`ghcr.io/gojargo/jargo:0.0.2`) or a digest rather than `latest`.
- **Architecture.** amd64 only today; arm64 is planned.
