# syntax=docker/dockerfile:1
#
# Container image for the jargo example bots. It bundles the two native
# dependencies jargo needs — libsoxr (linked) and the ONNX Runtime (loaded at
# run time for VAD + turn detection) — so the image runs anywhere with no host
# setup. Build with BuildKit/buildx; multi-arch works (amd64 + arm64) because
# each platform builds natively, which is what cgo needs.
#
#   docker build -t jargo-voicebot .
#   docker buildx build --platform linux/amd64,linux/arm64 -t jargo-voicebot .
#
# Override which example to build with --build-arg EXAMPLE=echo.

# ---- build the binary (cgo: needs libsoxr-dev) ----
FROM golang:1.26-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsoxr-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .

ARG EXAMPLE=voicebot
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /out/jargo-bot ./examples/${EXAMPLE}

# ---- fetch the ONNX Runtime shared library for the target arch ----
FROM debian:bookworm-slim AS onnx
ARG ORT_VERSION=1.26.0
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) ort_arch=x64 ;; \
      arm64) ort_arch=aarch64 ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz"; \
    mkdir -p /opt/ort && tar -xzf /tmp/ort.tgz -C /opt/ort --strip-components=1; \
    cp -L /opt/ort/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so

# ---- runtime ----
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libsoxr0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=onnx /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so
RUN ldconfig
ENV JARGO_ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so

COPY --from=build /out/jargo-bot /usr/local/bin/jargo-bot

# The example bots serve their web UI and signaling on :8080.
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/jargo-bot"]
