# syntax=docker/dockerfile:1
#
# jargo build base image.
#
# It carries the Go toolchain plus the native development libraries a cgo jargo
# application links against — libsoxr (resampling) and libopus (the -tags
# libopus Opus encoder) — so a downstream image can compile a bot without
# installing anything. Pair it with the distroless runtime base (docker/runtime
# .Dockerfile, published as ghcr.io/gojargo/jargo):
#
#   FROM ghcr.io/gojargo/jargo-build AS build
#   WORKDIR /src
#   COPY go.mod go.sum ./
#   RUN go mod download
#   COPY . .
#   RUN go build -tags libopus -ldflags="-s -w" -o /out/bot ./cmd/bot
#
#   FROM ghcr.io/gojargo/jargo            # distroless runtime base
#   COPY --from=build /out/bot /usr/local/bin/bot
#   ENTRYPOINT ["/usr/local/bin/bot"]

FROM golang:1.26-bookworm@sha256:b305420a68d0f229d91eb3b3ed9e519fcf2cf5461da4bef997bf927e8c0bfd2b

# libsoxr-dev + libopus-dev are the cgo link targets; pkg-config is how cgo's
# #cgo pkg-config directives locate them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsoxr-dev libopus-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# CGO is mandatory for jargo — libsoxr is linked, not optional.
ENV CGO_ENABLED=1
WORKDIR /src
