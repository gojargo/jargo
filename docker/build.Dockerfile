# syntax=docker/dockerfile:1
#
# jargo build base image.
#
# It carries the Go toolchain at the version jargo is built and tested with.
# jargo links no C library, so the build needs no development headers and no
# cgo. Pair it with the distroless runtime base (docker/runtime.Dockerfile,
# published as gojargo/jargo), which carries the two libraries a bot loads at
# run time:
#
#   FROM gojargo/jargo-build AS build
#   WORKDIR /src
#   COPY go.mod go.sum ./
#   RUN go mod download
#   COPY . .
#   RUN go build -ldflags="-s -w" -o /out/bot ./cmd/bot
#
#   FROM gojargo/jargo            # distroless runtime base
#   COPY --from=build /out/bot /usr/local/bin/bot
#   ENTRYPOINT ["/usr/local/bin/bot"]

FROM golang:1.26-bookworm@sha256:b305420a68d0f229d91eb3b3ed9e519fcf2cf5461da4bef997bf927e8c0bfd2b

# jargo builds without cgo: the native runtimes it uses (the ONNX Runtime and
# RNNoise) are bound with purego and loaded at run time, not linked. The binary
# is still dynamically linked against libc and libdl, because purego reaches
# dlopen through them, which is why the runtime base is distroless/cc (glibc)
# rather than a scratch image.
ENV CGO_ENABLED=0
WORKDIR /src
